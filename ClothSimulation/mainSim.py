# -*- coding: utf-8 -*-
"""
+==============================================================================+
|          TAICHI CLOTH SIMULATION - Position Based Dynamics                  |
|          PBD + Wind + Sphere Collision + Phong Shading + Interactivity       |
+==============================================================================+

HOW TO RUN:
    pip install taichi numpy
    python cloth_simulation.py

CONTROLS:
    Mouse drag     - Pull cloth particles
    W              - Toggle wind on/off
    +/-            - Increase/decrease wind strength
    R              - Reset simulation
    SPACE          - Toggle sphere movement
    1/2/3          - Switch material preset (1=silk, 2=cotton, 3=denim)
    ESC            - Quit

PARAMETER TWEAKS FOR DIFFERENT EFFECTS:
    Silk:   stiffness=0.3, bending=0.05, damping=0.99, mass=0.05
    Cotton: stiffness=0.8, bending=0.2,  damping=0.98, mass=0.1
    Denim:  stiffness=1.0, bending=0.5,  damping=0.97, mass=0.2

EXTENSIONS:
    - Self-collision via spatial hashing
    - Tear/cut cloth on high tension
    - Multiple spheres / capsule colliders
    - Export to video with imageio
    - Fluid-cloth coupling
"""

import taichi as ti
import numpy as np
import math
import time

# -----------------------------------------------------------------------------
# Taichi initialization - tries GPU, falls back to CPU
# -----------------------------------------------------------------------------
try:
    ti.init(arch=ti.gpu)
    print("[OK] GPU acceleration enabled")
except Exception:
    ti.init(arch=ti.cpu)
    print("[!] Running on CPU (install CUDA/Metal for GPU)")

# -----------------------------------------------------------------------------
# Simulation constants - change these to tune behaviour
# -----------------------------------------------------------------------------
NX, NY       = 48, 48          # Grid resolution (particles)
REST_LEN     = 1.0 / (NX - 1) # Rest length between adjacent particles
DT           = 0.016           # Time step (~60 fps physics)
SOLVER_ITERS = 12              # Constraint iterations per frame (higher = stiffer)
GRAVITY      = -9.8            # m/s^2 (scaled to cloth size)
N_PARTICLES  = NX * NY

# Material presets - (stiffness, bend_stiffness, damping, mass)
MATERIALS = {
    "silk":   (0.30, 0.05, 0.995, 0.04),
    "cotton": (0.75, 0.20, 0.982, 0.10),
    "denim":  (0.98, 0.50, 0.970, 0.20),
}
#current_material = "cotton"
#current_material = "silk"
current_material = "denim"

# -----------------------------------------------------------------------------
# Taichi fields
# -----------------------------------------------------------------------------
pos      = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY))   # current positions
pos_old  = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY))   # previous positions (Verlet)
vel      = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY))   # velocities
force    = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY))   # accumulated forces
inv_mass = ti.field(dtype=ti.f32, shape=(NX, NY))             # 1/mass (0 = pinned)
normal   = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY))   # per-vertex normals

# Rendering triangles: 2 triangles per quad, (NX-1)*(NY-1)*2 triangles
N_TRIS = (NX - 1) * (NY - 1) * 2
tri_verts  = ti.Vector.field(3, dtype=ti.f32, shape=N_TRIS * 3)
tri_norms  = ti.Vector.field(3, dtype=ti.f32, shape=N_TRIS * 3)
tri_colors = ti.Vector.field(3, dtype=ti.f32, shape=N_TRIS * 3)

# Sphere collider
sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
sphere_radius  = ti.field(dtype=ti.f32, shape=())

# Simulation parameters (mutable from Python side)
stiffness      = ti.field(dtype=ti.f32, shape=())
bend_stiffness = ti.field(dtype=ti.f32, shape=())
damping        = ti.field(dtype=ti.f32, shape=())
mass_val       = ti.field(dtype=ti.f32, shape=())
wind_enabled   = ti.field(dtype=ti.i32, shape=())
wind_strength  = ti.field(dtype=ti.f32, shape=())
sim_time       = ti.field(dtype=ti.f32, shape=())

# -----------------------------------------------------------------------------
# Noise utilities (value noise for wind turbulence, GPU-friendly)
# -----------------------------------------------------------------------------
@ti.func
def hash2(p: ti.template()) -> ti.f32:
    """Fast hash for 2D integer input -> [0,1)."""
    n = p[0] * 127 + p[1] * 311
    n = (n << 13) ^ n
    return ti.cast((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff, ti.f32) / 2147483648.0

@ti.func
def value_noise_2d(x: ti.f32, y: ti.f32) -> ti.f32:
    """Smooth value noise in [0,1]."""
    ix = int(ti.floor(x))
    iy = int(ti.floor(y))
    fx = x - ti.floor(x)
    fy = y - ti.floor(y)
    # Smoothstep
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    a = hash2(ti.Vector([ix,     iy    ]))
    b = hash2(ti.Vector([ix + 1, iy    ]))
    c = hash2(ti.Vector([ix,     iy + 1]))
    d = hash2(ti.Vector([ix + 1, iy + 1]))
    return ti.math.mix(ti.math.mix(a, b, ux),
                       ti.math.mix(c, d, ux), uy)

@ti.func
def fbm_wind(x: ti.f32, y: ti.f32, t: ti.f32) -> ti.f32:
    """Fractal Brownian Motion - 3 octaves of noise for turbulent wind."""
    v  = value_noise_2d(x       + t * 0.3, y       + t * 0.2)
    v += value_noise_2d(x * 2.1 + t * 0.5, y * 2.1 - t * 0.4) * 0.5
    v += value_noise_2d(x * 4.3 - t * 0.7, y * 4.3 + t * 0.6) * 0.25
    return v / 1.75  # normalise to ~[0,1]

# -----------------------------------------------------------------------------
# Simulation kernels
# -----------------------------------------------------------------------------

@ti.kernel
def init_cloth():
    """Lay the cloth flat in the XZ plane, pinning the top edge."""
    for i, j in pos:
        x = (i / (NX - 1) - 0.5)      # [-0.5, 0.5]
        y = 0.5                         # top at y=0.5
        z = (j / (NY - 1) - 0.5)      # [-0.5, 0.5]
        pos[i, j]     = ti.Vector([x, y, z])
        pos_old[i, j] = ti.Vector([x, y, z])
        vel[i, j]     = ti.Vector([0.0, 0.0, 0.0])
        # Pin top corners and top edge
        if j == NY - 1:
            inv_mass[i, j] = 0.0   # pinned
        else:
            inv_mass[i, j] = 1.0 / mass_val[None]

@ti.kernel
def accumulate_forces():
    """
    Compute per-particle forces: gravity + aerodynamic wind.
    Wind uses procedural noise for realistic turbulence.
    """
    t = sim_time[None]
    ws = wind_strength[None]
    for i, j in pos:
        m = 1.0 / inv_mass[i, j] if inv_mass[i, j] > 0.0 else 0.0
        f = ti.Vector([0.0, m * GRAVITY * 0.05, 0.0])  # gravity (scaled)

        # -- Wind force via fBm noise --
        if wind_enabled[None] == 1:
            xi = ti.cast(i, ti.f32) / NX
            xj = ti.cast(j, ti.f32) / NY
            # Main wind direction: +X, turbulence modulates magnitude
            turb = fbm_wind(xi * 3.0, xj * 3.0, t)
            # Gusty bursts: sharp ramps in noise
            gust = ti.max(0.0, turb * 2.0 - 0.7)
            wind_x = ws * (0.5 + turb * 0.8 + gust * 1.5)
            wind_y = ws * (turb - 0.5) * 0.3
            wind_z = ws * fbm_wind(xi * 2.5 + 5.0, xj * 2.5, t) * 0.4
            f += ti.Vector([wind_x, wind_y, wind_z]) * m
        force[i, j] = f

@ti.kernel
def integrate():
    """
    Symplectic Euler integration with velocity damping.
    Stores old positions for constraint projection.
    """
    d = damping[None]
    for i, j in pos:
        if inv_mass[i, j] > 0.0:
            # Update velocity from force
            vel[i, j] += force[i, j] * inv_mass[i, j] * DT
            vel[i, j] *= d   # global damping
            # Save position, advance
            pos_old[i, j] = pos[i, j]
            pos[i, j] += vel[i, j] * DT

@ti.func
def satisfy_distance_constraint(i: int, j: int, ni: int, nj: int,
                                 rest: ti.f32, k: ti.f32):
    """
    PBD distance constraint between particles (i,j) and (ni,nj).
    Projects positions to satisfy |p1-p2| == rest.
    """
    p1 = pos[i, j]
    p2 = pos[ni, nj]
    diff = p1 - p2
    dist = diff.norm() + 1e-8
    correction = (dist - rest) / dist * diff
    w1 = inv_mass[i, j]
    w2 = inv_mass[ni, nj]
    wsum = w1 + w2
    if wsum > 0.0:
        pos[i,  j ] -= k * w1 / wsum * correction
        pos[ni, nj] += k * w2 / wsum * correction

@ti.kernel
def solve_constraints():
    """
    Project structural, shear, and bending constraints.
    Called SOLVER_ITERS times per frame for convergence.
    """
    k_s = stiffness[None]
    k_b = bend_stiffness[None]
    rl  = REST_LEN

    # -- Structural: horizontal + vertical neighbours --
    for i, j in pos:
        if i + 1 < NX:
            satisfy_distance_constraint(i, j, i+1, j, rl, k_s)
        if j + 1 < NY:
            satisfy_distance_constraint(i, j, i, j+1, rl, k_s)

    # -- Shear: diagonal neighbours --
    diag_rest = rl * ti.sqrt(2.0)
    for i, j in pos:
        if i + 1 < NX and j + 1 < NY:
            satisfy_distance_constraint(i, j, i+1, j+1, diag_rest, k_s * 0.7)
        if i + 1 < NX and j - 1 >= 0:
            satisfy_distance_constraint(i, j, i+1, j-1, diag_rest, k_s * 0.7)

    # -- Bending: skip-one neighbours --
    bend_rest = rl * 2.0
    for i, j in pos:
        if i + 2 < NX:
            satisfy_distance_constraint(i, j, i+2, j, bend_rest, k_b)
        if j + 2 < NY:
            satisfy_distance_constraint(i, j, i, j+2, bend_rest, k_b)

@ti.kernel
def solve_collisions():
    """
    Sphere collision: push particles outside the sphere.
    Also apply ground plane collision at y=-0.6.
    """
    sc = sphere_center[0]
    sr = sphere_radius[None]
    friction = 0.85  # friction coefficient

    for i, j in pos:
        p = pos[i, j]

        # -- Sphere --
        diff = p - sc
        dist = diff.norm()
        if dist < sr + 0.005:
            # Push out to surface + small offset
            n = diff.normalized()
            pos[i, j] = sc + n * (sr + 0.005)
            # Zero out velocity component into sphere
            vn = vel[i, j].dot(n)
            if vn < 0.0:
                vel[i, j] -= vn * n        # remove inward component
                vel[i, j] *= friction      # friction on tangential

        # -- Ground plane --
        if pos[i, j].y < -0.55:
            pos[i, j].y = -0.55
            if vel[i, j].y < 0.0:
                vel[i, j].y = 0.0
            vel[i, j].x *= friction
            vel[i, j].z *= friction

@ti.kernel
def update_velocities():
    """
    After all constraint solving, recompute velocities from
    position change (standard PBD velocity update).
    """
    inv_dt = 1.0 / DT
    for i, j in pos:
        if inv_mass[i, j] > 0.0:
            vel[i, j] = (pos[i, j] - pos_old[i, j]) * inv_dt

@ti.kernel
def compute_normals():
    """
    Compute smooth per-vertex normals by averaging face normals
    of all adjacent triangles. Used for Phong shading.
    """
    for i, j in normal:
        normal[i, j] = ti.Vector([0.0, 0.0, 0.0])

    # Accumulate face normals onto vertices
    for i, j in ti.ndrange(NX - 1, NY - 1):
        p00 = pos[i,   j  ]
        p10 = pos[i+1, j  ]
        p01 = pos[i,   j+1]
        p11 = pos[i+1, j+1]
        # Triangle 1: 00, 10, 01
        e1 = p10 - p00; e2 = p01 - p00
        n1 = e1.cross(e2)
        # Triangle 2: 11, 01, 10
        e3 = p01 - p11; e4 = p10 - p11
        n2 = e3.cross(e4)
        normal[i,   j  ] += n1
        normal[i+1, j  ] += n1
        normal[i,   j+1] += n1 + n2
        normal[i+1, j+1] += n2
        normal[i,   j+1] += n2  # already added but fine (avg)
        normal[i+1, j  ] += n2

    # Normalise
    for i, j in normal:
        n = normal[i, j]
        nl = n.norm()
        if nl > 1e-8:
            normal[i, j] = n / nl
        else:
            normal[i, j] = ti.Vector([0.0, 1.0, 0.0])

@ti.kernel
def build_render_mesh(
    light_dir_x: ti.f32, light_dir_y: ti.f32, light_dir_z: ti.f32,
    cloth_r: ti.f32, cloth_g: ti.f32, cloth_b: ti.f32
):
    """
    Build triangle mesh for GGUI rendering with per-vertex
    Phong shading (diffuse + ambient + specular highlight).
    cloth_r/g/b is the base cloth colour.
    """
    light = ti.Vector([light_dir_x, light_dir_y, light_dir_z]).normalized()
    ambient   = 0.18
    specular_k = 0.4
    shininess  = 32.0
    # Eye direction (camera looks from +Z toward origin)
    eye_dir = ti.Vector([0.0, 0.2, 1.0]).normalized()

    for i, j in ti.ndrange(NX - 1, NY - 1):
        tri_base = (i * (NY - 1) + j) * 6  # 2 tris x 3 verts = 6 entries

        # Triangle 1 indices: (i,j), (i+1,j), (i,j+1)
        # Triangle 2 indices: (i+1,j+1), (i,j+1), (i+1,j)
        ii  = [i,   i+1, i,   i+1, i,   i+1]
        jj  = [j,   j,   j+1, j+1, j+1, j  ]

        for k in ti.static(range(6)):
            vi = ii[k]; vj = jj[k]
            p  = pos[vi, vj]
            n  = normal[vi, vj]

            # -- Phong shading --
            diff = ti.max(0.0, n.dot(light))

            # Specular (Blinn-Phong half-vector)
            half_vec = (light + eye_dir).normalized()
            spec = ti.pow(ti.max(0.0, n.dot(half_vec)), shininess) * specular_k

            # Back-face contribution (cloth is two-sided)
            diff_back = ti.max(0.0, (-n).dot(light)) * 0.4

            shading = ambient + diff + diff_back + spec

            # Subtle checkerboard weave texture pattern (UV-based)
            u = ti.cast(vi, ti.f32) / (NX - 1)
            v = ti.cast(vj, ti.f32) / (NY - 1)
            # Weave: fine grid oscillation
            weave = 0.5 + 0.5 * ti.sin(u * NX * math.pi * 2) * ti.sin(v * NY * math.pi * 2)
            # Subtle fabric lustre variation
            luster = 1.0 + 0.08 * (weave - 0.5)

            r = ti.min(1.0, cloth_r * shading * luster)
            g = ti.min(1.0, cloth_g * shading * luster)
            b = ti.min(1.0, cloth_b * shading * luster)

            tri_verts [tri_base + k] = p
            tri_norms [tri_base + k] = n
            tri_colors[tri_base + k] = ti.Vector([r, g, b])

# -----------------------------------------------------------------------------
# Python-level helpers
# -----------------------------------------------------------------------------

def set_material(name: str):
    """Apply a material preset to the simulation fields."""
    k_s, k_b, d, m = MATERIALS[name]
    stiffness[None]      = k_s
    bend_stiffness[None] = k_b
    damping[None]        = d
    mass_val[None]       = m

def reset_sim():
    """Full reset: reinitialise all cloth particle state."""
    set_material(current_material)
    sphere_center[0] = ti.Vector([0.0, 0.0, 0.0])
    sphere_radius[None] = 0.18
    sim_time[None]      = 0.0
    init_cloth()
    print(f"[RESET]  Simulation reset - material: {current_material}")

def apply_mouse_force(screen_xy, window_size):
    """
    Map 2D mouse position to cloth-space and apply impulse
    to nearby cloth particles.
    """
    mx = screen_xy[0] / window_size[0] - 0.5   # normalised -0.5…0.5
    my = -(screen_xy[1] / window_size[1] - 0.5)
    pos_np = pos.to_numpy()
    vel_np = vel.to_numpy()
    inv_np = inv_mass.to_numpy()

    # Project: cloth is in XY plane (ignoring Z depth for simplicity)
    RADIUS = 0.08
    STRENGTH = 2.5
    for i in range(NX):
        for j in range(NY):
            px = pos_np[i, j, 0]
            py = pos_np[i, j, 1]
            dx = mx - px
            dy = my - py
            d2 = dx * dx + dy * dy
            if d2 < RADIUS * RADIUS and inv_np[i, j] > 0.0:
                fac = STRENGTH * (1.0 - math.sqrt(d2) / RADIUS)
                vel_np[i, j, 0] += dx * fac
                vel_np[i, j, 1] += dy * fac

    vel.from_numpy(vel_np)

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main():
    global current_material

    # -- Window & camera --
    WINDOW_W, WINDOW_H = 1200, 800
    window  = ti.ui.Window("Taichi Cloth Simulation", (WINDOW_W, WINDOW_H),
                            vsync=True)
    canvas  = window.get_canvas()
    gui     = window.get_gui()
    scene   = window.get_scene()
    camera  = ti.ui.Camera()

    # Camera orbit parameters
    cam_theta = 0.3   # azimuth (rad)
    cam_phi   = 0.3   # elevation (rad)
    cam_dist  = 1.6
    cam_target = [0.0, 0.0, 0.0]

    # -- Initial state --
    wind_enabled[None]  = 1
    wind_strength[None] = 0.06
    reset_sim()

    # -- Sphere animation --
    sphere_moving = True
    sphere_t      = 0.0

    # -- Lighting direction --
    LIGHT_DIR = [0.6, 1.0, 0.5]  # from upper-right-front

    # -- Cloth colour (cotton beige) --
    CLOTH_COLORS = {
        "silk":   (0.85, 0.75, 0.80),   # dusty rose
        "cotton": (0.78, 0.72, 0.60),   # warm beige
        "denim":  (0.25, 0.38, 0.58),   # blue denim
    }

    frame = 0
    t0    = time.time()
    fps_display = 60.0

    # -- Mouse drag state --
    last_mouse = None

    print("\n=== Cloth Simulation Running ===")
    print("W: toggle wind | +/-: wind strength | R: reset")
    print("SPACE: toggle sphere | 1/2/3: silk/cotton/denim")
    print("Mouse drag: pull cloth | ESC: quit\n")

    while window.running:
        frame += 1

        # -- FPS counter --
        if frame % 60 == 0:
            elapsed = time.time() - t0
            fps_display = 60.0 / elapsed
            t0 = time.time()

        # ---------------------------------------------
        # Input handling
        # ---------------------------------------------
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == 'r':
                reset_sim()
            elif e.key == ' ':
                sphere_moving = not sphere_moving
                print(f"Sphere: {'moving' if sphere_moving else 'stopped'}")
            elif e.key == 'w':
                wind_enabled[None] = 1 - wind_enabled[None]
                print(f"Wind: {'ON' if wind_enabled[None] else 'OFF'}")
            elif e.key == '=':  # + key
                wind_strength[None] = min(0.3, wind_strength[None] + 0.01)
                print(f"Wind strength: {wind_strength[None]:.2f}")
            elif e.key == '-':
                wind_strength[None] = max(0.0, wind_strength[None] - 0.01)
                print(f"Wind strength: {wind_strength[None]:.2f}")
            elif e.key == '1':
                current_material = "silk";   set_material("silk");   reset_sim()
            elif e.key == '2':
                current_material = "cotton"; set_material("cotton"); reset_sim()
            elif e.key == '3':
                current_material = "denim";  set_material("denim");  reset_sim()

        # -- Mouse drag --
        if window.is_pressed(ti.ui.LMB):
            mp = window.get_cursor_pos()
            apply_mouse_force((mp[0] * WINDOW_W, mp[1] * WINDOW_H),
                               (WINDOW_W, WINDOW_H))

        # -- Camera orbit (right-drag) --
        if window.is_pressed(ti.ui.RMB):
            mp = window.get_cursor_pos()
            if last_mouse is not None:
                dx = mp[0] - last_mouse[0]
                dy = mp[1] - last_mouse[1]
                cam_theta += dx * 3.0
                cam_phi    = max(-1.2, min(1.2, cam_phi + dy * 2.0))
            last_mouse = mp
        else:
            last_mouse = None

        # ---------------------------------------------
        # Physics step
        # ---------------------------------------------
        sim_time[None] += DT

        # Animate sphere (figure-8 path)
        if sphere_moving:
            sphere_t += DT * 0.5
            sx = 0.22 * math.sin(sphere_t)
            sy = -0.15 + 0.10 * math.sin(sphere_t * 1.3)
            sz = 0.15 * math.sin(sphere_t * 0.7)
            sphere_center[0] = ti.Vector([sx, sy, sz])

        accumulate_forces()
        integrate()
        for _ in range(SOLVER_ITERS):
            solve_constraints()
            solve_collisions()
        update_velocities()

        # ---------------------------------------------
        # Rendering
        # ---------------------------------------------
        compute_normals()
        cr, cg, cb = CLOTH_COLORS[current_material]
        build_render_mesh(
            LIGHT_DIR[0], LIGHT_DIR[1], LIGHT_DIR[2],
            cr, cg, cb
        )

        # Camera position from spherical coords
        cx = cam_dist * math.cos(cam_phi) * math.sin(cam_theta)
        cy = cam_dist * math.sin(cam_phi) + 0.1
        cz = cam_dist * math.cos(cam_phi) * math.cos(cam_theta)
        camera.position(cx, cy, cz)
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])
        camera.up(0, 1, 0)
        camera.fov(45)

        scene.set_camera(camera)
        scene.ambient_light([0.12, 0.12, 0.14])
        scene.point_light(pos=(LIGHT_DIR[0]*2, LIGHT_DIR[1]*2, LIGHT_DIR[2]*2),
                          color=(1.0, 0.97, 0.92))

        # Draw cloth mesh
        scene.mesh(tri_verts, indices=None,
                   per_vertex_color=tri_colors,
                   two_sided=True)

        # Draw sphere collider (subtle wire)
        scene.particles(sphere_center, radius=sphere_radius[None],
                        color=(0.55, 0.55, 0.60))

        canvas.scene(scene)

        # -- HUD overlay (text) --
        with gui.sub_window("HUD", 0.01, 0.01, 0.98, 0.12):
            gui.text(
                (
                    f"FPS: {fps_display:.0f}  |  Material: {current_material.upper()}"
                    f"  |  Wind: {'ON' if wind_enabled[None] else 'OFF'} ({wind_strength[None]:.2f})"
                    f"  |  Drag with LMB  |  Orbit with RMB"
                ),
                color=(0.9, 0.9, 0.9),
            )

        window.show()

if __name__ == "__main__":
    main()