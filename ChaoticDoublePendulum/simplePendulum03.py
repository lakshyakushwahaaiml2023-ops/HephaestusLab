"""
3D SPHERICAL PENDULUM — Front + Top View
=========================================
CONFIG:
  THETA0_DEG  : initial polar tilt from vertical
  PHI0_DEG    : azimuthal start angle
  OMEGA_PHI0  : spin rate (gives 3D motion)

Controls: R = reset   ESC = quit
"""

import taichi as ti
import math

# ─── CONFIG ─────────────────────────────────────────────
THETA0_DEG  = 55.0
PHI0_DEG    = 35.0
OMEGA_PHI0  = 1.8
# ────────────────────────────────────────────────────────

ti.init(arch=ti.cpu)

# Canvas: two square panels side by side
PW, PH    = 700, 700       # each panel size
W, H      = PW * 2, PH     # total window
TRAIL_LEN = 800
G         = 9.81
L         = 0.28
DT        = 3e-4
STEPS     = 60
SCALE     = int(PH * 1.4)  # world → pixels

# Front panel pivot
FP_X = PW // 2
FP_Y = int(PH * 0.30)

# Top panel pivot (centre of right half)
TP_X = PW + PW // 2
TP_Y = PH // 2

# ── Fields ───────────────────────────────────────────────
theta  = ti.field(ti.f64, shape=())
phi    = ti.field(ti.f64, shape=())
dtheta = ti.field(ti.f64, shape=())
dphi   = ti.field(ti.f64, shape=())
tip    = ti.Vector.field(3, ti.f32, shape=())

trail  = ti.Vector.field(3, ti.f32, shape=TRAIL_LEN)
t_head = ti.field(ti.i32, shape=())
t_cnt  = ti.field(ti.i32, shape=())

img    = ti.Vector.field(3, ti.f32, shape=(W, H))

# ── Init ─────────────────────────────────────────────────
def init():
    theta[None]  = math.radians(THETA0_DEG)
    phi[None]    = math.radians(PHI0_DEG)
    dtheta[None] = 0.0
    dphi[None]   = OMEGA_PHI0
    t_head[None] = 0
    t_cnt[None]  = 0

# ── Physics ───────────────────────────────────────────────
@ti.func
def eom(th: ti.f64, ph: ti.f64, dth: ti.f64, dph: ti.f64):
    sth = ti.sin(th)
    ddth = sth * ti.cos(th) * dph * dph - (G / L) * sth
    ddph = ti.f64(0.0)
    if ti.abs(sth) > 1e-10:
        ddph = -2.0 * dth * dph * ti.cos(th) / sth
    return dth, dph, ddth, ddph

@ti.kernel
def step():
    th = theta[None]; ph = phi[None]
    dth = dtheta[None]; dph = dphi[None]

    a1,b1,c1,d1 = eom(th,ph,dth,dph)
    a2,b2,c2,d2 = eom(th+.5*DT*a1, ph+.5*DT*b1, dth+.5*DT*c1, dph+.5*DT*d1)
    a3,b3,c3,d3 = eom(th+.5*DT*a2, ph+.5*DT*b2, dth+.5*DT*c2, dph+.5*DT*d2)
    a4,b4,c4,d4 = eom(th+DT*a3, ph+DT*b3, dth+DT*c3, dph+DT*d3)

    theta[None]  = th  + (DT/6.)*(a1+2*a2+2*a3+a4)
    phi[None]    = ph  + (DT/6.)*(b1+2*b2+2*b3+b4)
    dtheta[None] = dth + (DT/6.)*(c1+2*c2+2*c3+c4)
    dphi[None]   = dph + (DT/6.)*(d1+2*d2+2*d3+d4)

    th2 = theta[None]; ph2 = phi[None]
    tip[None] = ti.Vector([
        ti.cast(L * ti.sin(th2) * ti.cos(ph2), ti.f32),
        ti.cast(L * (1.0 - ti.cos(th2)),       ti.f32),
        ti.cast(L * ti.sin(th2) * ti.sin(ph2), ti.f32),
    ])

@ti.kernel
def record():
    h = t_head[None]
    trail[h] = tip[None]
    t_head[None] = (h + 1) % TRAIL_LEN
    if t_cnt[None] < TRAIL_LEN:
        t_cnt[None] += 1

# ── Render ────────────────────────────────────────────────
@ti.kernel
def render():
    sc = ti.cast(SCALE, ti.f32)

    # ─────────────────────────────────────────────────────
    # Background — two distinct but harmonious dark themes
    # ─────────────────────────────────────────────────────
    for px, py in img:
        ny  = ti.cast(py, ti.f32) / ti.cast(H, ti.f32)
        lx  = ti.cast(px % PW, ti.f32) / ti.cast(PW, ti.f32)
        vig = (lx - 0.5)*(lx - 0.5)*1.2 + (ny - 0.5)*(ny - 0.5)*0.8

        # 0.0 = front panel, 1.0 = top panel  (no branching)
        t_panel = ti.cast(ti.min(px // PW, 1), ti.f32)

        # Front colour  (navy blue)
        bg_front = ti.Vector([0.03, 0.05, 0.11]) * (1.0 - vig * 0.55) \
                 + ti.Vector([0.0,  0.005, 0.02]) * ny
        # Top colour  (slate teal)
        bg_top   = ti.Vector([0.03, 0.06, 0.09]) * (1.0 - vig * 0.55) \
                 + ti.Vector([0.0,  0.01,  0.01]) * (1.0 - ny)

        img[px, py] = bg_front * (1.0 - t_panel) + bg_top * t_panel

    # ─────────────────────────────────────────────────────
    # Divider — glowing seam between panels
    # ─────────────────────────────────────────────────────
    for py in range(H):
        for off in range(-2, 3):
            px_ = PW + off
            if 0 <= px_ < W:
                strength = 1.0 - ti.cast(ti.abs(off), ti.f32) / 3.0
                img[px_, py] = img[px_, py] + ti.Vector([0.06, 0.14, 0.28]) * strength

    cnt  = t_cnt[None]
    head = t_head[None]
    tx = tip[None][0]; ty = tip[None][1]; tz = tip[None][2]

    # ═════════════════════════════════════════════════════
    # PANEL 0 — FRONT VIEW  (X horizontal, Y down)
    # ═════════════════════════════════════════════════════

    # ── Front trail ─────────────────────────────────────
    for j in range(cnt - 1):
        ia = (head - 1 - j + TRAIL_LEN) % TRAIL_LEN
        ib = (head - 2 - j + TRAIL_LEN) % TRAIL_LEN
        pa = trail[ia]; pb = trail[ib]

        fade = 1.0 - ti.cast(j, ti.f32) / ti.cast(cnt, ti.f32)
        fade = fade * fade
        z_n  = ((pa[2] + pb[2]) * 0.5 / L + 1.0) * 0.5
        col  = ti.Vector([0.15 + 0.55*z_n, 0.45 + 0.45*z_n, 1.0]) * (fade * 0.88 + 0.04)

        ax = FP_X + ti.cast(pa[0] * sc, ti.i32)
        ay = FP_Y + ti.cast(pa[1] * sc, ti.i32)
        bx = FP_X + ti.cast(pb[0] * sc, ti.i32)
        by = FP_Y + ti.cast(pb[1] * sc, ti.i32)
        dx = bx - ax; dy = by - ay
        ns = ti.max(ti.abs(dx), ti.abs(dy))
        if ns > 0:
            for s in range(ns + 1):
                t = ti.cast(s, ti.f32) / ti.cast(ns, ti.f32)
                cx = ax + ti.cast(t * dx, ti.i32)
                cy = ay + ti.cast(t * dy, ti.i32)
                if 0 <= cx < PW and 0 <= cy < H:
                    img[cx, cy] = ti.max(img[cx, cy], col)

    # ── Front rod ───────────────────────────────────────
    bob_fx = FP_X + ti.cast(tx * sc, ti.i32)
    bob_fy = FP_Y + ti.cast(ty * sc, ti.i32)
    dx = bob_fx - FP_X; dy = bob_fy - FP_Y
    rl = ti.max(ti.abs(dx), ti.abs(dy))
    if rl > 0:
        for s in range(rl + 1):
            t = ti.cast(s, ti.f32) / ti.cast(rl, ti.f32)
            cx = FP_X + ti.cast(t * dx, ti.i32)
            cy = FP_Y + ti.cast(t * dy, ti.i32)
            thick = ti.cast(3.0 - t * 1.8, ti.i32)
            rod_c = ti.Vector([0.70 + 0.20*t, 0.78 + 0.15*t, 0.90])
            for ox in range(-thick, thick + 1):
                for oy in range(-thick, thick + 1):
                    ppx = cx + ox; ppy = cy + oy
                    if 0 <= ppx < PW and 0 <= ppy < H:
                        e2 = ti.cast(ox*ox + oy*oy, ti.f32)
                        sp = ti.exp(-e2 * 0.35) * (0.5 + 0.5*(1.0-t))
                        img[ppx, ppy] = rod_c * sp + img[ppx, ppy] * (1.0 - sp * 0.7)

    # ── Front pivot ─────────────────────────────────────
    for ox in range(-9, 10):
        for oy in range(-9, 10):
            r2 = ti.cast(ox*ox + oy*oy, ti.f32)
            if r2 <= 81.0:
                ppx = FP_X + ox; ppy = FP_Y + oy
                if 0 <= ppx < PW and 0 <= ppy < H:
                    br = ti.exp(-r2 * 0.025)
                    img[ppx, ppy] = ti.Vector([0.78, 0.83, 0.92]) * br \
                                  + ti.Vector([0.15, 0.25, 0.45]) * (1.0 - br)
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            if ox*ox + oy*oy <= 9:
                ppx = FP_X + ox; ppy = FP_Y + oy
                if 0 <= ppx < PW and 0 <= ppy < H:
                    img[ppx, ppy] = ti.Vector([0.97, 0.98, 1.0])

    # ── Front bob (Phong + depth glow) ──────────────────
    z_n   = (tz / L + 1.0) * 0.5
    bob_r = ti.cast(12.0 + z_n * 5.0, ti.i32)
    glow_r = bob_r + 18
    for ox in range(-glow_r, glow_r + 1):
        for oy in range(-glow_r, glow_r + 1):
            dr = ti.sqrt(ti.cast(ox*ox + oy*oy, ti.f32))
            if dr <= ti.cast(glow_r, ti.f32):
                ppx = bob_fx + ox; ppy = bob_fy + oy
                if 0 <= ppx < PW and 0 <= ppy < H:
                    g = ti.exp(-(dr - ti.cast(bob_r, ti.f32)) * 0.22) * 0.35 * z_n
                    img[ppx, ppy] = img[ppx, ppy] + ti.Vector([0.2+0.5*z_n, 0.55+0.3*z_n, 1.0]) * g
    for ox in range(-bob_r, bob_r + 1):
        for oy in range(-bob_r, bob_r + 1):
            r2 = ti.cast(ox*ox + oy*oy, ti.f32)
            if r2 <= ti.cast(bob_r * bob_r, ti.f32):
                ppx = bob_fx + ox; ppy = bob_fy + oy
                if 0 <= ppx < PW and 0 <= ppy < H:
                    nx_ = ti.cast(ox, ti.f32) / ti.cast(bob_r, ti.f32)
                    ny_ = ti.cast(oy, ti.f32) / ti.cast(bob_r, ti.f32)
                    nz_ = ti.sqrt(ti.max(0.0, 1.0 - nx_*nx_ - ny_*ny_))
                    lx2, ly2, lz2 = ti.f32(-0.45), ti.f32(-0.70), ti.f32(0.55)
                    diff = ti.max(0.0, nx_*lx2 + ny_*ly2 + nz_*lz2)
                    spec = ti.pow(ti.max(0.0, -(2.0*diff*nz_ - lz2)), 18.0)
                    base_c = ti.Vector([0.18 + 0.55*z_n, 0.42 + 0.38*z_n, 0.95])
                    col2 = base_c * (0.12 + 0.78*diff) + ti.Vector([1.0, 1.0, 1.0]) * spec * 0.85
                    img[ppx, ppy] = ti.min(col2, ti.Vector([1.0, 1.0, 1.0]))

    # ═════════════════════════════════════════════════════
    # PANEL 1 — TOP VIEW  (X horizontal, Z vertical/down)
    # Pivot at centre; X→right, Z→down (same screen orientation as front Y)
    # ═════════════════════════════════════════════════════

    # ── Sweep-circle guide (max radius the bob can reach) ─
    max_r = ti.cast(L * sc, ti.i32)
    for deg in range(360):
        ang = ti.cast(deg, ti.f32) * 3.14159265 / 180.0
        cx = TP_X + ti.cast(ti.cos(ang) * ti.cast(max_r, ti.f32), ti.i32)
        cy = TP_Y + ti.cast(ti.sin(ang) * ti.cast(max_r, ti.f32), ti.i32)
        if PW <= cx < W and 0 <= cy < H:
            img[cx, cy] = img[cx, cy] * 0.4 + ti.Vector([0.1, 0.25, 0.22]) * 0.6

    # ── Cross-hair axis lines ────────────────────────────
    for i in range(-max_r, max_r + 1):
        # horizontal (X axis)
        px_ = TP_X + i; py_ = TP_Y
        if PW <= px_ < W and 0 <= py_ < H:
            img[px_, py_] = img[px_, py_] * 0.5 + ti.Vector([0.06, 0.15, 0.12]) * 0.5
        # vertical (Z axis)
        px_ = TP_X; py_ = TP_Y + i
        if PW <= px_ < W and 0 <= py_ < H:
            img[px_, py_] = img[px_, py_] * 0.5 + ti.Vector([0.06, 0.15, 0.12]) * 0.5

    # ── Top trail (X, Z) — purple/teal gradient ──────────
    for j in range(cnt - 1):
        ia = (head - 1 - j + TRAIL_LEN) % TRAIL_LEN
        ib = (head - 2 - j + TRAIL_LEN) % TRAIL_LEN
        pa = trail[ia]; pb = trail[ib]

        fade = 1.0 - ti.cast(j, ti.f32) / ti.cast(cnt, ti.f32)
        fade = fade * fade
        # colour by angle in XZ plane — nice rainbow orbit effect
        ang_c = (ti.atan2(pa[2], pa[0]) / (2.0 * 3.14159265) + 0.5)
        col_t = ti.Vector([
            0.5 + 0.45 * ti.sin(ang_c * 6.28),
            0.55 + 0.35 * ti.sin(ang_c * 6.28 + 2.09),
            0.85 + 0.15 * ti.sin(ang_c * 6.28 + 4.18),
        ]) * (fade * 0.88 + 0.04)

        ax = TP_X + ti.cast(pa[0] * sc, ti.i32)
        ay = TP_Y + ti.cast(pa[2] * sc, ti.i32)
        bx = TP_X + ti.cast(pb[0] * sc, ti.i32)
        by = TP_Y + ti.cast(pb[2] * sc, ti.i32)
        dx = bx - ax; dy = by - ay
        ns = ti.max(ti.abs(dx), ti.abs(dy))
        if ns > 0:
            for s in range(ns + 1):
                t = ti.cast(s, ti.f32) / ti.cast(ns, ti.f32)
                cx = ax + ti.cast(t * dx, ti.i32)
                cy = ay + ti.cast(t * dy, ti.i32)
                if PW <= cx < W and 0 <= cy < H:
                    img[cx, cy] = ti.max(img[cx, cy], col_t)

    # ── Top rod (pivot → bob shadow on XZ plane) ─────────
    bob_tx = TP_X + ti.cast(tx * sc, ti.i32)
    bob_ty = TP_Y + ti.cast(tz * sc, ti.i32)
    dx = bob_tx - TP_X; dy = bob_ty - TP_Y
    rl2 = ti.max(ti.abs(dx), ti.abs(dy))
    if rl2 > 0:
        for s in range(rl2 + 1):
            t = ti.cast(s, ti.f32) / ti.cast(rl2, ti.f32)
            cx = TP_X + ti.cast(t * dx, ti.i32)
            cy = TP_Y + ti.cast(t * dy, ti.i32)
            thick2 = ti.cast(2.5 - t * 1.5, ti.i32)
            rod_c2 = ti.Vector([0.35 + 0.30*t, 0.72 + 0.18*t, 0.65 + 0.15*t])
            for ox in range(-thick2, thick2 + 1):
                for oy in range(-thick2, thick2 + 1):
                    ppx = cx + ox; ppy = cy + oy
                    if PW <= ppx < W and 0 <= ppy < H:
                        e2 = ti.cast(ox*ox + oy*oy, ti.f32)
                        sp = ti.exp(-e2 * 0.5) * (0.55 + 0.45*(1.0-t))
                        img[ppx, ppy] = rod_c2 * sp + img[ppx, ppy] * (1.0 - sp * 0.65)

    # ── Top pivot ────────────────────────────────────────
    for ox in range(-9, 10):
        for oy in range(-9, 10):
            r2 = ti.cast(ox*ox + oy*oy, ti.f32)
            if r2 <= 81.0:
                ppx = TP_X + ox; ppy = TP_Y + oy
                if PW <= ppx < W and 0 <= ppy < H:
                    br = ti.exp(-r2 * 0.025)
                    img[ppx, ppy] = ti.Vector([0.6, 0.85, 0.78]) * br \
                                  + ti.Vector([0.08, 0.22, 0.18]) * (1.0 - br)
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            if ox*ox + oy*oy <= 9:
                ppx = TP_X + ox; ppy = TP_Y + oy
                if PW <= ppx < W and 0 <= ppy < H:
                    img[ppx, ppy] = ti.Vector([0.95, 1.0, 0.97])

    # ── Top bob (Phong, y-depth shading = how low bob hangs) ──
    y_n   = ty / L              # 0 (at top) .. 1 (fully down)
    bob_r2 = ti.cast(10.0 + y_n * 6.0, ti.i32)
    glow_r2 = bob_r2 + 16
    for ox in range(-glow_r2, glow_r2 + 1):
        for oy in range(-glow_r2, glow_r2 + 1):
            dr = ti.sqrt(ti.cast(ox*ox + oy*oy, ti.f32))
            if dr <= ti.cast(glow_r2, ti.f32):
                ppx = bob_tx + ox; ppy = bob_ty + oy
                if PW <= ppx < W and 0 <= ppy < H:
                    g = ti.exp(-(dr - ti.cast(bob_r2, ti.f32)) * 0.25) * 0.35 * y_n
                    img[ppx, ppy] = img[ppx, ppy] + ti.Vector([0.1, 0.8+0.2*y_n, 0.65]) * g
    for ox in range(-bob_r2, bob_r2 + 1):
        for oy in range(-bob_r2, bob_r2 + 1):
            r2 = ti.cast(ox*ox + oy*oy, ti.f32)
            if r2 <= ti.cast(bob_r2 * bob_r2, ti.f32):
                ppx = bob_tx + ox; ppy = bob_ty + oy
                if PW <= ppx < W and 0 <= ppy < H:
                    nx_ = ti.cast(ox, ti.f32) / ti.cast(bob_r2, ti.f32)
                    ny_ = ti.cast(oy, ti.f32) / ti.cast(bob_r2, ti.f32)
                    nz_ = ti.sqrt(ti.max(0.0, 1.0 - nx_*nx_ - ny_*ny_))
                    # light from top-left for top view
                    lx2, ly2, lz2 = ti.f32(-0.50), ti.f32(-0.65), ti.f32(0.57)
                    diff = ti.max(0.0, nx_*lx2 + ny_*ly2 + nz_*lz2)
                    spec = ti.pow(ti.max(0.0, -(2.0*diff*nz_ - lz2)), 16.0)
                    base_t = ti.Vector([0.15 + 0.40*y_n, 0.75 + 0.20*y_n, 0.58 + 0.22*y_n])
                    col_b  = base_t * (0.12 + 0.78*diff) + ti.Vector([1.0, 1.0, 1.0]) * spec * 0.8
                    img[ppx, ppy] = ti.min(col_b, ti.Vector([1.0, 1.0, 1.0]))


def main():
    init()
    gui = ti.GUI("Spherical Pendulum  |  Front & Top View  |  [R] reset  [ESC] quit",
                 res=(W, H), fast_gui=True)

    while gui.running:
        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == 'r':
                init()

        for _ in range(STEPS):
            step()
        record()
        render()

        gui.set_image(img)

        # lightweight text labels
        gui.text("FRONT  VIEW  (X – Y)",  pos=(0.03, 0.97), color=0x4488FF, font_size=20)
        gui.text("← X →",                 pos=(0.19, 0.92), color=0x223355, font_size=14)
        gui.text("↓ Y",                   pos=(0.03, 0.87), color=0x223355, font_size=14)

        gui.text("TOP  VIEW  (X – Z)",    pos=(0.53, 0.97), color=0x33DDAA, font_size=20)
        gui.text("← X →",                pos=(0.69, 0.92), color=0x1A3330, font_size=14)
        gui.text("↓ Z",                   pos=(0.53, 0.87), color=0x1A3330, font_size=14)

        th_deg = math.degrees(float(theta[None]))
        ph_deg = math.degrees(float(phi[None])) % 360
        gui.text(f"θ = {th_deg:+6.1f}°   φ = {ph_deg:5.1f}°",
                 pos=(0.03, 0.03), color=0x667799, font_size=15)

        gui.show()


if __name__ == "__main__":
    main()