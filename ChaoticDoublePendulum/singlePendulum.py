import taichi as ti
import math

ti.init(arch=ti.gpu)

# ── Physics ───────────────────────────────
G   = 9.81
L   = 0.23
DT  = 5e-4
STEPS = 40

# ── Rendering ────────────────────────────
W, H      = 900, 900
TRAIL_LEN = 120
PIVOT_X   = 0.5
PIVOT_Y   = 0.30

PEND_TOP       = 0.02
PEND_BOTTOM    = 0.68
WAVE_TOP       = 0.74
WAVE_BOTTOM    = 0.97
PANEL_EDGE_SOFTNESS = 0.018

WAVE_SAMPLES   = 700
wave_points    = ti.Vector.field(2, dtype=ti.f32, shape=WAVE_SAMPLES)

# ── State ────────────────────────────────
theta  = ti.field(dtype=ti.f32, shape=())
omega  = ti.field(dtype=ti.f32, shape=())

# Trail buffer
trail   = ti.Vector.field(2, dtype=ti.f32, shape=TRAIL_LEN)
t_head  = ti.field(dtype=ti.i32, shape=())
t_count = ti.field(dtype=ti.i32, shape=())

canvas_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

# ─────────────────────────────────────────
# Init
# ─────────────────────────────────────────
def init():
    theta[None] = math.radians(120)
    omega[None] = 0.0
    t_head[None] = 0
    t_count[None] = 0

# ─────────────────────────────────────────
# Physics (RK4)
# ─────────────────────────────────────────
@ti.func
def derivs(t, w):
    return w, -(G / L) * ti.sin(t)

@ti.kernel
def physics_step():
    t = theta[None]
    w = omega[None]

    k1_t, k1_w = derivs(t, w)
    k2_t, k2_w = derivs(t + 0.5*DT*k1_t, w + 0.5*DT*k1_w)
    k3_t, k3_w = derivs(t + 0.5*DT*k2_t, w + 0.5*DT*k2_w)
    k4_t, k4_w = derivs(t + DT*k3_t, w + DT*k3_w)

    theta[None] = t + (DT/6.0)*(k1_t + 2*k2_t + 2*k3_t + k4_t)
    omega[None] = w + (DT/6.0)*(k1_w + 2*k2_w + 2*k3_w + k4_w)

# ─────────────────────────────────────────
# Trail update
# ─────────────────────────────────────────
@ti.kernel
def record_tip():
    x = PIVOT_X + L * ti.sin(theta[None])
    y = PIVOT_Y - L * ti.cos(theta[None])

    h = t_head[None]
    trail[h] = ti.Vector([x, y])
    t_head[None] = (h + 1) % TRAIL_LEN

    if t_count[None] < TRAIL_LEN:
        t_count[None] += 1

# ─────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────
@ti.kernel
def render():
    # atmospheric background + panel split
    for px, py in canvas_pixels:
        nx = ti.cast(px, ti.f32) / ti.cast(W, ti.f32)
        ny = ti.cast(py, ti.f32) / ti.cast(H, ti.f32)

        top_mix = ti.max(0.0, 1.0 - ny * 1.35)
        col_top = ti.Vector([0.03, 0.08, 0.17])
        col_mid = ti.Vector([0.01, 0.02, 0.06])
        col_bottom = ti.Vector([0.02, 0.04, 0.10])

        base = col_mid * (1.0 - top_mix) + col_top * top_mix
        if ny > PEND_BOTTOM:
            band_t = ti.min(1.0, (ny - PEND_BOTTOM) / (WAVE_BOTTOM - PEND_BOTTOM + 1e-6))
            base = base * (1.0 - band_t) + col_bottom * band_t

        vignette = (nx - 0.5) * (nx - 0.5) + (ny - 0.5) * (ny - 0.5)
        base *= (1.03 - 0.45 * vignette)

        split_dist = ti.abs(ny - ((PEND_BOTTOM + WAVE_TOP) * 0.5))
        split_glow = ti.max(0.0, 1.0 - split_dist / PANEL_EDGE_SOFTNESS)
        base += ti.Vector([0.05, 0.07, 0.12]) * split_glow * 0.35

        canvas_pixels[px, py] = base

    # trail
    cnt  = t_count[None]
    head = t_head[None]

    for j in range(cnt - 1):
        idx_a = (head - 1 - j + TRAIL_LEN) % TRAIL_LEN
        idx_b = (head - 2 - j + TRAIL_LEN) % TRAIL_LEN

        pa = trail[idx_a]
        pb = trail[idx_b]

        alpha = 1.0 - ti.cast(j, ti.f32) / ti.cast(cnt, ti.f32)

        ax = ti.cast(pa[0] * W, ti.i32)
        ay = ti.cast(pa[1] * H, ti.i32)
        bx = ti.cast(pb[0] * W, ti.i32)
        by = ti.cast(pb[1] * H, ti.i32)

        dx = bx - ax
        dy = by - ay
        steps_line = ti.max(ti.abs(dx), ti.abs(dy))

        if steps_line > 0:
            for s in range(steps_line + 1):
                t_f = ti.cast(s, ti.f32) / ti.cast(steps_line, ti.f32)
                px_ = ax + ti.cast(t_f * dx, ti.i32)
                py_ = ay + ti.cast(t_f * dy, ti.i32)

                if 0 <= px_ < W and 0 <= py_ < H:
                    existing = canvas_pixels[px_, py_]
                    glow = ti.Vector([0.2, 0.6, 1.0]) * alpha
                    canvas_pixels[px_, py_] = existing * 0.86 + glow

# ─────────────────────────────────────────
# Draw rod + bob
# ─────────────────────────────────────────
@ti.kernel
def draw_pendulum():
    px0 = ti.cast(PIVOT_X * W, ti.i32)
    py0 = ti.cast(PIVOT_Y * H, ti.i32)

    x = PIVOT_X + L * ti.sin(theta[None])
    y = PIVOT_Y - L * ti.cos(theta[None])

    px1 = ti.cast(x * W, ti.i32)
    py1 = ti.cast(y * H, ti.i32)

    dx = px1 - px0
    dy = py1 - py0
    steps = ti.max(ti.abs(dx), ti.abs(dy))

    if steps > 0:
        for s in range(steps + 1):
            t_f = ti.cast(s, ti.f32) / ti.cast(steps, ti.f32)
            ppx = px0 + ti.cast(t_f * dx, ti.i32)
            ppy = py0 + ti.cast(t_f * dy, ti.i32)

            if 0 <= ppx < W and 0 <= ppy < H:
                rail_t = ti.cast(s, ti.f32) / ti.cast(steps + 1, ti.f32)
                rod_col = ti.Vector([0.85, 0.88, 0.92]) * (0.78 + 0.22 * rail_t)
                canvas_pixels[ppx, ppy] = rod_col

    # bob core + glow
    for bx in range(-10, 11):
        for by in range(-10, 11):
            ppx = px1 + bx
            ppy = py1 + by
            if 0 <= ppx < W and 0 <= ppy < H:
                r2 = ti.cast(bx * bx + by * by, ti.f32)
                if r2 <= 100.0:
                    glow = ti.exp(-r2 * 0.06)
                    existing = canvas_pixels[ppx, ppy]
                    add = ti.Vector([0.35, 0.8, 1.0]) * glow * 0.45
                    canvas_pixels[ppx, ppy] = existing + add
                if r2 <= 22.0:
                    canvas_pixels[ppx, ppy] = ti.Vector([0.97, 0.99, 1.0])

    # pivot cap
    for bx in range(-6, 7):
        for by in range(-6, 7):
            ppx = px0 + bx
            ppy = py0 + by
            if 0 <= ppx < W and 0 <= ppy < H:
                r2 = ti.cast(bx * bx + by * by, ti.f32)
                if r2 <= 30.0:
                    canvas_pixels[ppx, ppy] = ti.Vector([0.92, 0.95, 1.0])


@ti.kernel
def draw_sine_panel():
    # panel grid and axis
    for i in range(11):
        gx = ti.cast(i * (W - 1) / 10, ti.i32)
        for py in range(ti.cast(WAVE_TOP * H, ti.i32), ti.cast(WAVE_BOTTOM * H, ti.i32) + 1):
            if 0 <= gx < W and 0 <= py < H:
                canvas_pixels[gx, py] = canvas_pixels[gx, py] * 0.92 + ti.Vector([0.05, 0.08, 0.12])

    wave_mid = (WAVE_TOP + WAVE_BOTTOM) * 0.5
    ay = ti.cast(wave_mid * H, ti.i32)
    for px in range(W):
        if 0 <= ay < H:
            canvas_pixels[px, ay] = canvas_pixels[px, ay] * 0.6 + ti.Vector([0.18, 0.24, 0.33])

    # wave parameters tied to pendulum state
    th = theta[None]
    om = omega[None]
    th_norm = ti.min(1.0, ti.abs(th) / 3.1415926)

    amp = 0.05 + 0.09 * th_norm
    freq = 2.0 + 2.0 * th_norm
    phase = th + om * 0.12
    center = wave_mid

    for i in range(WAVE_SAMPLES):
        x = ti.cast(i, ti.f32) / ti.cast(WAVE_SAMPLES - 1, ti.f32)
        y = center + amp * ti.sin(2.0 * 3.1415926 * freq * x + phase)
        wave_points[i] = ti.Vector([x, y])

    for i in range(WAVE_SAMPLES - 1):
        pa = wave_points[i]
        pb = wave_points[i + 1]

        ax = ti.cast(pa[0] * (W - 1), ti.i32)
        ay0 = ti.cast(pa[1] * (H - 1), ti.i32)
        bx = ti.cast(pb[0] * (W - 1), ti.i32)
        by0 = ti.cast(pb[1] * (H - 1), ti.i32)

        dx = bx - ax
        dy = by0 - ay0
        steps_line = ti.max(ti.abs(dx), ti.abs(dy))

        if steps_line > 0:
            for s in range(steps_line + 1):
                t_f = ti.cast(s, ti.f32) / ti.cast(steps_line, ti.f32)
                px_ = ax + ti.cast(t_f * dx, ti.i32)
                py_ = ay0 + ti.cast(t_f * dy, ti.i32)

                if 0 <= px_ < W and 0 <= py_ < H:
                    for off in range(-2, 3):
                        yy = py_ + off
                        if 0 <= yy < H:
                            w = ti.exp(-ti.cast(off * off, ti.f32) * 0.55)
                            existing = canvas_pixels[px_, yy]
                            color = ti.Vector([0.24, 0.9, 0.65]) * w
                            canvas_pixels[px_, yy] = existing * (1.0 - 0.25 * w) + color

# ─────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────
def main():
    init()
    gui = ti.GUI("Single Pendulum + Responsive Sine Wave", res=(W, H), fast_gui=False)

    while gui.running:
        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == 'r':
                init()

        for _ in range(STEPS):
            physics_step()

        record_tip()
        render()
        draw_pendulum()
        draw_sine_panel()

        gui.set_image(canvas_pixels)
        theta_rad = float(theta[None])
        theta_deg = math.degrees(theta_rad)
        omega_rad = float(omega[None])
        omega_deg = math.degrees(omega_rad)

        gui.text(f"theta: {theta_deg:+7.2f} deg   ({theta_rad:+7.3f} rad)", pos=(0.02, 0.03),
             color=0xDCEBFF, font_size=18)
        gui.text(f"omega: {omega_deg:+7.2f} deg/s ({omega_rad:+7.3f} rad/s)", pos=(0.02, 0.065),
             color=0xB7FFD9, font_size=18)
        gui.text("R: reset   ESC: quit", pos=(0.02, 0.10), color=0x91A6C8, font_size=15)
        gui.show()

if __name__ == "__main__":
    main()