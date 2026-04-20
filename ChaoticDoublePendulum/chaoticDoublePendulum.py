import taichi as ti
import math

NUM_PENDULUMS   = 123
ANGLE_DELTA_DEG = 0.02

ti.init(arch=ti.gpu)

G       = 9.81
L1      = 0.25
L2      = 0.2
M1      = 1.0
M2      = 1.0
DT      = 5e-4
STEPS   = 40

W, H          = 900, 900
TRAIL_LEN     = 150
PIVOT_X       = 0.5
PIVOT_Y       = 0.62

N = NUM_PENDULUMS
DELTA = math.radians(ANGLE_DELTA_DEG)

theta1  = ti.field(dtype=ti.f32, shape=N)
theta2  = ti.field(dtype=ti.f32, shape=N)
omega1  = ti.field(dtype=ti.f32, shape=N)
omega2  = ti.field(dtype=ti.f32, shape=N)

trail   = ti.Vector.field(2, dtype=ti.f32, shape=(N, TRAIL_LEN))
t_head  = ti.field(dtype=ti.i32, shape=N)
t_count = ti.field(dtype=ti.i32, shape=N)

canvas_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

colors = ti.Vector.field(3, dtype=ti.f32, shape=N)

def hsv2rgb(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)

def init():
    theta1_np = []
    theta2_np = []
    base1 = math.radians(130.0)
    base2 = math.radians(-40.0)
    import numpy as np
    for i in range(N):
        theta1_np.append(base1 + i * DELTA + np.random.uniform(-1e-4, 1e-4))
        theta2_np.append(base2)

    for i in range(N):
        theta1[i]  = theta1_np[i]
        theta2[i]  = theta2_np[i]
        omega1[i]  = 0.0
        omega2[i]  = 0.0
        t_head[i]  = 0
        t_count[i] = 0
        hue = i / max(N, 1)
        r, g, b = hsv2rgb(hue, 0.95, 1.0)
        colors[i] = ti.Vector([r, g, b])

@ti.func
def derivs(t1: ti.f32, t2: ti.f32, w1: ti.f32, w2: ti.f32):
    delta = t2 - t1

    den1 = (M1 + M2) * L1 - M2 * L1 * ti.cos(delta)**2
    den2 = (L2 / L1) * den1

    a1 = (
        M2 * L1 * w1**2 * ti.sin(delta) * ti.cos(delta)
        + M2 * G * ti.sin(t2) * ti.cos(delta)
        + M2 * L2 * w2**2 * ti.sin(delta)
        - (M1 + M2) * G * ti.sin(t1)
    ) / den1

    a2 = (
        -M2 * L2 * w2**2 * ti.sin(delta) * ti.cos(delta)
        + (M1 + M2) * G * ti.sin(t1) * ti.cos(delta)
        - (M1 + M2) * L1 * w1**2 * ti.sin(delta)
        - (M1 + M2) * G * ti.sin(t2)
    ) / den2

    return w1, w2, a1, a2

@ti.kernel
def physics_step():
    for i in range(N):
        t1 = theta1[i]; t2 = theta2[i]
        w1 = omega1[i]; w2 = omega2[i]

        k1_t1, k1_t2, k1_w1, k1_w2 = derivs(t1, t2, w1, w2)

        k2_t1, k2_t2, k2_w1, k2_w2 = derivs(
            t1 + 0.5*DT*k1_t1, t2 + 0.5*DT*k1_t2,
            w1 + 0.5*DT*k1_w1, w2 + 0.5*DT*k1_w2)

        k3_t1, k3_t2, k3_w1, k3_w2 = derivs(
            t1 + 0.5*DT*k2_t1, t2 + 0.5*DT*k2_t2,
            w1 + 0.5*DT*k2_w1, w2 + 0.5*DT*k2_w2)

        k4_t1, k4_t2, k4_w1, k4_w2 = derivs(
            t1 + DT*k3_t1, t2 + DT*k3_t2,
            w1 + DT*k3_w1, w2 + DT*k3_w2)

        theta1[i] = t1 + (DT/6.0)*(k1_t1 + 2.0*k2_t1 + 2.0*k3_t1 + k4_t1)
        theta2[i] = t2 + (DT/6.0)*(k1_t2 + 2.0*k2_t2 + 2.0*k3_t2 + k4_t2)
        omega1[i] = w1 + (DT/6.0)*(k1_w1 + 2.0*k2_w1 + 2.0*k3_w1 + k4_w1)
        omega2[i] = w2 + (DT/6.0)*(k1_w2 + 2.0*k2_w2 + 2.0*k3_w2 + k4_w2)

@ti.kernel
def record_tips():
    for i in range(N):
        x1 = PIVOT_X + L1 * ti.sin(theta1[i])
        y1 = PIVOT_Y - L1 * ti.cos(theta1[i])
        x2 = x1     + L2 * ti.sin(theta2[i])
        y2 = y1     - L2 * ti.cos(theta2[i])
        h  = t_head[i]
        trail[i, h] = ti.Vector([x2, y2])
        t_head[i]   = (h + 1) % TRAIL_LEN
        cnt = t_count[i]
        if cnt < TRAIL_LEN:
            t_count[i] = cnt + 1

@ti.kernel
def render():
    for px, py in canvas_pixels:
        canvas_pixels[px, py] = ti.Vector([0.03, 0.03, 0.08])

    for i in range(N):
        cnt   = t_count[i]
        head  = t_head[i]
        col   = colors[i]
        for j in range(cnt - 1):
            idx_a = (head - 1 - j + TRAIL_LEN) % TRAIL_LEN
            idx_b = (head - 2 - j + TRAIL_LEN) % TRAIL_LEN
            pa = trail[i, idx_a]
            pb = trail[i, idx_b]

            alpha = ti.cast(j, ti.f32) / ti.cast(cnt, ti.f32)
            alpha = 1.0 - alpha

            ax = ti.cast(pa[0] * W, ti.i32)
            ay = ti.cast(pa[1] * H, ti.i32)
            bx = ti.cast(pb[0] * W, ti.i32)
            by = ti.cast(pb[1] * H, ti.i32)

            dx = bx - ax; dy = by - ay
            steps_line = ti.max(ti.abs(dx), ti.abs(dy))
            if steps_line > 0:
                for s in range(steps_line + 1):
                    t_f = ti.cast(s, ti.f32) / ti.cast(steps_line, ti.f32)
                    px_ = ax + ti.cast(t_f * dx, ti.i32)
                    py_ = ay + ti.cast(t_f * dy, ti.i32)
                    bright = col * (alpha * 0.85 + 0.05)
                    for ox in ti.static(range(-1, 2)):
                        for oy in ti.static(range(-1, 2)):
                            qx = px_ + ox
                            qy = py_ + oy
                            if 0 <= qx < W and 0 <= qy < H:
                                existing = canvas_pixels[qx, qy]
                                canvas_pixels[qx, qy] = existing * 0.92 + bright * 0.5

@ti.kernel
def draw_rods_and_bobs():
    """Draw rods and bobs on top of trails."""
    for i in range(N):
        col = colors[i]
        px0 = ti.cast(PIVOT_X * W, ti.i32)
        py0 = ti.cast(PIVOT_Y * H, ti.i32)

        x1 = PIVOT_X + L1 * ti.sin(theta1[i])
        y1 = PIVOT_Y - L1 * ti.cos(theta1[i])
        x2 = x1     + L2 * ti.sin(theta2[i])
        y2 = y1     - L2 * ti.cos(theta2[i])

        px1 = ti.cast(x1 * W, ti.i32)
        py1 = ti.cast(y1 * H, ti.i32)
        px2 = ti.cast(x2 * W, ti.i32)
        py2 = ti.cast(y2 * H, ti.i32)

        dx = px1 - px0; dy = py1 - py0
        steps_line = ti.max(ti.abs(dx), ti.abs(dy))
        if steps_line > 0:
            for s in range(steps_line + 1):
                t_f = ti.cast(s, ti.f32) / ti.cast(steps_line, ti.f32)
                ppx = px0 + ti.cast(t_f * dx, ti.i32)
                ppy = py0 + ti.cast(t_f * dy, ti.i32)
                if 0 <= ppx < W and 0 <= ppy < H:
                    canvas_pixels[ppx, ppy] = col * 0.7 + ti.Vector([0.3, 0.3, 0.3])

        dx2 = px2 - px1; dy2 = py2 - py1
        steps2 = ti.max(ti.abs(dx2), ti.abs(dy2))
        if steps2 > 0:
            for s in range(steps2 + 1):
                t_f = ti.cast(s, ti.f32) / ti.cast(steps2, ti.f32)
                ppx = px1 + ti.cast(t_f * dx2, ti.i32)
                ppy = py1 + ti.cast(t_f * dy2, ti.i32)
                if 0 <= ppx < W and 0 <= ppy < H:
                    canvas_pixels[ppx, ppy] = col * 0.7 + ti.Vector([0.3, 0.3, 0.3])

        for bx in range(-2, 3):
            for by in range(-2, 3):
                ppx = px1 + bx; ppy = py1 + by
                if 0 <= ppx < W and 0 <= ppy < H:
                    canvas_pixels[ppx, ppy] = ti.Vector([1.0, 1.0, 1.0])

        for bx in range(-8, 9):
            for by in range(-8, 9):
                if bx*bx + by*by <= 49:
                    ppx = px2 + bx; ppy = py2 + by
                    if 0 <= ppx < W and 0 <= ppy < H:
                        canvas_pixels[ppx, ppy] = col * 0.5 + ti.Vector([0.5, 0.5, 0.5])

        for bx in range(-2, 3):
            for by in range(-2, 3):
                ppx = px0 + bx; ppy = py0 + by
                if 0 <= ppx < W and 0 <= ppy < H:
                    canvas_pixels[ppx, ppy] = ti.Vector([0.9, 0.9, 0.9])

def main():
    init()
    gui = ti.GUI(
        f"Double Pendulum  |  N={N}  Δθ={ANGLE_DELTA_DEG}°  |  [R] reset  [SPACE] slow mo  [ESC] quit",
        res=(W, H),
        fast_gui=True,
    )

    frame = 0
    slow_mo = False
    while gui.running:
        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == 'r':
                init()
                frame = 0
            elif e.key == ti.GUI.SPACE:
                slow_mo = not slow_mo

        current_steps = STEPS // 4 if slow_mo else STEPS
        for _ in range(current_steps):
            physics_step()
        record_tips()

        render()
        draw_rods_and_bobs()

        gui.set_image(canvas_pixels)
        gui.show()
        frame += 1

if __name__ == "__main__":
    main()
