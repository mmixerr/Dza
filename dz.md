import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ============================================
# 1. Параметры
# ============================================
g = 9.81  # ускорение свободного падения (м/с²)
m = 1.0  # масса (кг)
Lx, Ly, Lz = 0.25, 0.15, 0.02  # размеры книги

# Моменты инерции
I_x = (m / 12.0) * (Ly**2 + Lz**2)
I_y = (m / 12.0) * (Lx**2 + Lz**2)
I_z = (m / 12.0) * (Lx**2 + Ly**2)
I = np.array([I_x, I_y, I_z])

print(f"Моменты инерции: Ix={I_x:.5f}, Iy={I_y:.5f}, Iz={I_z:.5f}")
print(f"Ускорение свободного падения: g = {g} м/с²")


# ============================================
# 2. Полная система уравнений
# ============================================
def full_dynamics(t, state):
    """
    state = [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
    """
    # Позиция и скорость центра масс
    x, y, z, vx, vy, vz = state[0:6]

    # Кватернион
    q0, q1, q2, q3 = state[6:10]

    # Угловые скорости
    wx, wy, wz = state[10:13]

    # 1. Поступательное движение (свободное падение)
    #    Начальная скорость вверх: vy0 > 0
    ax, ay, az = 0, -g, 0  # сила тяжести вдоль Y

    # 2. Вращательное движение (уравнения Эйлера)
    Mx = (I_y - I_z) * wy * wz
    My = (I_z - I_x) * wz * wx
    Mz = (I_x - I_y) * wx * wy

    dwx_dt = Mx / I_x
    dwy_dt = My / I_y
    dwz_dt = Mz / I_z

    # 3. Кинематика кватерниона
    omega_vec = np.array([wx, wy, wz])
    q_vec = np.array([q1, q2, q3])
    q_scalar = q0

    dq_scalar = -0.5 * np.dot(omega_vec, q_vec)
    dq_vec = 0.5 * (q_scalar * omega_vec + np.cross(omega_vec, q_vec))

    return [vx, vy, vz, ax, ay, az,
            dq_scalar, dq_vec[0], dq_vec[1], dq_vec[2],
            dwx_dt, dwy_dt, dwz_dt]


# ============================================
# 3. Начальные условия
# ============================================
# Подбрасываем книгу вверх со скоростью 3 м/с
x0, y0, z0 = 0, 0, 0  # начальная позиция
vx0, vy0, vz0 = 0, 3.0, 0  # начальная скорость (вверх)

# Вращение вокруг средней оси Y
wx0, wy0, wz0 = 0.1, 15.0, 0.05  # рад/с

# Начальная ориентация (кватернион)
q0 = np.array([1.0, 0.0, 0.0, 0.0])

# Вектор состояния
y0 = [x0, y0, z0, vx0, vy0, vz0, q0[0], q0[1], q0[2], q0[3], wx0, wy0, wz0]

# Время симуляции (до падения на землю)
t_max = 1.2  # секунд (высота подброса ~0.45 м)
t_eval = np.linspace(0, t_max, 800)

# Решение
sol = solve_ivp(full_dynamics, [0, t_max], y0, t_eval=t_eval, method='RK45', rtol=1e-8)

# Извлекаем результаты
pos = sol.y[:3].T
vel = sol.y[3:6].T
q_sol = sol.y[6:10].T
omega_sol = sol.y[10:13].T

# ============================================
# 4. График угловых скоростей и траектории
# ============================================
fig = plt.figure(figsize=(14, 5))

# График угловых скоростей
ax1 = plt.subplot(1, 2, 1)
ax1.plot(sol.t, omega_sol[:, 0], 'b-', label='ωx', linewidth=1.5)
ax1.plot(sol.t, omega_sol[:, 1], 'r-', label='ωy (средняя ось)', linewidth=2)
ax1.plot(sol.t, omega_sol[:, 2], 'g-', label='ωz', linewidth=1.5)
ax1.set_xlabel('Время (с)')
ax1.set_ylabel('Угловая скорость (рад/с)')
ax1.set_title('Эффект Джанибекова на Земле (g=9.81)')
ax1.legend()
ax1.grid(True)

# Траектория центра масс
ax2 = plt.subplot(1, 2, 2)
ax2.plot(sol.t, pos[:, 1], 'k-', linewidth=2)
ax2.set_xlabel('Время (с)')
ax2.set_ylabel('Высота (м)')
ax2.set_title('Траектория полёта книги')
ax2.grid(True)
ax2.axhline(y=0, color='r', linestyle='--', label='земля')
ax2.legend()

plt.tight_layout()
plt.show()

# ============================================
# 5. 3D анимация полёта книги
# ============================================


def quat_to_rotmat(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)]
    ])


# Вершины книги
vertices = np.array([
    [-Lx / 2, -Ly / 2, -Lz / 2],
    [Lx / 2, -Ly / 2, -Lz / 2],
    [Lx / 2, Ly / 2, -Lz / 2],
    [-Lx / 2, Ly / 2, -Lz / 2],
    [-Lx / 2, -Ly / 2, Lz / 2],
    [Lx / 2, -Ly / 2, Lz / 2],
    [Lx / 2, Ly / 2, Lz / 2],
    [-Lx / 2, Ly / 2, Lz / 2]
])

edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

fig3d = plt.figure(figsize=(12, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_xlim([-0.5, 0.5])
ax3d.set_ylim([0, 0.6])
ax3d.set_zlim([-0.5, 0.5])
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y (высота)')
ax3d.set_zlabel('Z')
ax3d.set_title('Полёт книги на Земле с вращением (эффект Джанибекова)')

# Рисуем землю
x_ground = np.linspace(-0.5, 0.5, 10)
z_ground = np.linspace(-0.5, 0.5, 10)
Xg, Zg = np.meshgrid(x_ground, z_ground)
Yg = np.zeros_like(Xg)
ax3d.plot_surface(Xg, Yg, Zg, alpha=0.3, color='brown')

lines = []
for edge in edges:
    line, = ax3d.plot([], [], [], 'b-', linewidth=2)
    lines.append(line)


def update(frame):
    q = q_sol[frame]
    R = quat_to_rotmat(q)

    # Позиция центра масс
    center = pos[frame]

    # Поворачиваем и перемещаем вершины
    vert_rot = vertices @ R.T + center

    for i, edge in enumerate(edges):
        x = [vert_rot[edge[0], 0], vert_rot[edge[1], 0]]
        y = [vert_rot[edge[0], 1], vert_rot[edge[1], 1]]
        z = [vert_rot[edge[0], 2], vert_rot[edge[1], 2]]
        lines[i].set_data(x, z)
        lines[i].set_3d_properties(y)

    ax3d.set_title(f'Книга в полёте - Время: {sol.t[frame]:.2f} с, Высота: {center[1]:.2f} м')
    return lines


ani = FuncAnimation(fig3d, update, frames=len(sol.t), interval=20, blit=False, repeat=True)

plt.show()