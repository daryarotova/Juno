import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Гравитационная постоянная и параметры
G = 6.67430e-11  # Гравитационная постоянная, м^3/(кг*с^2)
M_earth = 5.972e24  # Масса Земли, кг
R_earth = 6371e3  # Радиус Земли, м
M_jupiter = 1.898e27  # Масса Юпитера, кг
R_jupiter = 69911e3  # Радиус Юпитера, м
AU = 149.6e9  # Астрономическая единица, м

# Параметры зонда
mass_initial = 300000  # Начальная масса зонда, кг
fuel_consumption_rate = 5.0  # Скорость расхода топлива, кг/с
initial_altitude = R_earth + 200e3  # Начальная высота над уровнем Земли, м (200 км)
initial_velocity = 11000  # Начальная скорость, м/с (достаточная для выхода на орбиту)

# Время полета до Юпитера
time_to_jupiter = 3600 * 24 * 365 * 2  # 2 года

# Уравнения движения
def equations(t, y):
    x, y_pos, vx, vy, mass = y

    # Расстояния до Земли и Юпитера
    r_earth = np.sqrt(x**2 + y_pos**2)
    r_jupiter = np.sqrt((x - 5.2 * AU)**2 + y_pos**2)

    # Гравитационные силы
    F_gravity_earth = -G * M_earth * mass / max(r_earth, R_earth)**2 if r_earth > R_earth else 0
    F_gravity_jupiter = -G * M_jupiter * mass / max(r_jupiter, R_jupiter)**2 if r_jupiter > R_jupiter else 0

    # Ускорения
    ax = F_gravity_earth * (x / max(r_earth, R_earth)) + F_gravity_jupiter * ((x - 5.2 * AU) / max(r_jupiter, R_jupiter))
    ay = F_gravity_earth * (y_pos / max(r_earth, R_earth)) + F_gravity_jupiter * (y_pos / max(r_jupiter, R_jupiter))

    return [vx, vy, ax, ay, -fuel_consumption_rate]

# Начальные условия
initial_conditions = [0, initial_altitude, initial_velocity, 0, mass_initial]

# Решение системы ОДУ
t_eval = np.linspace(0, time_to_jupiter, 1000)
solution = solve_ivp(equations, [0, time_to_jupiter], initial_conditions, t_eval=t_eval, rtol=1e-8, atol=1e-8)

# Расчеты
x, y_pos, vx, vy, mass = solution.y
orbital_speed = np.sqrt(vx**2 + vy**2)
altitude = np.maximum(np.sqrt(x**2 + y_pos**2) - R_earth, 0) / 1000  # Убедиться, что высота не отрицательна и перевести в километры

# Преобразование времени в дни
time = solution.t / (3600 * 24)

# Построение графиков
plt.figure(figsize=(12, 8))

# Объединённый график скоростей
plt.subplot(2, 1, 1)
plt.plot(time, np.abs(vx), label='Наземная скорость (м/с)')
plt.plot(time, orbital_speed, label='Орбитальная скорость (м/с)', color='orange')
plt.xlabel('Время (дни)')
plt.ylabel('Скорость (м/с)')
plt.legend()
plt.grid()
plt.title('Наземная и орбитальная скорости')

# Высота
plt.subplot(2, 1, 2)
plt.plot(time, altitude, label='Высота (км)', color='green')
plt.xlabel('Время (дни)')
plt.ylabel('Высота (км)')
plt.legend()
plt.grid()
plt.title('Высота')

plt.tight_layout()
plt.show()