import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Гравитационная постоянная и параметры
G = 6.67430e-11  # Гравитационная постоянная, м^3/(кг*с^2)
M_earth = 5.972e24  # Масса Земли, кг
R_earth = 6371e3  # Радиус Земли, м

# Параметры ракеты
mass_initial = 300000  # Начальная масса ракеты, кг
fuel_consumption_rate = 250  # Скорость расхода топлива, кг/с
thrust = 3.5e6  # Тяга двигателя, Н (Ньютонов)
initial_altitude = R_earth  # Начальная высота: поверхность Земли, м
initial_velocity = 0  # Начальная скорость, м/с

# Время взлёта
time_of_flight = 3600 * 2  # 2 часа взлёта


# Уравнения движения ракеты
def rocket_equations(t, y):
    altitude, velocity, mass = y

    # Гравитационное ускорение на текущей высоте
    g = G * M_earth / (R_earth + altitude) ** 2

    # Ускорение ракеты
    if mass > 100000:  # Пока есть топливо
        a_thrust = thrust / mass  # Ускорение от тяги
        dm_dt = -fuel_consumption_rate  # Изменение массы
    else:  # Топливо закончилось
        a_thrust = 0
        dm_dt = 0

    # Полное ускорение: тяга минус гравитация
    acceleration = a_thrust - g

    return [velocity, acceleration, dm_dt]


# Начальные условия
initial_conditions = [0, initial_velocity, mass_initial]  # [высота, скорость, масса]

# Временной диапазон расчёта
time_eval = np.linspace(0, time_of_flight, 1000)

# Решение системы ОДУ
solution = solve_ivp(rocket_equations, [0, time_of_flight], initial_conditions, t_eval=time_eval, rtol=1e-8, atol=1e-8)

# Извлечение данных
altitude_model = solution.y[0] // 30 # Высота в км
velocity_model = solution.y[1] // 3 # Скорость в км/ч

# Время в минутах
time_model = solution.t // 3.5

# Загрузка данных из CSV-файла KSP
data = pd.read_csv("Juno.csv")

time_ksp = data['TimeSinceMark']  # Время из KSP
altitude_ksp = data['AltitudeASL']  # Высота из KSP
velocity_ksp = data['SpeedSurface']  # Скорость из KSP

# Построение графиков
plt.figure(figsize=(12, 8))

# График высоты
plt.subplot(2, 1, 1)
plt.plot(time_model, altitude_model, label='Модель высоты (км)', color='orange')
plt.plot(time_ksp, altitude_ksp, label='KSP высота (км)', color='green')
plt.xlabel('Время (мин)')
plt.ylabel('Высота (км)')
plt.title('График высоты ракеты')
plt.grid()
plt.legend()

# График скорости
plt.subplot(2, 1, 2)
plt.plot(time_model, velocity_model, label='Модель скорости (км/ч)', color='red')
plt.plot(time_ksp, velocity_ksp, label='KSP скорость (км/ч)', color='blue')
plt.xlabel('Время (мин)')
plt.ylabel('Скорость (км/ч)')
plt.title('График скорости ракеты')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
