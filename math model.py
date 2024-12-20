import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Константы
G = 6.67430e-11  # Гравитационная постоянная, м^3/(кг*с^2)
M_earth = 5.972e24  # Масса Земли в кг
R_earth = 6.371e6  # Радиус Земли в м
rho_0 = 1.225  # Плотность воздуха
h_0 = 8500  # Высота атмосферы в м
C_d = 0.5  # Коэффициент сопротивления
S = 20  # Поперечная площадь м^2
F_thrust = 35e6  # Тяга, Н
mass_initial = 300000  # Начальная масса, кг
fuel_consumption_rate = 2500  # Скорость расхода топлива, кг/с
altitude_targets = [70e3, 900e3]  # Высоты разных этапов вздёта
# Система дифференциальных уравнений
def rocket_equations(t, y, F_thrust):
    h, v, m = y  #высота, скорость, масса
    #Гравитационное ускорение
    g = G * M_earth / (R_earth + h)**2
    #Плотность воздуха
    rho = rho_0 * np.exp(-h / h_0) if h > 0 else rho_0
    #Сила сопротивления
    drag = 0.5 * C_d * rho * v**2 * S / m
    #Тяга
    F_current = F_thrust if m > 100000 else 0
    #Ускорение
    a = F_current / m - g - drag if m > 0 else -g
    #Изменение массы
    dm_dt = -fuel_consumption_rate if F_current > 0 else 0
    return [v, a, dm_dt]#Возвращаем производные
#Моделируем каждый этап
def simulate_stage(initial_conditions, target_altitude, F_thrust):
    #Параметры времени для каждого этапа
    t_span = (0, 2000)  #Время полета
    t_eval = np.linspace(0, 2000, 1000)  # Время полета, дальше объединено
    # Решаем систему уравнений
    solution = solve_ivp(
        rocket_equations, t_span, initial_conditions, t_eval=t_eval, args=(F_thrust,), rtol=1e-8, atol=1e-8
    )
    h_vals = solution.y[0]  # Значения высоты
    v_vals = solution.y[1]  # Значения скорости
    m_vals = solution.y[2]  # Значения массы
    t_vals = solution.t  # Значения времени
    h_vals = np.clip(h_vals, 0, target_altitude)
    #Определяем конечную точку, если достигаем целевой высоты
    for i, h in enumerate(h_vals):
        if h >= target_altitude:
            return t_vals[:i+1], h_vals[:i+1], v_vals[:i+1], m_vals[:i+1]
    return t_vals, h_vals, v_vals, m_vals
# Моделируем все этапы
h_vals_all, v_vals_all, t_vals_all = [], [], []
initial_conditions = [0, 0, mass_initial]
for target in altitude_targets:
    #Симуляция текущего этапа
    t_vals, h_vals, v_vals, m_vals = simulate_stage(initial_conditions, target, F_thrust)
    #Сохраним результаты для построения графиков
    if t_vals_all:
        t_vals += t_vals_all[-1]
    t_vals_all.extend(t_vals)
    h_vals_all.extend(h_vals)
    v_vals_all.extend(v_vals)
    #Обновим начальные условия для следующего этапа
    initial_conditions = [h_vals[-1], v_vals[-1], m_vals[-1]]
# Преобразуем результаты в массивы numpy
h_vals_all = np.array(h_vals_all)
v_vals_all = np.array(v_vals_all)
t_vals_all = np.array(t_vals_all)
#Построим графика скорости во времени
plt.figure(figsize=(10, 6))
plt.plot(t_vals_all, v_vals_all / 1000, label="Скорость (км/с)", color="blue")
plt.xlabel("Время (с)")
plt.ylabel("Скорость (км/с), ось Y")
plt.title("График скорости ракеты во времени (этапы)")
plt.grid()
plt.legend()
plt.show()
#Ниже построили графики высоты во времени
plt.figure(figsize=(10, 6))
plt.plot(t_vals_all, h_vals_all / 1000, label="Высота (км)", color="orange")
plt.xlabel("Время (с)")
plt.ylabel("Высота (км), ось Y")
plt.title("График высоты ракеты во времени (этапы)")
plt.grid()
plt.legend()
plt.show()
