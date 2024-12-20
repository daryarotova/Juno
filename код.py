import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11  # Gravitational constant, m^3/kg/s^2
M_earth = 5.972e24  # Earth mass, kg
R_earth = 6.371e6  # Earth radius, m
rho_0 = 1.225  # Air density at sea level, kg/m^3
h_0 = 8500  # Scale height of atmosphere, m
C_d = 0.9  # Drag coefficient
S = 20  # Cross-sectional area, m^2
F1 = 3350000
F2 = 3000000
F_thrust = F1 + F2 # Thrust, N
F3 = 650000
mass_initial = 440000  # Initial mass, kg
fuel_consumption_rate = 289  # Fuel consumption rate, kg/s
flag = True

# Altitude targets for stages
altitude_targets = [70e3, 900e3]  # Heights in meters

# System of differential equations
def rocket_equations(t, y, F_thrust):
    h, v, m = y  # Unpack variables: height, velocity, mass

    # Gravitational acceleration
    g = G * M_earth / (R_earth + h)**2

    # Air density
    rho = rho_0 * np.exp(-h / h_0) if h > 0 else rho_0

    # Drag force
    drag = 0.5 * C_d * rho * v**2 * S / m

    # Thrust (set to 0 if mass is below operational limit)
    F_current = F_thrust if m > 100000 else 0

    # Acceleration
    a = F_current / m - g - drag if m > 0 else -g

    # Rate of change of mass
    dm_dt = -fuel_consumption_rate if F_current > 0 else 0

    return [v, a, dm_dt]

# Function to simulate each stage
def simulate_stage(initial_conditions, target_altitude, F_thrust):
    # Time parameters for each stage
    t_span = (0, 2000)  # Time range in seconds
    t_eval = np.linspace(0, 2000, 1000)  # Time points for evaluation

    # Solve the system
    solution = solve_ivp(
        rocket_equations, t_span, initial_conditions, t_eval=t_eval, args=(F_thrust,), rtol=1e-8, atol=1e-8
    )

    # Extract results
    h_vals = solution.y[0]  # Height values
    v_vals = solution.y[1]  # Velocity values
    m_vals = solution.y[2]  # Mass values
    t_vals = solution.t  # Time values

    # Enforce altitude limit for the stage
    h_vals = np.clip(h_vals, 0, target_altitude)

    # Determine the stopping point if we reach the target altitude
    for i, h in enumerate(h_vals):
        if h >= target_altitude:
            return t_vals[:i+1], h_vals[:i+1], v_vals[:i+1], m_vals[:i+1]

    return t_vals, h_vals, v_vals, m_vals

# Simulate all stages
h_vals_all, v_vals_all, t_vals_all = [], [], []
initial_conditions = [0, 0, mass_initial]  # Start from the ground

for target in altitude_targets:
    t_vals, h_vals, v_vals, m_vals = simulate_stage(initial_conditions, target, F_thrust)

    # Store results for plotting
    if t_vals_all:
        t_vals += t_vals_all[-1]  # Continue time from the last stage

    t_vals_all.extend(t_vals)
    h_vals_all.extend(h_vals)
    v_vals_all.extend(v_vals)

    if h_vals[-1] >= 70000:
      F_thrust = F2
      fuel_consumption_rate = 100
    if t_vals[-1] > 180 and flag:
      F_thrust = F3
      flag = False
      fuel_consumption_rate = 18
    #m_vals[-1] = mass_initial - fuel_consumption_rate * t_vals[-1]


    # Update initial conditions for the next stage
    initial_conditions = [h_vals[-1], v_vals[-1], m_vals[-1]]

# Convert results to numpy arrays
h_vals_all = np.array(h_vals_all)
v_vals_all = np.array(v_vals_all)
t_vals_all = np.array(t_vals_all)

# Загрузка данных из CSV
data = pd.read_csv('Juno.csv')  # Укажите путь к вашему файлу
t_csv = data['TimeSinceMark'].values  # Предполагая, что столбец времени называется 'TimeSinceMark'
v_csv = data['SpeedSurface'].values  # Предполагая, что столбец скорости называется 'SpeedSurface'
h_csv = data['AltitudeASL'].values  # Предполагая, что столбец высоты называется 'AltitudeASL'

# Построение графика скорости от времени
plt.figure(figsize=(10, 6))
plt.plot(t_vals_all, v_vals_all / 1000, label="Скорость (модель) (км/с)", color="blue")
plt.plot(t_csv, v_csv / 1000, label="Скорость (данные) (км/с)", color="red")
plt.xlabel("Время (с)")
plt.ylabel("Скорость (км/с)")
plt.title("Сравнение скорости ракеты во времени")
plt.grid()
plt.legend()
plt.show()

# Построение графика высоты от времени
plt.figure(figsize=(10, 6))
plt.plot(t_vals_all, h_vals_all / 1000, label="Высота (модель) (км)", color="orange")
plt.plot(t_csv, h_csv / 1000, label="Высота (данные) (км)", color="green")
plt.xlabel("Время (с)")
plt.ylabel("Высота")
plt.title("Сравнение высоты ракеты во времени")
plt.grid()
plt.legend()
