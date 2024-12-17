v_ksp_max = 2496.22692338753 # из данных KSP
v_mat_max = 2849.0 # из данных мат модели

abs_error = abs(v_mat_max - v_ksp_max)
rel_error = round(abs_error / v_ksp_max * 100, 2)
print(f"Относительная погрешность расчетов = {rel_error}%")

