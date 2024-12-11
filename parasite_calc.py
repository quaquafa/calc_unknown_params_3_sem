import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import csv

# Набор данных : zond 15 35, zond 20 40, zond 25 45
# Найти; 
# 1) Среднюю функцию, аппроксимировать и найти неизвестные параметры. 
#   Мб попробовать np.mean и np.std (для ошибки)
# 2) Из мараметров найти необходимую ёмкость 

# 15 - 35 МГц


# СНЯТЫЕ ДАННЫЕ #################

prt_15_35_x = np.array([])
realprt_15_35_y = np.array([])
imprt_15_35_y = np.array([])

with open('zond_15_35_copy.csv', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=';')
    
    for ROWS in plotting:
        prt_15_35_x = np.append(prt_15_35_x, float(ROWS[0]))
        realprt_15_35_y = np.append(realprt_15_35_y, (float(ROWS[1])/1.61)/4) # 1.61 - коэффициент масштабирования максимумов
        imprt_15_35_y = np.append(imprt_15_35_y, (float(ROWS[2])/1.61)/4)


# 20 - 40 МГц

prt_20_40_x = np.array([])
realprt_20_40_y = np.array([])
imprt_20_40_y = np.array([])

with open('zond_20_40_copy.csv', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=';')
    
    for ROWS in plotting:
        prt_20_40_x = np.append(prt_20_40_x, float(ROWS[0]))
        realprt_20_40_y = np.append(realprt_20_40_y, float(ROWS[1])/4)
        imprt_20_40_y = np.append(imprt_20_40_y, (float(ROWS[2]))/4)


# 25 - 45 МГц

prt_25_45_x = np.array([])
realprt_25_45_y = np.array([])
imprt_25_45_y = np.array([])

with open('zond_25_45_copy.csv', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=';')
    
    for ROWS in plotting:
        prt_25_45_x = np.append(prt_25_45_x, float(ROWS[0]))
        realprt_25_45_y = np.append(realprt_25_45_y, (float(ROWS[1])/1.07)/4) # 1.07 - коэффициент масштабирования максимумов
        imprt_25_45_y = np.append(imprt_25_45_y, (float(ROWS[2]))/4)



# УСРЕДНЕНИЕ ДАННЫХ #################


realprt_mean_y = np.array([])
imprt_mean_y = np.array([])
prt_mean_x = prt_20_40_x.copy()
error_realprt_mean_y = np.array([])
error_imprt_mean_y = np.array([])

for x in range(251):
    realprt_mean_y = np.append(realprt_mean_y, (realprt_15_35_y[x] + realprt_20_40_y[x] + realprt_25_45_y[x]) / 3)
    imprt_mean_y = np.append(imprt_mean_y, (imprt_15_35_y[x] + imprt_20_40_y[x] + imprt_25_45_y[x]) / 3)
    error_realprt_mean_y = np.append(error_realprt_mean_y, np.std([realprt_15_35_y[x], realprt_20_40_y[x], realprt_25_45_y[x]])/np.sqrt(3))
    error_imprt_mean_y = np.append(error_imprt_mean_y, np.std([imprt_15_35_y[x], imprt_20_40_y[x], imprt_25_45_y[x]])/np.sqrt(3))

error_imprt_mean_y_upper = imprt_mean_y + error_imprt_mean_y
error_imprt_mean_y_lower = imprt_mean_y - error_imprt_mean_y

error_realprt_mean_y_upper = realprt_mean_y + error_realprt_mean_y
error_realprt_mean_y_lower = realprt_mean_y - error_realprt_mean_y


# АППРОКСИМАЦИЯ #################

def real_func(frec, R, L, C, U):
    "Аппроксимирующая функция (действительная)"
    omega = 2 * np.pi * frec
    d1 = (L * R)/C - (R*(omega * L - 1/(omega * C)))/(omega * C)
    d2 = R**2 + (omega * L - 1/(omega * C))**2  
    return U * d1/d2

def im_func(frec, R, L, C, U):
    "Аппроксимирующая функция (мнимая)"
    omega = 2 * np.pi * frec
    d1 = (R**2 / (omega * C)) + (L / C) * (omega * L - 1/(omega * C))
    d2 = R**2 + (omega * L - 1/(omega * C))**2
    return -U * d1/d2


fitxreal, _ = sp.curve_fit(real_func, prt_mean_x, realprt_mean_y, p0=(1.9, 560e-9, 41e-12, 0))
fitxim, _ = sp.curve_fit(im_func, prt_mean_x, imprt_mean_y, p0=(1.9, 560e-9, 41e-12, 0))

errfitxrealupper, _ = sp.curve_fit(real_func, prt_mean_x, error_realprt_mean_y_upper, p0=(1.9, 560e-9, 41e-12, 0))
errfitxreallower, _ = sp.curve_fit(real_func, prt_mean_x, error_realprt_mean_y_lower, p0=(1.9, 560e-9, 41e-12, 0))

errfitximupper, _ = sp.curve_fit(im_func, prt_mean_x, error_imprt_mean_y_upper, p0=(1.9, 560e-9, 41e-12, 0))
errfitximlower, _ = sp.curve_fit(im_func, prt_mean_x, error_imprt_mean_y_lower, p0=(1.9, 560e-9, 41e-12, 0))


print(f"value of unknowns params(R, L, C):{[(fitxreal[i] + fitxim[i])/2 for i in range(3)]}")

print(f"standard error:{[((np.abs(errfitxrealupper[i] - errfitxreallower[i])) + (np.abs(errfitximupper[i] - errfitximlower[i])))/4 for i in range(3)]}") # SE = sigma / sqrt(n)


# ГРАФИКА #################

fig, axs = plt.subplots(1, 2, figsize=(18, 6))



plt.xlim(3e7, 3.5e7)

axs[0].get_yaxis().set_visible(False)
axs[1].get_yaxis().set_visible(False)

axs[0].plot(prt_15_35_x, realprt_15_35_y, ':b', label='15-35 МГц')
axs[0].plot(prt_20_40_x, realprt_20_40_y, ':r', label='20-40 МГц')
axs[0].plot(prt_25_45_x, realprt_25_45_y, ':g', label='25-45 МГц')
axs[0].plot(prt_mean_x, realprt_mean_y, color='black', label='Среднее значение')
axs[1].plot(prt_15_35_x, imprt_15_35_y, ':b')
axs[1].plot(prt_20_40_x, imprt_20_40_y, ':r')
axs[1].plot(prt_25_45_x, imprt_25_45_y, ':g')
axs[1].plot(prt_mean_x, imprt_mean_y, color='black')

axs[0].plot(prt_mean_x, real_func(prt_mean_x, fitxreal[0], fitxreal[1], fitxreal[2], fitxreal[3]), 'y', label='Аппроксимация Re(Z)')
axs[1].plot(prt_mean_x, im_func(prt_mean_x, fitxim[0], fitxim[1], fitxim[2], fitxim[3]), 'c', label='Аппроксимация Im(Z)')

axs[0].grid(which='major')
axs[1].grid(which='major')

fig.legend(fontsize=15)

plt.show()

