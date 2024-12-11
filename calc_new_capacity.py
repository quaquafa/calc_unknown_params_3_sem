import numpy as np


res_f = 13.56e6
res_omega = 2 * np.pi * res_f
C_parasite = 43.6e-12
C_hight = 1.5e-9
C_final = 250.5e-12
L = 549e-9
R = 52.5
# U = R / np.sqrt(R ** 2 + (res_omega * L - 1 / (res_omega * C) - 1/(C_hight * res_omega) - 1/(C_parasite * res_omega) ) ** 2)

a = C_parasite + C_hight
b = C_parasite * C_hight

print(1/(1/(b * C_final) - a/(b**2)))
print((a * C_final)/(a - C_final))