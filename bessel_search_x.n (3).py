import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jn_zeros

g = 9.8 #m/s^2
OMEGA = 100 #rad/s
MASS = 0.2#kg
LENGTH = 2 #m
DENSITY = 0.0001 #kg/m

k_n= 2*OMEGA/np.sqrt(g/(LENGTH+MASS/DENSITY))

all_roots = jn_zeros(0,1000)
nearest_index = np.argmin(np.abs(all_roots - k_n))
nearest_root = all_roots[nearest_index]

print(f'Число x_n:{k_n:.10f}')
print(f'Ближайший корень функции Бессселя:{nearest_root:.10f}')
print(f'Номер этого корня:{nearest_index + 1}')

x_min = 0
x_max = LENGTH
x_array = np.linspace(x_min, x_max, 1000)
y = jv(0, nearest_root * np.sqrt((LENGTH + MASS/DENSITY - x_array)/(LENGTH + MASS/DENSITY)))

arg_start = nearest_root
arg_end = nearest_root * np.sqrt((MASS/DENSITY) / (LENGTH + MASS/DENSITY))
total_half_waves = (arg_start - arg_end) / np.pi

print(f'количество полуволн:{total_half_waves:.10f}')

plt.plot(x_array,y)
plt.title(f'OMEGA={OMEGA}rad/s, MASS={MASS}kg, LENGTH={LENGTH}m, DENSITY={DENSITY}kg/m, число полуволн={total_half_waves} ')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()