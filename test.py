from sympy import *
import numpy as np
x = Symbol('x')
res = solve(54 * x**6 + 45 * x**5 - 102 * x**4 - 69 * x**3 + 35 * x**2 + 16 * x  - 4, x)
print(res)
res = solve(x**2 + 1, x)
for r in res:
    print(r)

def g(x):
    return 54 * x**6 + 45 * x**5 - 102 * x**4 - 69 * x**3 + 35 * x**2 + 16 * x  - 4

'''
from scipy.optimize import fsolve
r = fsolve(g, -2)
print(r)
'''

import matplotlib.pyplot as plt
import math
import random
n = 15
x = np.linspace(0, 1, n)
y = x**2
plt.plot(x, y, 'r', linewidth=2)
plt.show()

n = 100
x = np.linspace(0, 1, n)
y = [xi**2 for xi in x]
lxx = sum([xi**2 for xi in x]) - n * np.mean(x)**2
lxy = sum([x[i]*y[i] for i in range(n)]) - n * np.mean(x) * np.mean(y)
c2 = lxy/lxx
c1 = np.mean(y) - c2*np.mean(x)
print("c1: {} c2: {}".format(c1, c2))

