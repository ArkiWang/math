from cmath import sin

import numpy as np
from sympy import *
'''
def f(x):
    if x == 0:
        x = Symbol('x')
        return limit(f(x), x, 0)
    return sin(x)/x
    
a = 0
b = np.pi/2
n = 7
h = (b - a)/n
x = [h*i+a for i in range(n+1)]
'''
def f(x):
    return x**2 * sin(x)

a, b = -2, 2
n = 80
h = (b - a)/n
x = [h*i +a for i in range(n+1)]


def T(n, f, x:[], a, b):
    ans = 0
    for i in range(1, n):
        ans += 2*f(x[i])
    ans += f(a) + f(b)
    ans *= (b-a)/(2*n)
    return ans
print(T(n, f, x, a, b))

def S(n, f, x:[], a, b):
    ans = 0
    for i in range(1, n):
        ans += 2*f(x[i]) + 4*f((x[i] + x[i+1])/2)
    ans += f(a) + 4*f((x[0] + x[1])/2) + f(b)
    ans *= (b - a)/(6*n)
    return ans
print(S(n, f, x, a, b))

x = Symbol('x')
print(integrate(f(x), (x, a, b)))

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
y = [1.2, 1.45, 1.25, 1.3, 1.25, 1.05, 1.1]
n = len(x) - 1
a, b = 0, 0.6

def T_(n, y:[], a, b):
    ans = 0
    for i in range(1, n):
        ans += 2*y[i]
    ans += y[0] + y[-1]
    ans *= (b-a)/(2*n)
    return ans
print(T_(n, y, a, b))

def S_(n, y:[], a, b):
    ans = 0
    for i in range(1, n):
        ans += 2*y[i] + 4*((y[i] + y[i+1])/2)
    ans += y[0] + 4*((y[0] + y[1])/2) + y[-1]
    ans *= (b - a)/(6*n)
    return ans
print(S_(n, y, a, b))



