import collections
#modify N your self
N = 10**6

Sn = 1/2*(3/2-1/N-1/(N+1))
Sn1, Sn2 = 0, 0
for i in range(2, N+1):
    Sn1 += 1/(i**2 - 1)

for i in range(N, 1, -1):
    Sn2 += 1/(i**2 - 1)

print("1.1 Sn: {}, Sn1: {}, Sn2: {}".format(Sn, Sn1, Sn2))

#秦九韶
n = 4
a = [7, 3, -5, 11]
x = 23
ans = 0
for i in range(n):
    ans = ans * x + a[i]

print("1.2 ans {}".format(ans))
#=======================================
import numpy as np
#print("2.1")
#backward
def backward(A, b, r):
    x = [0] * r
    for i in range(r - 1, -1, -1):
        tmp = b[i]
        for j in range(i, r):
            tmp -= A[i][j] * x[j]
        x[i] = tmp / A[i][i]
    return x

def forward(A, b, r):
    x = [0] * r
    for i in range(r):
        tmp = b[i]
        for j in range(i+1):
            tmp -= A[i][j] * x[j]
        x[i] = tmp / A[i][j]
    return x

from copy import deepcopy
class Matrix_Solver:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.r = len(A)
        self.c = len(A[0])

    def Gaussian(self):
        A, r = deepcopy(self.A), self.r
        for i in range(r):
            for k in range(i + 1, r):
                A[k] = A[k] - A[k][i] / A[i][i] * A[i]
        return A

    def get_max_abs_ii(self, A, r, c):
        ans = A[r][c]
        pos = r
        for i in range(r + 1, len(A)):
            if abs(ans) < abs(A[i][c]):
                ans, pos = A[i][c], i
        return pos

    def Gaussian_with_pivot(self):
        A = deepcopy(self.A)
        for i in range(r):
            ex = self.get_max_abs_ii(A, i, i)
            A[i], A[ex] = A[ex], A[i]
            for k in range(i + 1, r):
                A[k] = A[k] - A[k][i] / A[i][i] * A[i]
        return A


print("2.1")
A = np.array([[31, -13, 0, 0, 0, -10, 0, 0, 0, -15],
              [-13, 35, -9, 0, -11, 0, 0, 0, 0, 27],
              [0, -9, 31, -10, 0, 0, 0, 0, 0, -23],
              [0, 0, -10, 79, -30, 0, 0, 0, -9, 0],
              [0, 0, 0, -30, 57, -7, 0, -5, 0, -20],
              [0, 0, 0, 0, -7, 47, -30, 0, 0, 12],
              [0, 0, 0, 0, 0, -30, 41, 0, 0, -7],
              [0, 0, 0, 0, -5, 0, 0, 27, -2, 7],
              [0, 0, 0, -9, 0, 0, 0, -2, 29, 10]])
b = np.array([-15, 27, -23, 0, -20, 12, -7, 7, 10]).T
r, c = len(A), len(A[0])
mat_s = Matrix_Solver(A, b)
A = mat_s.Gaussian()
x = backward(A, b, r)
print(x)
A = mat_s.Gaussian_with_pivot()
x = backward(A, b, r)
print(x)

#==================================
print("2.2")
A = np.array([[1, 2, -5, 1],
              [1, -5, 2, 7],
              [0, 2, 1, -1],
              [1, 7, -1, -4]])
b = np.array([13, -9, 6, 0]).T
r = len(A)
c = len(A[0])

from cmath import sqrt
class decompostion:
    def __init__(self, A):
        self.A = A
        self.r = len(A)
        self.c = len(A[0])

    def cholesky(self):
        A, r, c = deepcopy(self.A), self.r, self.c
        L = np.zeros((r, c), complex)
        for j in range(c):
            tmp = 0
            for k in range(j):
                tmp += L[j][k] ** 2
            L[j][j] = sqrt(A[j][j] - tmp)
            tmp = 0
            for i in range(j + 1, r):
                for k in range(j):
                    tmp += L[i][k] * L[j][k]
                L[i][j] = (A[j][j] - tmp) / L[j][j]
        return L

    def LU_decompose(self):
        A, r, c = deepcopy(self.A), self.r, self.c
        L, U = np.zeros((r, c)), np.zeros((r, c))
        for i in range(r):
            L[i][i] = 1
        for j in range(c):
            U[0][j] = A[0][j]
            L[j][0] = A[j][0]/U[0][0]

        for i in range(r):
            for j in range(i, c):
                U[i][j] = A[i][j] - sum([L[i][k]*U[k][j] for k in range(i)])
                L[i][j] = (A[i][j] - sum(L[j][k]*U[k][i] for k in range(i)))/U[i][i]

        return L, U




de = decompostion(A)
L = de.cholesky()
print(L)
y = forward(L, b, r)
print(y)
x = backward(L.T, y, r)
print(x)
L, U = de.LU_decompose()
y = forward(L, b, r)
x = backward(U, y, r)
print("L", L)
print("U", U)
print("x", x)

#============================================
class iteration_solver:

    def __init__(self, A, N):
        D, L, U = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
        for i in range(len(A)):
            D[i][i] = A[i][i]
            for j in range(i):
                L[i][j] = -A[i][j]
            for k in range(i + 1, len(A)):
                U[i][k] = -A[i][k]
        self.N = N
        self.D, self.L, self.U = D, L, U

    def jacobi(self):
        D, L, U = self.D, self.L, self.U
        B = np.linalg.inv(D).dot(L + U)
        f = np.linalg.inv(D).dot(b)
        x = [0] * N
        x = np.array(x).T
        for i in range(10000):
            x = np.dot(B, x) + f
        print("jacobi", x)

    def Gauss_Seidel(self):
        D, L, U = self.D, self.L, self.U
        B = np.linalg.inv(D - L).dot(U)
        f = np.linalg.inv(D - L).dot(b)
        x = [0] * N
        x = np.array(x).T
        for i in range(100000):
            x = np.dot(B, x) + f
        print("Gauss - Seidel", x)

def buildMatrix(N):
    A = [[0] * N for _ in range(N)]
    b = [1] * N
    b[0], b[-1] = 2, 2
    for i in range(N):
        A[i][i] = 3
        if i + 1 < N: A[i][i+1] = -1
        if i > 0: A[i][i-1] = -1
    return np.array(A), np.array(b).T

N = 100
A, b = buildMatrix(N)
print("2.7")
print("solve", np.linalg.solve(A,b))
iter_s = iteration_solver(A, N)
iter_s.jacobi()
iter_s.Gauss_Seidel()

#========================
# binary search
import matplotlib.pyplot as plt

def y(x):
    return  x * np.cos(x) + 2

l, h = -4, -2
def binary(l, h, y):
    mid = (l + h) / 2
    precision = 0.0000000000000001
    while y(mid) < precision:
        mid = (l + h) / 2
        y_l = y(l)
        y_mid = y(mid)
        y_h = y(h)
        if y_l * y_mid < 0:
            h = mid
        if y_mid * y_h < 0:
            l = mid
    return mid

ans = binary(l, h, y)
print("3.2", ans)

def f(x):
    return x**3 + 2*x**2 + 10*x - 100

def f_1(x):
    return 3*x**2 + 4*x + 10

def Newton_iter(f, f_1, N, x0, m=1):
    x = x0
    xs, ys = [x], [f(x)]
    for _ in range(N):
        x = x - m * f(x)/f_1(x)
        xs.append(x)
        ys.append(f(x))
    return x, xs, ys

def secant_iter(f, N, x0, x1):
    xs, ys = [x0, x1], [f(x0), f(x1)]
    for _ in range(N):
        x2 = x1 - f(x1) / ((f(x1) - f(x0)) / (x1 - x0) if abs(x1 - x0) != 0 else 0.000000001)
        x0 = x1
        x1 = x2
        xs.append(x2)
        ys.append(f(x2))

    return x2, xs, ys

def draw(xs, ys, title):
    plt.plot(xs, ys, 'r', marker='o', linewidth=1)
    xs = np.linspace(0, 10, 1000)
    ys = [f(x) for x in xs]
    plt.plot(xs, ys, 'b')
    plt.title(title)
    plt.show()

N = 1000
x0, x1 = 0, 1
ans, xs, ys = Newton_iter(f, f_1, N, x0)
print(ans)
draw(xs, ys, "Newton")
ans, xs, ys = secant_iter(f, N, x0, x1)
print(ans)
draw(xs, ys, "Secant")
#===============================

import matplotlib.pyplot as plt
from sympy import *
def g(x):
    return 54 * x**6 + 45 * x**5 - 102 * x**4 - 69 * x**3 + 35 * x**2 + 16 * x  - 4

def g_1(x):
    return 324 * x**5 + 225 * x**4 - 408 * x**3 - 207 * x**2 + 70 * x + 16

def solve_3_6():
    x = np.linspace(-2, 2, 100)
    y = g(x)
    plt.plot(x, y, 'r', linewidth=2)
    plt.show()
    x, m = 0, 5
    precision = 0.00001
    cnt = 0
    ans = []
    while cnt < 5:
        x = x - m * g(x) / g_1(x)
        if abs(g(x)) < precision:
            cnt += 1
            ans.append(x)
    cnt = 0
    x0, x1 = -2, 2
    ans1 = []
    while cnt < 5:
        x2 = x1 - g(x1)/((g(x1) - g(x0))/(x1 - x0))
        if abs(g(x2)) < precision:
            cnt += 1
            ans1.append(x2)
        x0 = x1
        x1 = x2
    xs = [-2 / 3, 1 / 2, -5 / (3 * (-1 / 2 - sqrt(3) * 1j / 2) * (9 / 2 + sqrt(419) * 1j / 2) ** (1 / 3)) - (
                -1 / 2 - sqrt(3) * 1j / 2) * (9 / 2 + sqrt(419) * 1j / 2) ** (1 / 3) / 3,
          -(-1 / 2 + sqrt(3) * 1j / 2) * (9 / 2 + sqrt(419) * 1j / 2) ** (1 / 3) / 3 - 5 / (
                      3 * (-1 / 2 + sqrt(3) * 1j / 2) * (9 / 2 + sqrt(419) * 1j / 2) ** (1 / 3)),
          -(9 / 2 + sqrt(419) * 1j / 2) ** (1 / 3) / 3 - 5 / (3 * (9 / 2 + sqrt(419) * 1j / 2) ** (1 / 3))]
    e, e1 = [], []
    for i in range(5):
        e.append(ans[i] - xs[i])
        e1.append(ans1[i] - xs[i])
    rat, rat_2 = [], []
    rat1, rat1_2 = [], []
    for i in range(4):
        rat.append(e[i+1]/e[i])
        rat_2.append(e[i+1]/e[i]**2)
        rat1.append(e1[i+1]/e1[i])
        rat1_2.append(e1[i+1]/e1[i]**2)
    return ans, ans1, rat, rat_2, rat1, rat1_2
ans, ans1, rat, rat_2, rat1, rat1_2 = solve_3_6()
#print("3.6 \nans:{}\n ans1: {}\n rat:{}\n rat_2:{}\n rat1:{}\n rat1_2:{}".format(ans, ans1, rat, rat_2, rat1, rat1_2))
x = Symbol('x')
res = solve(54 * x ** 6 + 45 * x ** 5 - 102 * x ** 4 - 69 * x ** 3 + 35 * x ** 2 + 16 * x - 4, x)
#print(res)
#================================

class Lagrange:
    def __init__(self, f, interval, Range):
        self.f = f
        l, h = Range
        self.n = int((h - l)/interval)
        self.x_list = [l + i*interval for i in range(self.n)]
        self.Range = Range

    def w_k(self, x:[], k):
        ans = 1
        for i, e in enumerate(x):
            ans *= (x[k] - e)
        return ans

    def l_k(self, X:[], k, x):
        ans = 1
        for i in range(len(X)):
            if i != k:
                ans *= (x - X[i])
                ans /= (X[k] - X[i])
        return ans

    def p_n(self, x):
        ans = 0
        X = self.x_list
        for i in range(self.n):
            ans += self.f(X[i]) * self.l_k(X, i, x)
        return ans

    def draw_figure(self):
        x = np.linspace(self.Range[0], self.Range[1], 10000)
        y0 = [self.f(xi) for xi in x]
        plt.plot(x, y0, 'r')
        y1 = [self.p_n(xi) for xi in x]
        plt.plot(x, y1, 'b')
        plt.show()

def f(x):
    return 1/(1+x**2)
def f_1(x):
    return -2*x/(1 + x**2)**2
interval = 2
Range = (-5, 5)
lag = Lagrange(f, interval, Range)
lag.draw_figure()
#==============================
class cubic_spline:

    def __init__(self, f,f_1, interval, Range):
        self.f = f
        self.f_1 = f_1
        l, h = Range
        self.n = int((h - l) / interval) - 1
        self.x = [l + i * interval for i in range(self.n + 1)]
        self.y = [f(xi) for xi in self.x]
        self.Range = Range
        self.calcu_para(self.x, self.y)

    def s(self, x):
        X, Y, h= self.x, self.y, self.h
        m = self.first_border()
        ans = []
        for k in range(self.n - 1):
            ans.append((h[k] + 2*(x - X[k]))/h[k]**3 * (x - X[k+1])**2 * Y[k]
        + (h[k] - 2*(x - X[k+1]))/h[k]**3 * (x - X[k])**2 * Y[k+1]
        + (x - X[k])*(x - X[k+1])**2/h[k]**2 * m[k]
        + (x - X[k+1])*(x - X[k])**2/h[k]**2 * m[k+1])
        return ans

    def calcu_para(self, X: [], Y: []):
        n = self.n + 1
        h, lambd, miu, g = [0]*n, [0]*n, [0]*n, [0]*n

        for i in range(self.n):
            h[i] = X[i + 1] - X[i]
        print("h", h)
        for i in range(1, self.n):
            lambd[i] = h[i]/(h[i] + h[i-1])
            miu[i] = h[i-1]/(h[i] + h[i-1])
        for i in range(1, self.n):
            g[i] = 3*(miu[i]*(Y[i+1] - Y[i])/h[i] + lambd[i]*(Y[i] - Y[i-1])/h[i-1])
        self.h, self.lambd, self.miu, self.g = h, lambd, miu, g

    def first_border(self):
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            A[i][i] = 2
            if i+1 < self.n - 1:A[i][i+1] = self.miu[i+1]
            if i-1 >= 0:A[i][i - 1] = self.lambd[i+1]
        b = np.zeros((self.n, 1))
        for i in range(self.n):
            b[i][0] = self.g[i+1]
        b[0][0] -= self.lambd[1]*self.f_1(self.x[0])
        b[-1][0] -= self.miu[-1]*self.f_1(self.x[-1])
        m = np.linalg.solve(A, b)
        return m.reshape(-1)

    def draw(self):
        x = np.linspace(self.Range[0], self.Range[1], 10000)
        y0 = [self.f(xi) for xi in x]
        plt.plot(x, y0, 'r')
        y1 = [self.s(xi) for xi in x]
        plt.plot(x, y1, 'b')
        plt.show()

cb = cubic_spline(f, f_1, interval, Range)
cb.draw()
#=================
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

#========================
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

#================================
from cmath import exp
import matplotlib.pyplot as plt

def f(t, u):
    return 2/t * u + t**2 * exp(t)

class differential:
    def __init__(self, h, a, b, u0):
        self.h = h
        self.a = a
        self.b = b
        self.u0 = u0
        self.N = int((b - a) / h)

    def Euler(self):
        a, b, h, N = self.a, self.b, self.h, self.N
        u, t = self.u0, a
        us, ts = [u], [t]
        for _ in range(N):
            u = u + h * f(t, u)
            t += h
            us.append(u.real)
            ts.append(t)
        return us, ts

    def improved_Euler(self):
        a, b, h, N = self.a, self.b, self.h, self.N
        u, t = self.u0, a
        us, ts = [u], [t]
        for _ in range(N):
            u_ = u + h*f(t, u)
            u = u + h/2*(f(t, u) + f(t+h, u_))
            t += h
            us.append(u)
            ts.append(t)
        return us, ts

    def Runge_Kutta(self):
        a, b, h, N = self.a, self.b, self.h, self.N
        u, t = self.u0, a
        us, ts = [u], [t]
        for _ in range(N):
            k1 = f(t, u)
            k2 = f(t + 1/2 * h, u + 1/2 * h * k1)
            k3 = f(t + 1/2 * h, u + 1/2 * h * k2)
            k4 = f(t + h, u + h * k3)
            u = u + h/6 * (k1 + 2*k2 + 2*k3 +k4)
            t += h
            us.append(u)
            ts.append(t)
        return us, ts

    def draw(self, us: [], ts: [], tittle):
        plt.plot(ts, us, 'b', marker='o')
        plt.title(tittle)
        plt.show()




h = 0.01
a, b = 1, 2
u = 1
diff = differential(h, a, b, u)
us, ts = diff.Euler()
diff.draw(us, ts, 'Euler')
us, ts = diff.improved_Euler()
diff.draw(us, ts, 'improved Euler')
us, ts = diff.Runge_Kutta()
diff.draw(us, ts, 'Runge_Kutta')





























