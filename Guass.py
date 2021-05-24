from sympy import *
import numpy as np
class Guass:
    def __init__(self, f, n):
        self.f = f
        self.n = n
        self.x = []
        self.A = []
        self.paras(n)

    def paras(self, n):
        for i in range(n):
            self.x.append(np.cos((2*i+1)/(2*n)*np.pi))
            self.A.append(np.pi/n)

    def Gauss_Chebyshev(self):
        ans = 0
        for i in range(self.n):
            ans += self.A[i]*self.f(self.x[i])
        return ans

def f(x):
    return x**2
n = 3
gauss = Guass(f, 3)
ans = gauss.Gauss_Chebyshev()
print(ans)
x = Symbol('x')
print(integrate(x**2/sqrt(1 - x**2), (x, -1, 1)))