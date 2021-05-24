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




h = 0.1
a, b = 1, 2
u = 1
diff = differential(h, a, b, u)
us, ts = diff.Euler()
diff.draw(us, ts, 'Euler')
us, ts = diff.improved_Euler()
diff.draw(us, ts, 'improved Euler')
us, ts = diff.Runge_Kutta()
diff.draw(us, ts, 'Runge_Kutta')