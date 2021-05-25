import numpy as np
x = [0, 2, 3, 4, 5, 7, 8]

def f(x):
    return x**3 + 70*x**2 + 10*x + 5


def divide_difference(x):
    n = len(x)
    ans = []
    for l in range(1, n+1):
        tmp = []
        if l == 1:
            for i in range(n+1 - l):
                tmp.append(f(x[i]))
        else:
            for i in range(len(ans[-1]) - 1):
                tmp.append((ans[-1][i+1] - ans[-1][i])/(x[i+l-1] - x[i]))
        ans.append(tmp)
    return ans

ans = divide_difference(x)
print(ans)


