from cmath import sqrt
from statistics import mean


def calcu_lxx(x: []):
    x_av = mean(x)
    ans = 0
    for i, e in enumerate(x):
        ans += e**2 - x_av**2
    return ans

def calcu_lxy(x: [], y: []):
    x_av = mean(x)
    y_av = mean(y)
    n = len(x)
    ans = 0
    for i in range(n):
        ans += x[i]*y[i] - x_av*y_av
    return ans

def calcu_sigma2(n, x_av, y_av, lxx, lxy, lyy):
    beta1 = lxy/lxx
    beta0 = y_av - beta1 * x_av
    sum_xy = lxy + n * x_av * y_av
    sum_x2 = lxx + n * x_av ** 2
    sum_y2 = lyy + n * y_av ** 2
    sigma2 = sum_y2 - n * beta0**2 - 2 * beta0 * beta1 * n * x_av - beta1**2 * sum_x2
    return sigma2/(n-2)

def calcu_sigma2_2(n, sum_x2, sum_xy, sum_y2, sum_x, sum_y):
    x_av, y_av = sum_x/n, sum_y/n
    lxx = sum_x2 - n * x_av**2
    lxy = sum_xy - n * x_av * y_av
    lyy = sum_y2 - n * y_av**2
    beta1 = lxy / lxx
    beta0 = y_av - beta1 * x_av
    print("beta", beta1, beta0)
    sigma2 = calcu_sigma2(n, x_av, y_av, lxx, lxy, lyy)
    print(beta1**2 * lxx/ sigma2)
    y0 = beta0 + beta1 * 225
    print(y0 - 1.96 * sqrt(sigma2), y0 + 1.96 * sqrt(sigma2))
    return sigma2

def __main__():
    n = 16
    x_av, y_av = 153.625, 94.4375
    lxx, lxy, lyy = 609.75, 438.625, 339.9375
    print(calcu_sigma2(n, x_av, y_av, lxx, lxy, lyy))

    n = 12
    sum_x, sum_y = 643, 753
    sum_x2, sum_xy, sum_y2 = 34843, 40830, 48139
    print(calcu_sigma2_2(n, sum_x2, sum_xy, sum_y2, sum_x, sum_y))

    x = [40, 20, 25, 20, 30, 50, 40, 20, 50, 40, 25, 50]
    y = [385, 400, 395, 365, 475, 440, 490, 420, 560, 525, 480, 510]
    x = [104, 180, 190, 177, 147, 134, 150, 191, 204, 121]
    y = [100, 200, 210, 185, 155, 135, 170, 205, 235, 125]
    n = len(x)
    lxx, lxy, lyy = calcu_lxx(x), calcu_lxy(x, y), calcu_lxx(y)
    x_av, y_av = mean(x), mean(y)
    beta1 = lxy/lxx
    beta0 = y_av - beta1 * x_av
    print("beta", beta1, beta0)
    sigma2 = calcu_sigma2(n, x_av, y_av, lxx, lxy, lyy)
    print("sigma2", sigma2)
    y0 = beta0 + beta1 * 160
    print(y0 - 1.96*sqrt(sigma2), y0 + 1.96*sqrt(sigma2))
    print(beta1**2 * lxx/ sigma2)
    t = beta0/(sqrt(sigma2)*sqrt(1/n + x_av**2/lxx))
    print(t)
    print(beta1 - 2.2281*sqrt(sigma2)/sqrt(lxx), beta1 + 2.2281*sqrt(sigma2)/sqrt(lxx))
    print(1.96*sqrt(sigma2))

    n = 12
    sum_x, sum_y = 643, 753
    sum_x2, sum_xy, sum_y2 = 34843, 40830, 48139
    sum_x, sum_y = 2460, 871.2
    sum_x2, sum_xy, sum_y2 = 518600, 182943, 64572.94
    print("sigma2", calcu_sigma2_2(n, sum_x2, sum_xy, sum_y2, sum_x, sum_y))

    x = [-2, -1, 0, 1, 2]
    y = [-3.1, -0.9, 1, 3.1, 4.9]
    lxx, lxy = calcu_lxx(x), calcu_lxy(x , y)
    beta1 = lxy/lxx
    beta0 = mean(y) - beta1 * mean(x)
    print("result: beta1 {} beta0 {}".format(beta1, beta0))


if __name__ == "__main__":
    __main__()

