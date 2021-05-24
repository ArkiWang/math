import numpy as np
from numpy.linalg import det

'''
x = np.matrix([[0, -1/3, -1/3], [0, 0, -2], [0, 2/3, 2/3]])
#x = np.matrix([[3, 1, 1], [0, 1, 2], [1, 0, 0.5]])
#x = np.matrix([[5,2,1], [-2,-8,-3], [1,-1,-4]])
#x = np.matrix([[1,-9,-7], [0,2,2], [0,-2,1]])
# 计算矩阵特征值与特征向量
D_L = [[5, 0, 0], [-2, -8, 0], [1, -1, -4]]
U = [[0, -2, -1], [0, 0, 3], [0, 0, 4]]
D_L = [[1, 0, 0], [0, 2, 0], [0, -2, 1]]
U = [[0,9,7],[0,0,-2],[0,0,0]]
B = np.linalg.inv(D_L)*U
e, v = np.linalg.eig(B)
print(e, v)

t = 2 + np.sqrt(5)
res = (3 - t)*(1 - t)/4
print(res)

x = 1.5
res = 1/3*(1+x**2)**(-2/3)*2*x
print(res)

res = 1/2*(x**3 - 1)**(-1/2)*3*x**2
res = -1/2*(x-1)**(-3/2)
print(res)
B = np.matrix([[0, 2, -2],[0, 2, -1],[0, 8, -6]])
f = np.matrix([[1], [2], [7]])
x = np.matrix([[1],[2], [3]])
#x = 1
for i in range(3):
    #x = (1+x**2)**(1/3)
    #x = x - (x**3 + 2*x**2 +10*x - 20)/(3*x**2 + 4*x+10)
    x = B*x + f
    print(x)
'''

D = np.matrix([[2, 0, 0],[0, 1, 0], [0, 0, -2]])
L = np.matrix([[0, 0, 0], [-1, 0, 0], [-1, -1, 0]])
U = np.matrix([[0, 1, -1], [0, 0, -1], [0, 0, 0]])
D
#B = np.linalg.inv(D)*(L+U)
B = np.linalg.inv(D-L)*U
e, v = np.linalg.eig(B)
print(e)

'''
x = np.array([[0], [0]])
r = np.array([[0], [1]])
A = np.array([[6,3],[3, 2]])
p = r
b = r
print(p)
print((A.dot(p)))

print(np.dot((A.dot(p)).T, p).reshape(-1)[0])
while r.any() != 0 and np.dot((A.dot(p)).T, p).reshape(-1)[0]!= 0:
    alpha = np.dot(r.T, r).reshape(-1)[0]/np.dot((A.dot(p)).T, p).reshape(-1)[0]
    x = x + alpha*p
    r_ = r
    r = r - alpha*A.dot(p)
    beta = np.dot(r.T, r).reshape(-1)[0]/np.dot(r_.T, r_).reshape(-1)[0]
    p = r + beta*p
    print(x)
print(A.dot(x))
'''

deta = np.array([[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]])
print(det(deta))

A = np.array([[5, 15], [15, 55]])
b = np.array([[29], [104.2]])
x = np.linalg.solve(A, b)
print(x)