from statistics import mean

data = [[370,420,450,490,500,450],
[490,380,400,390,500,410],
[330,340,400,380,470,360],
[410,480,400,420,380,410]]

data = [[77, 86, 81, 88, 83],
[95, 92, 78, 96, 89],
[71, 76, 68, 81, 74],
[80, 84, 79, 70, 82]]

k = len(data)
n = len(data[0])

sumd = 0
for i in range(k):
    sumd += sum(data[i])
print(sumd)
x__mean = sumd/(k*n)
print(x__mean)
SSe, SSt = 0, 0
for i in range(k):
    xi_mean = mean(data[i])
    SSt += (x__mean - xi_mean)**2
    print(xi_mean)
    for j in range(n):
        SSe += (data[i][j] - xi_mean)**2
SSt *= n
print(SSt, SSe)




