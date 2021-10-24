import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd

def YT (x,a,b):
    C2 = (2 - np.exp(b)*(-2*np.cos(b)-4*np.sin(b))/5)/(-2*np.exp(-2*b))
    C1 = 1 - C2*np.exp(-2*a) - np.exp(a)*(np.cos(a) - 3*np.sin(a))/5
    y = C1 + C2*np.exp(-2*x) + np.exp(x)*(np.cos(x) - 3*np.sin(x))/5
    return y

def FX (x):
    return -2*np.exp(x)*(np.sin(x) + np.cos(x))

def KM (h, n):
    # Коэффециенты в матрице.
    pi = 2
    qi = 0
    
    km = [[0 for i in range(n+1)] for j in range(n+1)]
    # Граничные условия.
    km[0][0] = 1
    km[-1][-1] = -1 / h**2
    km[-1][-2] =  1 / h**2

    for i in range(1, n):
        km[i][i] = qi * h * h - 2
        km[i][i-1] = 1 - pi * h/2
        km[i][i+1] = 1 + pi * h/2

    return km

def F (h, n, y0, yy0):
    x_list = [a + h*i for i in range(n + 1)]

    f = lambda x: -2*np.exp(x)*(np.sin(x) + np.cos(x))
    fx = [f(x_list[i]) * h * h for i in range(0,n+1)]
    fx[0] = y0
    fx[-1] = yy0
    return fx


n = 10
a = 0
b = 1
h = (b - a) / n

y0 = 1
yy0 = 2

x = [ a + i * h for i in range (n + 1)]
yt = [ YT(i, a, b) for i in x]
km = KM(h, n)
fx = F(h, n, y0, yy0)

u = 1
l = 1
def AB (fx, km, n):

    m = n + 1
    a = np.array(km)
    ab = np.zeros((u + l + 1, m))
    for j in range(m):
        for i in range(m):
            index = u + i - j
            if 0 <= index < u + l + 1:
                ab[index][j] = a[i][j]
    return ab


ab = AB(fx, km, n)
print(ab)
print()
yp = linalg.solve_banded((l, u), ab, fx)

yp = np.linalg.solve(km, fx)

print (km)
print (yp)

n2 = int(n / 2)
h2 = h*2
x2 = [ a + i * h2 for i in range (n2 + 1)]
km2 = KM(h2, n2)
fx2 = F(h2, n2, y0, yy0)

ab2 = AB(fx2, km2, n2)
yp2 = linalg.solve_banded((l, u), ab2, fx2)

# yp2 = np.linalg.solve(km2, fx2)

ytt = np.around(yt, decimals=6)
pogr = [ np.abs(yp[i] - yt[i]) for i in range(n + 1) ]

table_data = []
e = []
p = 1
j = 0
for i in range(n + 1):
    if( i % 2 == 0):
        ee = (yp[i] - yp2[j]) / (2**p - 1)
        e.append(ee)
        table_data.append([x[i], yp[i], ytt[i], pogr[i], x2[j], yp2[j], ee])
        j += 1
    else:
        table_data.append([x[i], yp[i], ytt[i], pogr[i], '-', '-', '-'])

tb = pd.DataFrame(table_data, columns=["X (h)", "Y прог. (h)", "Y теор.", "Погрешность","X (2h)", "Y прог. (2h)", "Правило Рунге"])

pogr_max = np.max(pogr)
e_max = np.max(e)

print ()
print (tb)
print ()
print ('Макс. погрешность: {0}'.format(pogr_max))
print ('Макс. правило Рунге: {0}'.format(e_max))
print ('Шаг (h): {0}'.format(h))
print ()

plt.plot(x, yt)
plt.plot(x, yp)
plt.show()