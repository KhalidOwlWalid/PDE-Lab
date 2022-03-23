import numpy as np
import matplotlib.pyplot as plt
acc = 400
n = 10
k = 2
def u(x,t,n):
    summ = 0
    for n in range (1, n):
        value = 1/(2*k*np.pi)*((-1)**n/(n**3)*(1-np.exp(-4*k*n**2*t))*np.sin(2*n*x))
        summ += value
    summ += 2*x*t/np.pi + 0.5 * np.cos(x) * np.exp(-k*t)
    return summ
t_max = 1
lines = 10
x = np.linspace(0,0.5*np.pi,acc)
t = np.linspace(0,t_max,lines)
R = np.linspace(0,1,lines)
B = np.linspace(1,0,lines)
G = 0
for i in range(0,lines):
    t_new = t[i]
    us = u(x,t_new,n)
    plt.plot(x,us, color = [R[i],G,B[i]])
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend(np.around(t,2), title = "t = ")
plt.show()