# Simple Gauss's derivative plotter
import numpy as np
import matplotlib.pyplot as plt

Gauss = lambda x,s: 1/(s* np.sqrt(2*np.pi)) * np.exp( -x**2 / ( 2*s**2 ) ) 
Weigner = lambda x,R: 1/(np.pi*R**2) * np.sqrt( R**2 - (x/3)**2 )  
Laplace = lambda x,b: 1/(2*b) * np.exp( - np.abs(x)/(b) ) 

xs = np.arange(-10,10,0.001)
plt.axhline(y=0, color='grey')
plt.axvline(x=0, color='grey')
ys = Gauss(xs,2)
plt.plot(xs,ys,'k-')
plt.grid(which='major', b = True)
plt.minorticks_on()

ys = Weigner(xs,2)
plt.plot(xs,ys,'-',color = 'magenta')

ys = 1.1* Laplace(xs,2)
plt.plot(xs,ys,'-', color = 'blue')

# plt.ylim([-3,3])
plt.show()