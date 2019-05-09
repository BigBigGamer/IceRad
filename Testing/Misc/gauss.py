# Simple Gauss's derivative plotter
import numpy as np
import matplotlib.pyplot as plt

Gauss = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) ) 

xs = np.arange(-10,10,0.01)
ys = Gauss(xs,1.5)
plt.axhline(y=0, color='grey')
plt.axvline(x=0, color='grey')
plt.plot(xs,ys,'k-')
plt.ylim([-3,3])
plt.show()