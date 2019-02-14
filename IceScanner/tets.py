# libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal
# xs = np.arange(0, 5*np.pi, 0.05)
# data = np.sin(xs)
# sigNS = np.sin(xs)**2 + np.sin(0.5*xs)

# x = np.arange(-0.5*np.pi, 0.5*np.pi, 0.05)
# sigma = 0.3

# Gauss = -x * np.exp( -x**2 / ( 2*sigma**2 ) )
# convd = np.convolve( Gauss,sigNS,'same')
# convd2 = np.convolve( sigNS,Gauss,'same')

# def findEdgesH(sigNS):
#     size = len(sigNS)
#     stepsize = 50
#     x = np.arange(-stepsize,stepsize)
#     sigma = 6
#     step = np.heaviside(-stepsize,1) - np.heaviside(stepsize,1)
#     Gauss = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) )
#     convd = np.convolve(sigNS, Gauss(x,sigma),'same')
#     # snr = np.abs(convd) / ( np.sqrt( sp.integrate.quad( Gauss**2,  ) ) )
#     print(np.convolve(step,sigNS**2,'same'))
#     snr = np.abs(convd) / ( np.sqrt( np.convolve(step,sigNS**2,'same') ) ) 
#     return snr



# stepsize = 50
# x=100
# step = -np.heaviside(x-stepsize,1) + np.heaviside(x+stepsize,1)
# print(step)

# print(np.convolve(step,sigNS**2,'same'))

# peakind,_ = signal.find_peaks(data)
# print(peakind)
# plt.plot(xs,sigNS)
# plt.plot(x,Gauss)
# plt.plot(xs,convd)
# plt.plot(xs[peakind],data[peakind],'r.')
# plt.show(block = False)
# plt.figure()
# plt.plot(xs,sigNS)

# plt.plot(xs,convd2)
# plt.show()
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# xdata = np.linspace(0, 4, 50)
# y = func(xdata, 2.5, 1.3, 0.5)
# ydata = y 


# popt, pcov = curve_fit(func, xdata, ydata)

# plt.plot(xdata,func(xdata, 2.5, 1.3, 0.5),'.')
# plt.show(block = False)
# plt.plot(xdata,func(xdata,popt[0],popt[1],popt[2]))
# plt.show()
# print(25)


a = np.array([1,0,1,2])
a = a.reshape(len(a),1)
print(a)
b = np.empty((3,4))
b = np.repeat(a,3,axis = 1)
print(b)