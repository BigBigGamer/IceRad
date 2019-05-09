# Supposed to be a test function for edge-detection
import numpy as np
import matplotlib.pyplot as plt

def findEdgesH(sigNS,EdgeF):
    # sigNS - 1xm массив, по сути для выделенного угла.
    # EdgeF - функция скачка 
    
    size = len(sigNS)
    stepsize = 50    # характерный размер скачка
    x = np.arange(-stepsize,stepsize)
    sigma = 6
    step = -np.heaviside(x-stepsize,1) + np.heaviside(x+stepsize,1)
    
    convd = np.convolve(sigNS, EdgeF(x,sigma),'same')   # свертка
    snr = np.abs(convd) / ( np.sqrt( np.convolve(step,sigNS**2,'same') ) ) 

    return snr

Gauss = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) )  # гауссовый скачок


a = np.random.rand(200)
b = findEdgesH(a,Gauss)
plt.plot(a)
plt.plot(b)
plt.show()