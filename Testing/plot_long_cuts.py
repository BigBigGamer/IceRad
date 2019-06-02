import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

plt.rc('text', usetex = True)
plt.rc('font', size=13, family = 'serif')
plt.rc('legend', fontsize=14)
# plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

import numpy as np
import math
import tkinter.filedialog as fd
import toolbar
import scipy.signal as sp

def findEdgesH(sigNS):
    size = len(sigNS)
    stepsize = 50
    x = np.arange(-stepsize,stepsize)
    sigma = 6
    step = -np.heaviside(x - stepsize,1) + np.heaviside(x + stepsize,1)
    Gauss = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) )
    convd = np.convolve(sigNS, Gauss(x,sigma),'same')
    # snr = np.abs(convd) / ( np.sqrt( sp.integrate.quad( Gauss**2,  ) ) )
    # print(np.convolve(step,sigNS**2,'same'))
    snr = np.abs(convd) / ( np.sqrt( np.convolve(step,sigNS**2,'same') ) ) 
    # snr2 = np.abs(convd) / ( np.sqrt( np.trapz( sigNS**2 ) ) ) 
    return snr


def movingAverage(data,window):
    size = data.shape
    print(size)
    data_new = np.zeros(size)
    for i in range(0 + window, size[1] - window):
        data_new[:,i] = np.sum(data[:, i - window : i + window],1)/window/2
    return data_new

# pathNS = fd.askdirectory()
pathNS = 'd27m12y2016S014815'

# File reading
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)

# sigNSS = sp.savgol_filter(sigNS,11,3)
sigNSS = movingAverage(sigNS,4)

fig = plt.figure(figsize=(13, 4))
for i in range(25,26,1):
    # plt.plot(sigNS[i,100:400])
    detector = findEdgesH(sigNS[i,:]) 

    sss = '$\\sigma^{0}_{dB}(\\theta = $'+ str(math.floor(thetaNS[i,0])) + '$^{\\circ})$'
    plt.plot(range(100,400),sigNSS[i,100:400],label = sss,color = 'black')
    plt.plot(range(100,400),4*detector[100:400],label = '$S(x)$')
plt.axvline(135.5,0,1,linestyle = '--',color = 'k')
plt.axvline(195.2,0,1,linestyle = '--',color = 'k')
plt.plot(135.5,22.45,'k.',markersize = 20)
plt.plot(195.2,4.27,'k.',markersize = 20)



plt.grid(which='major', b = True)
plt.minorticks_on()
plt.ylabel('$\sigma^0 , dB$')
# plt.grid(which='minor', b =True, linestyle = '--')
plt.legend(loc = 'upper right')
plt.savefig('long2.pdf', bbox_inches='tight')
plt.show()