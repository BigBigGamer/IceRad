import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

plt.rc('text', usetex = True)
plt.rc('font', size=30, family = 'serif')
plt.rc('legend', fontsize=23)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

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

fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(111)
for i in range(24,25,1):
    # plt.plot(sigNS[i,100:400],label = )
    detector = findEdgesH(sigNS[i,:]) 

    sss = '$\\sigma^{0}_{dB}(\\theta = $'+ str(math.floor(thetaNS[i,0])) + '$^{\\circ})$'
    a = ax1.plot(range(160,300),sigNSS[i,160:300],label = sss)
    ax2 = ax1.twinx()
    b = ax2.plot(range(160,300),0.5*detector[160:300],label = '$S(x)$',color = 'red')
# plt.axvline(135.5,0,1,linestyle = '--',color = 'k')
ax1.axvline(196.53,0,1,linestyle = '--',color = 'k')
# plt.plot(135.5,22.45,'k.',markersize = 20)
ax2.plot(196.53,0.87,'k.',markersize = 20)



ax2.set_ylim([0,1])

ax1.grid(which='major', b = True)
ax1.minorticks_on()
ax1.set_ylabel('$\sigma^0 , dB$',fontsize = 40)
ax1.set_xlabel('Scan number',fontsize = 40)
ax2.set_ylabel('$S(x)$',fontsize = 40)
# plt.grid(which='minor', b =True, linestyle = '--')
c = a+b
cabs = [l.get_label() for l in c]
ax1.legend(c,cabs,loc = 'upper right')
# ax2.legend(loc = 'upper right')
# plt.savefig('imgs/31.png', bbox_inches='tight',dpi=900)
plt.savefig('imgs/32.png', bbox_inches='tight',dpi=900)
plt.show()