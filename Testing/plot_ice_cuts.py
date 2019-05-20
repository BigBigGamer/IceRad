import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

plt.rc('text', usetex = True)
plt.rc('font', size=13, family = 'serif')
plt.rc('legend', fontsize=14)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

import numpy as np
from scipy.optimize import curve_fit
import math
import tkinter.filedialog as fd
import toolbar

# pathNS = fd.askdirectory()
pathNS = 'd27m12y2016S014815'

# File reading
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)

size =  thetaNS.shape
for i in range(0,size[1]):
    for j in range(0,size[0]):
        if j < math.floor(size[0]/2):
            thetaNS[j][i] *= -1  

# Ice Detecting
def HypApp(xdata,p1,p2,p3):
    return p1 * abs( 1/(abs(xdata) + p2) ) + p3


plt.figure(1)

colFlag_h = np.zeros((size),dtype = bool)
errh = -1*np.ones(size[1],dtype = np.float32)
for i in range(0,size[1]):
    try:
        # new_psh, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,5,0],bounds =([150,2,-np.inf],[500,7,np.inf]) )
        new_psh, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,5,0])
        diff_h = np.subtract( sigNS[:,i], HypApp( thetaNS[:,i],new_psh[0],new_psh[1],new_psh[2] ))
            
        errh[i] = np.mean( diff_h**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
        if ( new_psh[0] < 2000 ) & ( new_psh[0] > 15 ) & ( new_psh[2] < 100 ) & ( errh[i] < 30 ):
            if (i > 150) and ( i < 250 ):
                plt.plot(thetaNS[:,i],sigNS[:,i],'.')
            colFlag_h[:,i] = True
        else:
            colFlag_h[:,i] = False
    except RuntimeError:
        print('Not fitted_1')

plt.xlabel('$ \\theta, ^{\\circ}$')
plt.ylabel('$ \\sigma^0, dB$')
plt.ylim([-20,40])
plt.grid(which = 'major')
# plt.savefig('ice_cuts.pdf', bbox_inches='tight')

plt.show()
print('Done!')