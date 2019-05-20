# Plotting 4 statisticall parameters for the track.
# Mean, Dispersion, Assymetry, Kurtosis

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter.filedialog as fd
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
import toolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('text', usetex = True)
plt.rc('font', size=13, family = 'serif')
plt.rc('legend', fontsize=14)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

pathNS = 'd27m12y2016S014815'
# pathNS = 'd25m03y2017S0026'
# pathNS =fd.askdirectory() 

# Reading files in folder
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)

size =  thetaNS.shape
for i in range(0,size[1]):
    for j in range(0,size[0]):
        if j < math.floor(size[0]/2):
            thetaNS[j][i] *= -1  
sigNSun = sigNS # dB
thetaNSun = thetaNS/180 *np.pi 
# normalization 
sigNS = np.power(10, sigNS*0.1)
thetaNS = np.tan( thetaNS/180 * np.pi)
mu = [[],[],[],[]]
colFlag_h = np.zeros((size),dtype = bool)
for i in range(0,size[1]):
    mu_up_2,mu_up_3,mu_up_4,mu_down = 0,0,0,0
    m2,m3,m4 = 0,0,0
    cut_n = i
    y = sigNS[:,cut_n] # sig_0
    x = thetaNS[:,cut_n] # tan

    mean = np.sum([ x[j] * y[j] * np.cos(thetaNSun[j][i])**4  for j in range(0,size[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun[j][i])**4  for j in range(0,size[0]) ]) 

    for j in range(0,size[0]):
        mu_up_2 +=(x[j]- mean)**2 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_4 += (x[j]- mean)**4 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_down += y[j] * np.cos(thetaNSun[j][i])**4

    dispersion = np.sqrt(mu_up_2/mu_down)
    skewness = (mu_up_3/mu_down) / (dispersion**3)
    kurtosis = (mu_up_4/mu_down) / (dispersion**4) - 3 
    
    mu[0].append(mean)
    mu[1].append(dispersion)
    mu[2].append(skewness)
    mu[3].append(kurtosis)

    if kurtosis > 10**8:
        colFlag_h[:,i] = True
    else:
        colFlag_h[:,i] = False



fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# ax1.plot(mu[0],label = 'Mean')
# ax1.plot(mu[1],label = 'Dispersion' )
# ax1.plot(mu[2],label = 'Assymetry')
ax1.plot(mu[3],'k-')
ax1.set_xlim([0,size[1]])
ax1.set_ylabel('$\\gamma_2$')
ax1.grid(which = 'both')
# ax1.legend()

im = ax2.imshow(sigNSun,extent=[0,size[1], 0,size[0]],aspect = 'auto',cmap = 'jet')

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# axins = inset_axes(ax2,
#                    width="5%",  # width = 5% of parent_bbox width
#                    height="50%",  # height : 50%
#                    loc='lower left',
#                    bbox_to_anchor=(1.05, 0., 1, 1),
#                    bbox_transform=ax2.transAxes,
#                    borderpad=0,
#                    )

# divider = make_axes_locatable(ax2)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(im, cax = cax, orientation='vertical')

# cbar.ax.set_xlabel('$\sigma^0,dB$')

# plt.savefig('kurt2.pdf', bbox_inches='tight')

plt.show()

print('Done!')
