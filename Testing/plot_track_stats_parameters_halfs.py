# Plotting 4 statisticall parameters for the track separated in halfs.
# Mean, Dispersion, Assymetry, Kurtosis

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter.filedialog as fd
import toolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
plt.rc('text', usetex = True)
plt.rc('font', size=30, family = 'serif')
plt.rc('legend', fontsize=30)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

pathNS = 'd27m12y2016S014815'

# Reading files in folder
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)
LoNS = LoNS[:,100:300]
LaNS = LaNS[:,100:300]
sigNS = sigNS[:,100:300]
thetaNS = thetaNS[:,100:300]
size =  thetaNS.shape
for i in range(0,size[1]):
    for j in range(0,size[0]):
        if j < math.floor(size[0]/2):
            thetaNS[j][i] *= -1  

sigNSun = sigNS # dB
sigNSun_up,sigNSun_down = toolbar.getHalfs(sigNSun)

# plt.plot(sigNSun_up[:,60],'rs-')
# plt.plot(sigNSun[:,60],'k.-')
# plt.plot(sigNSun_down[:,60])

thetaNSun = thetaNS/180 *np.pi
thetaNSun_up,thetaNSun_down = toolbar.getHalfs(thetaNSun)

# normalization 
sigNS = np.power(10, sigNS*0.1)
sigNS_up,sigNS_down = toolbar.getHalfs(sigNS)


thetaNS = np.tan( thetaNS/180 * np.pi)
thetaNS_up,thetaNS_down = toolbar.getHalfs(thetaNS)


mu_up = [[],[],[],[]]
mu_down = [[],[],[],[]]

size_half =  thetaNS_up.shape
colFlag_h = np.zeros((size),dtype = bool)
colFlag_h_up = np.zeros((size_half),dtype = bool)
colFlag_h_down = np.zeros((size_half),dtype = bool)


for i in range(0,size_half[1]):
    mu_up_2,mu_up_3,mu_up_4,mu_downs = 0,0,0,0
    # m2,m3,m4 = 0,0,0
    cut_n = i
    y = sigNS_up[:,cut_n] # sig_0
    x = thetaNS_up[:,cut_n] # tan

    mean_up = np.sum([ x[j] * y[j] * np.cos(thetaNSun_up[j][i])**4  for j in range(0,size_half[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun_up[j][i])**4  for j in range(0,size_half[0]) ]) 

    for j in range(0,size_half[0]):
        mu_up_2 += (x[j]- mean_up)**2 * y[j] * np.cos(thetaNSun_up[j][i])**4
        # mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_4 += (x[j]- mean_up)**4 * y[j] * np.cos(thetaNSun_up[j][i])**4
        mu_downs += y[j] * np.cos(thetaNSun_up[j][i])**4

    dispersion_up = np.sqrt(mu_up_2/mu_downs)
    # skewness = (mu_up_3/mu_down) / (dispersion**3)
    kurtosis_up = (mu_up_4/mu_downs) / (dispersion_up**4) - 3 
    
    mu_up[0].append(mean_up)
    mu_up[1].append(dispersion_up)
    # mu_up[2].append(skewness)
    mu_up[3].append(kurtosis_up)



    mu_up_2,mu_up_3,mu_up_4,mu_downs = 0,0,0,0
    y = sigNS_down[:,cut_n] # sig_0
    x = thetaNS_down[:,cut_n] # tan

    mean_down = np.sum([ x[j] * y[j] * np.cos(thetaNSun_down[j][i])**4  for j in range(0,size_half[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun_down[j][i])**4  for j in range(0,size_half[0]) ]) 

    for j in range(0,size_half[0]):
        mu_up_2 += (x[j]- mean_down)**2 * y[j] * np.cos(thetaNSun_down[j][i])**4
        # mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_4 += (x[j]- mean_down)**4 * y[j] * np.cos(thetaNSun_down[j][i])**4
        mu_downs += y[j] * np.cos(thetaNSun_down[j][i])**4

    dispersion_down = np.sqrt(mu_up_2/mu_downs)
    # skewness = (mu_up_3/mu_downs) / (dispersion**3)
    kurtosis_down = (mu_up_4/mu_downs) / (dispersion_down**4) - 3 
    
    mu_down[0].append(mean_down)
    mu_down[1].append(dispersion_down)
    # mu_down[2].append(skewness)
    mu_down[3].append(kurtosis_down)



# Merge up-half and down-half
kurtosis  = np.empty(size)
kurtosis[0:25,:] = mu_up[3]
kurtosis[24:,:] = mu_down[3]
flag = np.zeros(size)

# flag = np.floor(kurtosis/5) # Interesting ##################################
flag[kurtosis>5] = 1


fig = plt.figure(figsize = (20,5))

ax1 = fig.add_subplot(311)
ax11 = fig.add_subplot(312)
# ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(313)


# ax1.plot(np.arange(134,300),mu_up[3][34:],'k-',label = 'Upper half',lw=2)
# ax1.plot(np.arange(134,300),mu_down[3][34:],'r-',label = 'Lower half',lw=2)
# ax1.set_xlim([134,300])
# ax1.set_ylabel('$\\gamma_2$',fontsize = 40)
# ax1.set_xlabel('Scan number',fontsize = 40)
# ax1.grid(which = 'both')
# ax1.legend()

ax11.imshow(kurtosis[:,34:], extent=[134,300, -18,18],aspect = 'auto')


im = ax1.imshow(sigNSun[:,34:], extent=[134,300, -18,18],aspect = 'auto',cmap = 'jet')
# im = ax1.imshow(sigNSun[:,34:], extent=[134,300, -18,18],aspect = 'auto',cmap = 'viridis')
# im = ax1.imshow(sigNSun[:,34:], extent=[134,300, -18,18],aspect = 'auto',cmap = 'BuGn_r')
ax1.set_xlabel('Scan number',fontsize = 40)
ax1.set_ylabel('$\\theta,^{\\circ}$',fontsize = 40)

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax, orientation='vertical')

cbar.ax.set_title('$\sigma^0,dB$',fontsize = 40)

# plt.savefig('imgs/22.png', bbox_inches='tight',dpi=900)
# plt.savefig('imgs/21.png', bbox_inches='tight',dpi=900)
plt.savefig('imgs/21j.png', bbox_inches='tight',dpi=900)

plt.show()

print('Done!')
