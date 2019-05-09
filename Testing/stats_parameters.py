import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter.filedialog as fd
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
plt.rc('text', usetex = True)
plt.rc('font', size=13, family = 'serif')
# plt.rc('text.latex',unicode=True)
plt.rc('legend', fontsize=14)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

pathNS = 'd27m12y2016S014815'
# pathNS = 'd25m03y2017S0026'

# pathNS =fd.askdirectory() 
# File reading
sigNS = np.loadtxt(pathNS+'\SigKu.txt')
LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
LoNS =  np.loadtxt(pathNS+'\LoKu.txt')    
thetaNS = np.loadtxt(pathNS+'\IncKu.txt')
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
    # moments calculation 

    # mean

    mean = np.sum([ x[j] * y[j] * np.cos(thetaNSun[j][i])**4  for j in range(0,size[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun[j][i])**4  for j in range(0,size[0]) ]) 

    for j in range(0,size[0]):
        mu_up_2 +=(x[j]- mean)**2 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_4 += (x[j]- mean)**4 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_down += y[j] * np.cos(thetaNSun[j][i])**4

    dispersion = np.sqrt(mu_up_2/mu_down)
    # print(dispersion)
    skewness = (mu_up_3/mu_down) / (dispersion**3)
    kurtosis = (mu_up_4/mu_down) / (dispersion**4) - 3 
    
    mu[0].append(mean)
    mu[1].append(dispersion)
    mu[2].append(skewness)
    mu[3].append(kurtosis)
    # print(mean)
    if i==162:
        print('Ice ',kurtosis)
    if i==250:
        print('Water ',kurtosis)
    if kurtosis > 10**8:
        # print(kurtosis)
        colFlag_h[:,i] = True
    else:
        colFlag_h[:,i] = False



fig = plt.figure()
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)

# ax1.plot(mu[0],label = 'Среднее')
# ax1.plot(mu[1],label = 'Дисперсия' )
# ax1.plot(mu[2],label = 'Коэффициент асимметрии')
ax1.plot(mu[3],'r.-')
ax1.set_xlim([0,size[1]])
ax1.set_ylabel('$\\gamma_2$')
# print(colFlag_h)
# ax1.imshow(colFlag_h,extent=[0,size[1], 0,size[0]],aspect = 'auto')
# ax1.set_title(r'Коэффициент эксцесса')
# ax1.legend()

# ax2.imshow(sigNSun,extent=[0,size[1], 0,size[0]],aspect = 'auto',cmap = 'jet')
# ax2.set_title('Base Data')
# ax2.set_xlabel('№ swath')
plt.show()

print('Done!')
