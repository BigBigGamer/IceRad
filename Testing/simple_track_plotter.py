import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
plt.rc('text', usetex = True)
plt.rc('font', size=13, family = 'serif')
# plt.rc('text.latex',unicode=True)
plt.rc('legend', fontsize=14)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

import numpy as np
from scipy.optimize import curve_fit
import math
import tkinter.filedialog as fd

pathNS = 'd27m12y2016S014815'
sigNS = np.loadtxt(pathNS+'\SigKu.txt')
LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
LoNS =  np.loadtxt(pathNS+'\LoKu.txt')    
thetaNS = np.loadtxt(pathNS+'\IncKu.txt')
size =  thetaNS.shape
for i in range(0,size[1]):
    for j in range(0,size[0]):
        if j < math.floor(size[0]/2):
            thetaNS[j][i] *= -1  

track_num = 162
track_num_w = 250
sig = sigNS[:,track_num]
theta = thetaNS[:,track_num]
sig_w = sigNS[:,track_num_w]
theta_W = thetaNS[:,track_num_w]
plt.plot(theta,sig,'k.-')     
plt.plot(theta_W,sig_w,'r.-')     
plt.xlabel('$\\theta,^{\\circ}$')
plt.ylabel('$\\sigma^0,dB$')
plt.grid(which = 'major')
plt.legend()
plt.show() 