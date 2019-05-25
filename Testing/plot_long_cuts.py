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
for i in range(5,26,5):
    # plt.plot(sigNS[i,100:400])
    sss = '$\\theta = $'+ str(math.floor(thetaNS[i,0])) + '$^{\\circ}$'
    plt.plot(range(100,400),sigNSS[i,100:400],label = sss)

plt.grid(which='major', b = True)
plt.minorticks_on()
plt.ylabel('$\sigma^0 , dB$')
# plt.grid(which='minor', b =True, linestyle = '--')
plt.legend(loc = 'upper right')
plt.savefig('long.pdf', bbox_inches='tight')
plt.show()