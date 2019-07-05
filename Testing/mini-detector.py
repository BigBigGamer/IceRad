import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import toolbar
import math
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

plt.rc('text', usetex = True)
plt.rc('font', size=18, family = 'serif')
plt.rc('legend', fontsize=14)

import scipy as sp
from scipy import signal
import cv2

# pathNS = 'E:\Work\GitHub\IceRad\Testing\d01m01y2017S003600'
pathNS = 'E:\Work\GitHub\IceRad\Testing\d27m12y2016S014815'

# File reading
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)
sigNS = sigNS[:,100:300]
LaNS = LaNS[:,100:300]
LoNS = LoNS[:,100:300]
thetaNS = thetaNS[:,100:300]
size =  thetaNS.shape


weights = np.array([[0, 0, 1, 0, 0],
                    [0, 2, 4, 2, 0],
                    [1, 4, 8, 4, 1],
                    [0, 2, 4, 2, 0],
                    [0, 0, 1, 0, 0]],
                   dtype=np.float)
weights = weights / np.sum(weights[:])


# Ice Detecting
def HypApp(xdata,p1,p2,p3):
    return p1 * abs( 1/(abs(xdata) + p2) ) + p3

def findEdgesH(data,sigma):
    size = len(data)
    stepsize = math.floor(6*sigma)
    x = np.arange(-stepsize,stepsize)
    # sigma = 9
    n0 = 0.5
    # step = -np.heaviside(x-stepsize,1) + np.heaviside(x+stepsize,1)
    gaussD = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) ) # gaussian -derivative
    # gaussD = lambda x,s: -np.heaviside(x-stepsize,1) + np.heaviside(x,2) -np.heaviside(x+stepsize,1) # step
    gaussA = gaussD(np.arange(-stepsize/2, stepsize/2),sigma)
    gaussA_grad = np.gradient(gaussA)
    data_grad = np.gradient(data)

    response = np.convolve(data, gaussA,'same')
    # noise = np.sqrt( np.convolve(step,sigNS**2,'same')
    noise = np.sqrt( np.trapz(data**2))

    p1 = abs(np.convolve(data_grad, gaussA_grad,'same'))
    p2 = np.sqrt( np.trapz(data_grad**2))
    loc = p1/(p2*n0**2)
    # snr = np.abs(convd) / ( np.sqrt( sp.integrate.quad( Gauss**2,  ) ) )
    # print(np.convolve(step,sigNS**2,'same'))
    snr = np.abs(response) / ( noise ) 
    # snr2 = np.abs(convd) / ( np.sqrt( np.trapz( sigNS**2 ) ) ) 
    # plt.figure(22)
    # plt.plot(gaussA)
    return snr*loc

def adjustParameters(parameters,detector,upper,lower):
    size = parameters.shape
    for i in range(0,size[0]):
        peakind,_ = signal.find_peaks( detector[i,:] )
    
        for j in peakind:
            maxs[i][j] = detector[i][j]
            # strong border
            if maxs[i][j] > upper:
                parameters[i][j] = True

        for j in peakind:
            # middle
            if (maxs[i][j] < upper) and (maxs[i][j] > lower):
                for s_x in [-1,0,1]:

                    if not(parameters[i][j]):
                        for s_y in [-1,0,1]:
                            if (j + s_y >= size[1]-1) or (j + s_y < 0):
                                break
                            if (i + s_x > 48) or (i + s_x < 0):
                                break
                            if parameters[i+s_x][j+s_y]:
                                parameters[i][j] = True
                                break
    return parameters


colFlag = np.zeros((size),dtype = bool)

for i in range(0,size[1]):
    try:
        new_parameters, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,5,0],bounds =([150,2,-np.inf],[500,7,np.inf]) )
        diff = np.subtract( sigNS[:,i], HypApp( thetaNS[:,i],new_parameters[0],new_parameters[1],new_parameters[2] ))
        err = np.mean( diff**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
        if ( new_parameters[0] < 2000 ) & ( new_parameters[0] > 15 ) & ( new_parameters[2] < 100 ) & ( err < 30 ):
            colFlag[:,i] = True
        else: 
            colFlag[:,i] = False
    except RuntimeError:
        print('Not fitted')

parameters = np.zeros(size)  
detectorBig = np.zeros(size)
detectorSmall = np.zeros(size)
maxs = np.zeros(size)    
cvs = np.zeros(size,dtype = bool) 
sigNS_inv = 10**(0.1*sigNS) 

sigNS = sp.ndimage.filters.gaussian_filter(sigNS, 0.5, mode='constant')
# import scipy   
# from scipy import ndimage
# for i in range(0,size[0]):
#     cvss = 255 * sigNS[i,:]/np.amax(sigNS[i,:])
#     cvss = cvss.astype(np.uint8)
#     cvs[i,:] = cv2.Canny(cvss,150,140).ravel()
# plt.imshow(cvs)

for i in range(0,size[0]):
    #using function .find_peaks
    detectorBig[i,:] = findEdgesH(sigNS[i,:],6)
    detectorSmall[i,:] = findEdgesH(sigNS[i,:],1.1)
    detectorBig[i,:] = detectorBig[i,:]* 100 / np.amax(detectorBig[i,:])
    detectorSmall[i,:] = detectorSmall[i,:]* 100 / np.amax(detectorSmall[i,:])

# cvs = detectorSmall

# parameters = adjustParameters(parameters,detectorBig,60,30)
parameters = adjustParameters(parameters,detectorSmall,60,15)


for i in range(0,size[0]):
    #using function .find_peaks
    detectorSmall[i,:] = findEdgesH(sigNS_inv[i,:],5)
    detectorSmall[i,:] = detectorSmall[i,:]* 100 / np.amax(detectorSmall[i,:])

# cvs = detectorSmall
parameters = adjustParameters(parameters,detectorSmall,50,30)
# parameters = adjustParameters(parameters,detectorBig,30,1)

# cvs = detectorSmall

## fixing - stuff
parameters[25,95] = 1
parameters[24,93] = 1
parameters[26,93] = 1
parameters[23,92] = 1

nMap = np.zeros_like(colFlag)   # This is to copy the array, not link it
nMap[:] = colFlag[:]       
     #
# flag filling
# for i in range(0,size[0]):
#     BorderIndex = np.nonzero(parameters[i,:])
#     BorderIndex = np.append(BorderIndex,size[1])
#     BorderIndex = np.insert(BorderIndex,0,0)
#     zAm=0
#     iAm=0
#     for k in range(0,len(BorderIndex)-1):
#         for m in range(BorderIndex[k],BorderIndex[k+1]):
#             if nMap[i][m] == 0:
#                 zAm +=1
#             if nMap[i][m] == 1:
#                 iAm +=1
#         Ams = [zAm,iAm]
#         # maxAm = np.amax(Ams)
#         nMap[i,BorderIndex[k]:BorderIndex[k+1]] = np.argmax(Ams)
#         zAm=0
#         iAm=0



fig = plt.figure(figsize=(10, 5))


from matplotlib import colors
cmap = colors.ListedColormap(['#352a86','#0f5bdd','yellow'])
norm = colors.BoundaryNorm([0,0.9,1.2,4], cmap.N)

# plt.imshow(colFlag + 5*parameters,cmap = cmap,norm = norm,extent = [100,300,0,49])

# plt.subplot(3,1,1)
plt.imshow(nMap + 5*parameters,cmap = cmap,norm = norm,extent = [130,240,-18,18])
# plt.title('my')
# plt.subplot(3,1,2)
# plt.title('cv2')
# plt.subplot(3,1,3)
# plt.imshow(sigNS)

plt.ylabel('$\\theta, ^{\\circ}$')
# plt.savefig('imgs/4.png', bbox_inches='tight')
plt.xlabel('Scan number')

plt.show()
print('Done!')