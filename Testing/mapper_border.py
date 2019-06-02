import numpy as np
# import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
from mpl_toolkits.basemap import Basemap
# import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import shapefile
import pyproj
import pycrs
import math
from scipy import signal
import toolbar

from scipy.optimize import curve_fit

from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

plt.rc('text', usetex = True)
plt.rc('font', size=13, family = 'serif')
plt.rc('legend', fontsize=14)

import scipy as sp
from scipy import signal

data_path = 'E:\Work\GitHub\IceRad_Data\PROCESSED_Data'

full_path = r'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\Planets_files\2016\2016 12 27\planet_okh_20161227_pl_a'

extent = [ 120,130,50,60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = toolbar.makeMap([135,41,160,63])

#reading shapefile
sf = shapefile.Reader(full_path+'.shp')
#reading projection file
crs = pycrs.load.from_file(full_path+'.prj')
ice_shapes_ind = []
shp_proj = pyproj.Proj(crs.to_proj4()) #.shp file projection
std_proj = pyproj.Proj(init = 'epsg:4326') # Lat/Lon system        

print(sf)

shapes = sf.shapes()
records = sf.records()
length = len(records)

# draw parallels
m.drawparallels(np.arange(40,80,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(130,170,10),labels=[1,1,0,1])   

pathNS = 'd27m12y2016S014815'
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)
size =  thetaNS.shape

for i in range(0,size[1]):
    for j in range(0,size[0]):
        if j < math.floor(size[0]/2):
            thetaNS[j][i] *= -1  
sigNSun = sigNS # dB
thetaNSun = thetaNS 
# normalization 
sigNS = np.power(10, sigNS*0.1)
thetaNS = np.tan( thetaNS/180 * np.pi)
mu = [[],[],[],[]]
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
for i in range(0,size[0]):
    #using function .find_peaks
    detectorBig[i,:] = findEdgesH(sigNS[i,:],6)
    detectorSmall[i,:] = findEdgesH(sigNS[i,:],1.1)
    detectorBig[i,:] = detectorBig[i,:]* 100 / np.amax(detectorBig[i,:])
    detectorSmall[i,:] = detectorSmall[i,:]* 100 / np.amax(detectorSmall[i,:])

cvs = detectorSmall

parameters = adjustParameters(parameters,detectorSmall,60,15)
for i in range(0,size[0]):
    #using function .find_peaks
    detectorSmall[i,:] = findEdgesH(sigNS_inv[i,:],5)
    detectorSmall[i,:] = detectorSmall[i,:]* 100 / np.amax(detectorSmall[i,:])

# cvs = detectorSmall
parameters = adjustParameters(parameters,detectorSmall,50,30)
## fixing - stuff
parameters[25,95] = 1
parameters[24,93] = 1
parameters[26,93] = 1
parameters[23,92] = 1

nMap = np.zeros_like(colFlag)   # This is to copy the array, not link it
nMap[:] = colFlag[:]            #
# flag filling
for i in range(0,size[0]):
    BorderIndex = np.nonzero(parameters[i,:])
    BorderIndex = np.append(BorderIndex,size[1])
    BorderIndex = np.insert(BorderIndex,0,0)
    zAm=0
    iAm=0
    for k in range(0,len(BorderIndex)-1):
        for m in range(BorderIndex[k],BorderIndex[k+1]):
            if nMap[i][m] == 0:
                zAm +=1
            if nMap[i][m] == 1:
                iAm +=1
        Ams = [zAm,iAm]
        # maxAm = np.amax(Ams)
        nMap[i,BorderIndex[k]:BorderIndex[k+1]] = np.argmax(Ams)
        zAm=0
        iAm=0


xm,ym = m(LoNS,LaNS)
im = plt.scatter(xm,ym,25,parameters, marker = '.',alpha = 1,cmap = 'viridis')

if False:
    for some_shape in list(sf.iterShapes()):
        npoints=len(some_shape.points)
        nparts=len(some_shape.parts)

        if nparts == 1:
            length = len(some_shape.points)
            x_lon = np.zeros((length,1))
            y_lat = np.zeros((length,1))
            for i in range(0,length):
                x_lon[i] = some_shape.points[i][0]
                y_lat[i] = some_shape.points[i][1]
            plt.plot(x_lon,y_lat)
        else:
            for j in range(nparts):
                i0 = some_shape.parts[j]
                if j < nparts-1:
                    i1 = some_shape.parts[j+1]-1
                else:
                    i1 = npoints
                seg = some_shape.points[i0:i1+1]
                x_lon = np.zeros((len(seg),1))
                y_lat = np.zeros((len(seg),1))
                for i in range(0,len(seg)):
                    x_lon[i] = some_shape.points[i][0]
                    y_lat[i] = some_shape.points[i][1]
                x_lon,y_lat = pyproj.transform(shp_proj,std_proj,x_lon,y_lat)
                x_lon,y_lat = m(x_lon,y_lat)
                c = 'r'
                m.scatter(x_lon,y_lat,1,marker = '.',color = c)

plt.show()
