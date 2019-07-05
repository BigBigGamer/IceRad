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
plt.rc('legend', fontsize=16)

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


# Ice Detecting
def HypApp(xdata,p1,p2,p3):
    return p1 * abs( 1/(abs(xdata) + p2) ) + p3

def findEdgesH(data,sigma):
    size = len(data)
    stepsize = math.floor(6*sigma)
    x = np.arange(-stepsize,stepsize)
    # sigma = 9
    n0 = 0.5
    gaussD = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) ) # gaussian -derivative
    gaussA = gaussD(np.arange(-stepsize/2, stepsize/2),sigma)
    gaussA_grad = np.gradient(gaussA)
    data_grad = np.gradient(data)

    response = np.convolve(data, gaussA,'same')
    noise = np.sqrt( np.trapz(data**2))

    p1 = abs(np.convolve(data_grad, gaussA_grad,'same'))
    p2 = np.sqrt( np.trapz(data_grad**2))
    loc = p1/(p2*n0**2)
    snr = np.abs(response) / ( noise ) 
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


#### ICE DETECTION ####

# colFlag = np.zeros((size),dtype = bool)

# for i in range(0,size[1]):
#     try:
#         new_parameters, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,5,0],bounds =([150,2,-np.inf],[500,7,np.inf]) )
#         diff = np.subtract( sigNS[:,i], HypApp( thetaNS[:,i],new_parameters[0],new_parameters[1],new_parameters[2] ))
#         err = np.mean( diff**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
#         if ( new_parameters[0] < 2000 ) & ( new_parameters[0] > 15 ) & ( new_parameters[2] < 100 ) & ( err < 30 ):
#             colFlag[:,i] = True
#         else: 
#             colFlag[:,i] = False
#     except RuntimeError:
#         print('Not fitted')


#### ICE DETECTION ####

sigNSun = sigNS # dB
thetaNSun = thetaNS 

sigNS = np.power(10, sigNS*0.1)
thetaNS = np.tan( thetaNS/180 * np.pi)
mu = [[],[],[],[]]
colFlag = np.zeros((size),dtype = bool)
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
    skewness = mu_up_3/mu_down / (dispersion)**3
    kurtosis = mu_up_4/mu_down / (dispersion)**4 - 3 
    
    mu[0].append(mean)
    mu[1].append(dispersion)
    mu[2].append(skewness)
    mu[3].append(kurtosis)
    if kurtosis > 3:
        colFlag[:,i] = True


#### ICE DETECTION END ####



#### EDGE DETECTION ####


parameters = np.zeros(size)  
detectorBig = np.zeros(size)
detectorSmall = np.zeros(size)
maxs = np.zeros(size)    
cvs = np.zeros(size,dtype = bool) 
sigNS_inv = 10**(0.1*sigNSun) 

sigNS = sp.ndimage.filters.gaussian_filter(sigNSun, 0.5, mode='constant')

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
parameters = adjustParameters(parameters,detectorSmall,50,30)
## fixing - stuff
parameters[25,95] = 1
parameters[24,93] = 1
parameters[26,93] = 1
parameters[23,92] = 1

nMap = np.zeros_like(colFlag)   # This is to copy the array, not link it
nMap[:] = colFlag[:]            #

#### FLAG FILLING ####

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

#### FLAG FILLING ####

from matplotlib import colors
cmap = colors.ListedColormap(['#352a86','blue','black'])
# cmap = colors.ListedColormap(['#352a86','#0f5bdd','m'])
norm = colors.BoundaryNorm([0,0.9,1.2,4], cmap.N)


import matplotlib.path as mplPath
import shapefile
import pyproj
import pycrs

data_path = 'E:\Work\GitHub\IceRad_Data\PROCESSED_Data'

full_path = r'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\Planets_files\2016\2016 12 27\planet_okh_20161227_pl_a'

extent = [ 138, 55, 148, 60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = toolbar.makeMap(extent)

# draw parallels
m.drawparallels(np.arange(45,65,2),labels=[1,0,0,1],color = 'grey')
# draw meridians
m.drawmeridians(np.arange(135,165,2),labels=[1,0,0,1],color = 'grey')  

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

LoNS = LoNS[:,30:]
LaNS = LaNS[:,30:]
nMap = nMap[:,30:]
parameters = parameters[:,30:]
# parameters[0,:] = 1
# parameters[48,:] = 1
xm,ym = m(LoNS,LaNS)
toPlot = 5*parameters
iz = plt.scatter(xm[toPlot>0],ym[toPlot>0],15,toPlot[toPlot>0], marker = '.',alpha = 1,cmap = cmap,norm=norm,zorder = 6)
plt.plot(xm[0,:],ym[0,:],'k--')
plt.plot(xm[48,:],ym[48,:],'k--')
# toolbar.polyPlotShapeFile(m,ax,sf,crs)


from mpl_toolkits.axes_grid1 import make_axes_locatable

path = 'E:\Work\GitHub\IceRad_Data\Radiometer_data\m12y2016\d27m12y2016S014815'

areaS1=np.loadtxt(path+r'\areaS1.txt')
areaS2=np.loadtxt(path+r'\areaS2.txt')

LaS1=areaS1[:,0]                        
LoS1=areaS1[:,1]  

LaS2=areaS2[:,0]                        
LoS2=areaS2[:,1]

TcS1 = areaS1[:,2:10]
TcS2 = areaS1[:,2:5]

Boundries=[40, 140, 20, 115]

# %Boundries=[64, 168, 40, 132]; %define boundries of the needed area

S1_titles={'10.65 GHz V-Pol','10.65 GHz H-Pol','18.7 GHz V-Pol',
           '18.7 GHz H-Pol','23.8 GHz V-Pol','36.64 GHz V-Pol',
           '36.64 GHz H-Pol','89.0 GHz V-Pol ','89.0 GHz H-Pol'};

S2_titles={'166.0 GHz V-Pol','166.0 GHz H-Pol','183.31 +/-3 GHz V-Pol','183.31 +/-7 GHz V-Pol'};

la1 = 40
la2 = 64
lo1 = 132
lo2 = 168

# print(TcS1[:,6].shape)
xm,ym = m(LoS1,LaS1)
# im=plt.contour(xm,ym,65,TcS1[:,6])
im=plt.scatter(xm,ym,65,TcS1[:,6],cmap = 'BuGn',alpha = 1)
# plt.title(1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.set_xlabel('$Tb,K$')



plt.savefig('imgs/52.png', bbox_inches='tight',dpi = 900)





plt.show()
print('Done!')