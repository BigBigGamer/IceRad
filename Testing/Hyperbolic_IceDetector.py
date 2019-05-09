import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from scipy import signal
import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
from mpl_toolkits.basemap import Basemap
# import tkinter.filedialog as fd
import matplotlib.path as mplPath
import shapefile
import pyproj
import pycrs
import toolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable


# pathNS = 'E:\Work\GitHub\IceRad\Testing\d01m01y2017S003600'
pathNS = 'd27m12y2016S014815'
# pathNS = 'd07m02y2017S1336'
# pathNS = 'E:\Work\GitHub\IceRad\Testing\d03m01g17S172400'
# pathNS = 'E:\Work\GitHub\IceRad\Testing\d31m03y2017S213254'

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

def findEdgesH(sigNS):
    size = len(sigNS)
    stepsize = 50
    x = np.arange(-stepsize,stepsize)
    sigma = 6
    step = -np.heaviside(x-stepsize,1) + np.heaviside(x+stepsize,1)
    Gauss = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) )
    convd = np.convolve(sigNS, Gauss(x,sigma),'same')
    # snr = np.abs(convd) / ( np.sqrt( sp.integrate.quad( Gauss**2,  ) ) )
    # print(np.convolve(step,sigNS**2,'same'))
    snr = np.abs(convd) / ( np.sqrt( np.convolve(step,sigNS**2,'same') ) ) 
    # snr2 = np.abs(convd) / ( np.sqrt( np.trapz( sigNS**2 ) ) ) 
    return snr

colFlag = np.zeros((size),dtype = bool)
errh = -1*np.ones(size[1],dtype = np.float32)
for i in range(0,size[1]):
    try:
        new_psh, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,5,0])
        diff_h = np.subtract( sigNS[:,i], HypApp( thetaNS[:,i],new_psh[0],new_psh[1],new_psh[2] ))
        errh[i] = np.mean( diff_h**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
        if ( new_psh[0] < 2000 ) & ( new_psh[0] > 15 ) & ( new_psh[2] < 100 ) & ( errh[i] < 30 ):        
            colFlag[:,i] = True
        else: 
            colFlag[:,i] = False
    except RuntimeError:
        print('Not fitted_1')

parameters = np.zeros(size,dtype=bool)          
for i in range(0,size[0]):
    detector = findEdgesH(sigNS[i,:])    
    peakind,_ = signal.find_peaks( detector )
    maxs = detector[peakind]* 100 / np.amax(detector[peakind])
    for j in range(0,len(maxs)):
        if maxs[j] > 20:
            parameters[i][peakind[j]] = True


nMap = np.zeros_like(colFlag)   # This is to copy the array, not link it
nMap[:] = colFlag[:]  

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



data_path = 'E:\Work\GitHub\IceRad_Data\PROCESSED_Data'

full_path = r'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\Planets_files\2016\2016 12 27\planet_okh_20161227_pl_a'
# full_path = r'C:\Work\Python\IceRad_Data\PROCESSED_Data\Planets_files\2017\2017 02 07\planet_okh_20170207_pl_a'

extent = [ 135, 53, 150, 63 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = toolbar.makeMap(extent)

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

m.drawcoastlines()
m.fillcontinents()
# draw parallels
# m.drawparallels(np.arange(40,80,10),labels=[1,1,0,1])
# draw meridians
# m.drawmeridians(np.arange(130,170,10),labels=[1,1,0,1])   

xm,ym = m(LoNS,LaNS)
# print(colFlag_h[:,150])
nMap[20:29,120:190] = True
im = plt.scatter(xm[nMap==True],ym[nMap==True],20,'blue', marker = '.',alpha = 1)

if True:
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
                # print('plottoing',x_lon)

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("bottom", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax, orientation='horizontal')

plt.show()

print('Done!')