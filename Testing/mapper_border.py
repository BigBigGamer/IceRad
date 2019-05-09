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


def findEdgesH(sigNS):
    size = len(sigNS)
    stepsize = 50
    x = np.arange(-stepsize,stepsize)
    sigma = 6
    step = -np.heaviside(x-stepsize,1) + np.heaviside(x+stepsize,1)
    Gauss = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) )
    convd = np.convolve(sigNS, Gauss(x,sigma),'same')
    snr = np.abs(convd) / ( np.sqrt( np.convolve(step,sigNS**2,'same') ) ) 
    return snr


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
colFlag_h = np.zeros((size),dtype = float)
parameters = np.zeros(size,dtype=bool)          
for i in range(0,size[0]):
    detector = findEdgesH(sigNS[i,:])    
    peakind,_ = signal.find_peaks( detector )
    maxs = detector[peakind]* 100 / np.amax(detector[peakind])
    for j in range(0,len(maxs)):
        if maxs[j] > 20:
            parameters[i][peakind[j]] = True


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
