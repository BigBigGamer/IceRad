# Function, accepts folder destination and returns separate files
import numpy
from mpl_toolkits.basemap import Basemap

def readFolder(pathNS):
    sigNS = numpy.loadtxt(pathNS+'\SigKu.txt')
    LaNS = numpy.loadtxt(pathNS+'\LaKu.txt')     
    LoNS =  numpy.loadtxt(pathNS+'\LoKu.txt')    
    thetaNS = numpy.loadtxt(pathNS+'\IncKu.txt')
    return sigNS,LaNS,LoNS,thetaNS

def makeMap(extent):
    m = Basemap(llcrnrlon = extent[0], llcrnrlat = extent[1], urcrnrlon = extent[2], urcrnrlat = extent[3],\
            rsphere = (6378137.00, 6356752.3142),\
            resolution = 'h', projection = 'merc',\
            lat_0 = 40., lon_0 = -20., lat_ts = 20.)
    m.fillcontinents(zorder = 0) 
    m.drawcoastlines(zorder = 1)
    return m