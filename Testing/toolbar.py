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

def polyPlotShapeFile(m,ax,sf,crs):
        import numpy as np
        import pyproj
        from matplotlib.collections import LineCollection
        from matplotlib import cm
        import matplotlib.pyplot as plt

        shapes = sf.shapes()
        records = sf.records()

        shp_proj = pyproj.Proj(crs.to_proj4()) #.shp file projection
        std_proj = pyproj.Proj(init = 'epsg:4326') # Lat/Lon system

        for record, shape in zip(records,shapes):
            lons,lats = zip(*shape.points)
            lons,lats = pyproj.transform(shp_proj,std_proj,lons,lats)
            data = np.array(m(lons, lats)).T

            if record['POLY_TYPE'] == 'I':
                if len(shape.parts) == 1:
                    segs = [data,]
                else:
                    segs = []
                    for i in range(1,len(shape.parts)):
                        index = shape.parts[i-1]
                        index2 = shape.parts[i]
                        segs.append(data[index:index2])
                    segs.append(data[index2:])
        
                lines = LineCollection(segs,antialiaseds=(1,),zorder=2,alpha=0.2)
                lines.set_facecolors('b')
                # lines.set_facecolors(cm.jet(np.random.rand(1)))
                lines.set_edgecolors('k')
                lines.set_linewidth(1)
                ax.add_collection(lines)