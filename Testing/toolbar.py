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
    m.drawcoastlines(zorder = 3)
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
        
                lines = LineCollection(segs,antialiaseds=(1,),zorder=1,alpha=0.2)
                lines.set_facecolors('b')
                # lines.set_facecolors(cm.jet(np.random.rand(1)))
                lines.set_edgecolors('k')
                lines.set_linewidth(1)
                ax.add_collection(lines)

def borderPlotShapeFile(m,ax,sf,crs):
    import numpy as np
    import pyproj
    import matplotlib.pyplot as plt
    
    shapes = sf.shapes()
    records = sf.records()

    shp_proj = pyproj.Proj(crs.to_proj4()) #.shp file projection
    std_proj = pyproj.Proj(init = 'epsg:4326') # Lat/Lon system
    
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
                c = 'b'
                m.scatter(x_lon,y_lat,0.5,marker = '.',color = c)