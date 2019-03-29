import numpy as np
import pandas as pd
import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
from mpl_toolkits.basemap import Basemap
# import tkinter.filedialog as fd
import matplotlib.pyplot as plt





data_path = 'E:\Work\GitHub\IceRad\PROCESSED_Data'
year = '2017'
month = '03'
days = ['01','02']
# days = ['28','29','30','31']

extent = [ 120,130,50,60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=130.,llcrnrlat=40.,urcrnrlon=160.,urcrnrlat=65.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\
            lat_0=40.,lon_0=-20.,lat_ts=20.)

# m.drawcoastlines()
# m.fillcontinents()
# draw parallels
m.drawparallels(np.arange(40,60,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(130,150,10),labels=[1,1,0,1])   

filenames = [ day + 'm' + month + 'y' + year for day in days]
# filenames = ['d'+ day + 'm' + month + 'y' + year for day in days]
folder = data_path + r'/IceMaps/m%s'%month+'y%s'%year
files = os.listdir(folder)
for file in files:
    if any(filename in file for filename in filenames):
        cur_file = pd.read_csv(folder+'/'+file, delimiter = '\t')
        # names = ['La','Lo','Sig','Theta','Ice']
        print('got One!')
        La = cur_file.La.values.tolist()
        Lo = cur_file.Lo.values.tolist()
        Sig = cur_file.Sig.values.tolist()
        xm,ym = m(Lo,La)
        m.scatter(xm,ym,5,marker='.',color = 'green')

# print(cur_file)
df = pd.read_csv(data_path + r'/Planets_files\2017 03 28\planet_okh_20170328_pl_a.txt',sep=';')
# create new figure, axes instances.
# Tk().withdraw()
# pathNS = fd.askdirectory() 
# pathNS = 'DataProcessing/Hydro/NS/m03y2018/d04m03y2018S133003'
# LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
# LoNS =  np.loadtxt(pathNS+'\LoKu.txt')  
# sigNS =  np.loadtxt(pathNS+'\sigKu.txt') 
x, y = m(df['Lon'].values.tolist(),df['Lat'].values.tolist())
# xm,ym = m(LoNS,LaNS)
# m.pcolormesh(xm,ym,sigNS,cmap = 'jet')
# m.scatter(xm,ym,5,sigNS, marker = '.',alpha =0.9,cmap = 'jet')
# m.imshow(sigNS,extent=extent, alpha=0.6)
m.plot(x,y,'.',markersize = 0.3,color='r')
plt.title(days[0]+'.'+ month + '.' + year  )
plt.show()