import numpy as np
import os
import tkinter.filedialog as fd
import shapefile
import matplotlib.path as mplPath
import h5py as hdf
import pyproj
import pycrs
from tqdm import tqdm


def loadShp(path):
    d = shape_files[-2] + shape_files[-1]
    m = shape_files[-5] + shape_files[-4]
    y = shape_files[-10] + shape_files[-9] + shape_files[-8] + shape_files[-7]
    path+='\planet_okh_'+ y + m + d + '_pl_a'
    # print("Loading: ",path)
    sf = shapefile.Reader(path+'.shp')
    #reading projection file
    crs = pycrs.load.from_file(path+'.prj')
    # shp_proj = pyproj.Proj(crs.to_proj4()) #.shp file projection
    # std_proj = pyproj.Proj(init = 'epsg:4326') # Lat/Lon system
    # print('Load!')
    shapes = sf.shapes()
    records = sf.records()
    # length = len(records)
    return sf, crs

def getShapeIndex(sf,crs, keyword = '09'):
    ice_shapes_ind = []
    shapes = sf.shapes()
    records = sf.records()
    length = len(records)
    global shp_proj,std_proj
    shp_proj = pyproj.Proj(crs.to_proj4()) #.shp file projection
    std_proj = pyproj.Proj(init = 'epsg:4326') # Lat/Lon system
    # print('Coordinates re-calculation:')
    for i in range(0,length):
        if records[i]['POLY_TYPE'] == 'I':
            # if records[i]['COLORSA'] == keyword:
                # x,y = shapes[i].points[0]
                # lo,la = pyproj.transform(shp_proj,std_proj,x,y)
            ice_shapes_ind.append(i)
    return ice_shapes_ind


# FolderPath = fd.askdirectory(title = 'Choose a month folder to look for data')
# SaveFilePath = fd.askdirectory(title = 'Where to save to?')
FolderPath = r'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\Hydro\NS\m03y2017'
SaveFilePath = r'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\IceMaps'

folders = []
for i,j,k in os.walk(FolderPath):
    folders.append(i)
folders.pop(0)
# print(folders)
MonthYear = FolderPath[len(FolderPath) - 8:len(FolderPath)]
print(MonthYear)
filename = SaveFilePath + '\Ice Data.hdf5'
f = hdf.File(filename,'w')
group = f.create_group(MonthYear)
# dset2 = group.create_dataset('d01m01y2016',(100,100),dtype = 'f')


# Shape-file searching
shapes_folder = r'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\Planets_files\2017'
shapes_folders = []
for i,j,k in os.walk(shapes_folder):
    shapes_folders.append(i)
shapes_folders.pop(0)
# print(shapes_folders)

## _dataset_ = [[_LaNS_],[_LoNS_],[_SigNS_],[_thetaNS_]]
_dataset_ = np.array([[],[],[],[]])
_idset_ = np.array([[],[],[]])
FirstTime = True
previous_day = 1
for currentFolder in folders:
    # print('savepath=%s'%SaveFilePath)
    # print('currentFolder = %s'%currentFolder)
    data_id = currentFolder[len(currentFolder) - 16:len(currentFolder)]
    day = int( data_id[1:3] )
    month = 3 ## walking stick

    if day is not previous_day:
        # if not FirstTime:
            # data[...] = _dataset_
        # FirstTime = False
        if _dataset_.size is not 0:
            print('Saving dataset')    
            data = group.create_dataset(prev_data_id[0:11],data = _dataset_,dtype = 'f')    
            data.attrs['Structure'] = 'Lat, Lon, Sig, Theta(degs)'
            _dataset_ = np.array([[],[],[],[]])
            # print(_dataset_.size)
            shape = _idset_.shape
            data_ids = group.create_dataset(prev_data_id[0:11]+'_ID',(3,shape[1]),data = _idset_,dtype = 'S10')
            data_ids.attrs['Structure'] = 'Poly_ID, Poly_Color, Track'
            _idset_ = np.array([[],[],[]])    

    has_shapefile = False
    for shape_files in shapes_folders:
        shape_day = int(shape_files[-2] + shape_files[-1])
        shape_month = int(shape_files[-5] + shape_files[-4])
        if month == shape_month:
            if abs(shape_day-day) < 3:
                has_shapefile = True
                sf,crs = loadShp(shape_files)
                ice_shapes_ind = getShapeIndex(sf,crs,'09')
    
    
    if has_shapefile:
        print(month,day,'Has shapefile')         
        pathNS = currentFolder 
        sigNS = np.loadtxt(pathNS+'\SigKu.txt')
        LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
        LoNS =  np.loadtxt(pathNS+'\LoKu.txt')    
        thetaNS = np.loadtxt(pathNS+'\IncKu.txt')

        LaNS_f = LaNS.flatten()
        LoNS_f = LoNS.flatten()
        sigNS_f = sigNS.flatten()
        thetaNS_f = thetaNS.flatten()
        size = len(LaNS_f)
        x,y = pyproj.transform(std_proj,shp_proj,LoNS_f,LaNS_f)
        # Points = np.array([LoNS_f, LaNS_f]).transpose().tolist()
        Points = np.array([x, y]).transpose().tolist()

        bool_mask = []
        print('Looking for polys')
        for j in tqdm(ice_shapes_ind):
                polygon = sf.shapes()[j].points
                # print(polygon[0])
                m_path = mplPath.Path(polygon)
                bool_mask.append(m_path.contains_points(Points))
        bool_mask = np.array(bool_mask)

        for i in range(0,len(ice_shapes_ind)):
            _LaNS_ = LaNS_f[bool_mask[i] > 0]
            _LoNS_ = LoNS_f[bool_mask[i] > 0]
            _SigNS_ = LaNS_f[bool_mask[i] > 0]
            _thetaNS_ = thetaNS_f[bool_mask[i] > 0]
            _dataset_ = np.concatenate((_dataset_, [_LaNS_,_LoNS_,_SigNS_,_thetaNS_] ),axis = 1)
            Poly_ID = np.string_(sf.records()[ ice_shapes_ind[i] ]['ID']) ## hdf5 only accepts this
            Poly_Color = np.string_(sf.records()[ ice_shapes_ind[i] ]['COLORSA'])
            Track = np.string_(data_id)
            Poly_ID = np.array([Poly_ID]*len(_LaNS_))
            Poly_Color = np.array([Poly_Color]*len(_LaNS_))
            Track = np.array([Track]*len(_LaNS_))
            _idset_ = np.concatenate((_idset_,[Poly_ID,Poly_Color,Track]),axis = 1 )
            # print(_idset_.shape)
            # if _idset_.size>0:
                # data_ids = group.create_dataset(prev_data_id[0:11]+'_ID',data = _idset_,dtype = 'S10')

        # if np.any(bool_mask):
            # print('Match!')
        # bool_mask = np.sum(bool_mask,0)
        # print(bool_mask)
        
        
        
    else: 
        print(month,day,'Has no shapefile')   
    previous_day = day
    prev_data_id = data_id




    # if day is not previous_day:
    #     data = group.create_dataset(data_id[0:11],dtype = 'f')    
    #     data_ids = group.create_dataset(data_id[0:11]+'_ID',dtype = 'c')    
    # else:
        # if day is not previous_day:

           





