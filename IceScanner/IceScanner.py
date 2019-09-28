import math
import os
from tkinter import ttk
import tkinter.messagebox
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') #change backend for mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # figure and toolbar
from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
import scipy as sp
from scipy.optimize import curve_fit
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

# def findEdgesH(sigNS):
#     size = len(sigNS)
#     stepsize = 50
#     x = np.arange(-stepsize,stepsize)
#     sigma = 6
#     step = -np.heaviside(x-stepsize,1) + np.heaviside(x+stepsize,1)
#     Gauss = lambda x,s: -x * np.exp( -x**2 / ( 2*s**2 ) )
#     convd = np.convolve(sigNS, Gauss(x,sigma),'same')
#     # snr = np.abs(convd) / ( np.sqrt( sp.integrate.quad( Gauss**2,  ) ) )
#     # print(np.convolve(step,sigNS**2,'same'))
#     snr = np.abs(convd) / ( np.sqrt( np.convolve(step,sigNS**2,'same') ) ) 
#     # snr2 = np.abs(convd) / ( np.sqrt( np.trapz( sigNS**2 ) ) ) 
#     return snr

class IceScanner(tk.Tk):
    #initializaton function
    def __init__(self):
        tk.Tk.__init__(self)
        # tk.Tk.iconbitmap(self,default ='path to icon file.ico ')
        tk.Tk.wm_title(self,'Ice Scanner')
        mainwindow = tk.Frame(self, width = 1000, height = 500)
        # self.geometry('1000x700')
        mainwindow.pack(side = 'bottom',fill = 'both',expand = True)
        self.graphs = GraphSet(mainwindow,self)
        self.tools = ToolSet(mainwindow,self)

        self.tools.pack(side = tk.TOP, fill = tk.X)
        self.graphs.pack(side = tk.TOP, fill = tk.BOTH)

        self.loaded = False
        mainwindow.grid_rowconfigure(0,weight=1)
        mainwindow.grid_columnconfigure(0,weight=1)


class ToolSet(tk.Frame):
    
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        self.loaded = False
        # elements
        self.LoadButton = ttk.Button(self, text = 'Load', command = self.loader )
        self.DetectButton = ttk.Button(self, text = 'Detect', command = self.ice_detection )
        self.TrackButton = ttk.Button(self, text = 'Show track', command = self.controller.graphs.show_tracks )
        self.MassButton = ttk.Button(self, text = 'Mass detect', command = self.mass_detection )
        self.VerifyButton = ttk.Button(self, text = 'Verify results', command = self.verification )
        self.LoadLabel = ttk.Label(self,text = 'Load files')
        self.StatusLabel = ttk.Label(self,text = 'Awaiting commands')
        self.method = tk.StringVar(self)
        self.methods = ['By Approximation','By Kurtosis']
        # self.methods.set(self.choices[0])
        self.DetectorSelect = ttk.OptionMenu(self,self.method,self.methods[0],*self.methods)
        # packing
        self.LoadLabel.grid(row = 0, column= 1)
        self.StatusLabel.grid(row = 0, column= 2)
        self.LoadButton.grid(row = 0, column = 0, padx=10, pady=10)
        self.MassButton.grid(row = 2, column = 2, padx=10, pady=10)
        self.VerifyButton.grid(row = 2, column = 3, padx=10, pady=10)
        self.DetectButton.grid(row = 2, column = 0, padx=10, pady=10)
        self.TrackButton.grid(row = 2, column = 1, padx=10, pady=10)
        self.DetectorSelect.grid(row = 3, column = 0, padx=0, pady=10)
    

    def loader(self):
        try:
            self.pathNS = fd.askdirectory() 
            self.LoadLabel['text'] = 'Loaded: ' + self.pathNS[len(self.pathNS) - 18:-1] 
            self.sigNS = np.loadtxt(self.pathNS+'\SigKu.txt')
            self.LaNS = np.loadtxt(self.pathNS+'\LaKu.txt')     
            self.LoNS =  np.loadtxt(self.pathNS+'\LoKu.txt')    
            self.thetaNS = np.loadtxt(self.pathNS+'\IncKu.txt')
            for i in range(0,self.thetaNS.shape[1]):
                for j in range(0,math.floor(self.thetaNS.shape[0]/2)):
                    self.thetaNS[j][i] = -self.thetaNS[j][i]    
            self.controller.loaded = True
            self.StatusLabel['text'] = 'Successfully'
            self.controller.graphs.update_map()
        except:
            tk.messagebox.showerror('Error', 'No data selected, or something else.')

    
    def ice_detection(self):


        def kurtosisFilter(self):
            LaNS = self.controller.tools.LaNS
            LoNS = self.controller.tools.LoNS
            thetaNS = self.controller.tools.thetaNS # deg
            sigNS = self.controller.tools.sigNS
            size =  thetaNS.shape

            for i in range(0,size[1]):
                for j in range(0,size[0]):
                    if j < math.floor(size[0]/2):
                        thetaNS[j][i] *= -1  
            sigNSun = sigNS # dB
            sigNSun_up,sigNSun_down = self.getHalfs(sigNSun)

            thetaNSun = thetaNS 
            thetaNSun_up,thetaNSun_down = self.getHalfs(thetaNSun)

            # normalization 
            sigNS = np.power(10, sigNS*0.1)
            sigNS_up,sigNS_down = self.getHalfs(sigNS)

            thetaNS = np.tan( thetaNS/180 * np.pi)
            thetaNS_up,thetaNS_down =  self.getHalfs(thetaNS)

            mu_up = [[],[],[],[]]
            mu_down = [[],[],[],[]]

            size_half =  thetaNS_up.shape
            colFlag_h_up = np.zeros((size_half),dtype = bool)
            colFlag_h_down = np.zeros((size_half),dtype = bool)

            for i in range(0,size_half[1]):
                mu_up_2,mu_up_3,mu_up_4,mu_downs = 0,0,0,0
                cut_n = i
                y = sigNS_up[:,cut_n] # sig_0
                x = thetaNS_up[:,cut_n] # tan
                mean_up = np.sum([ x[j] * y[j] * np.cos(thetaNSun_up[j][i])**4  for j in range(0,size_half[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun_up[j][i])**4  for j in range(0,size_half[0]) ]) 

                for j in range(0,size_half[0]):
                    mu_up_2 += (x[j]- mean_up)**2 * y[j] * np.cos(thetaNSun_up[j][i])**4
                    # mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
                    mu_up_4 += (x[j]- mean_up)**4 * y[j] * np.cos(thetaNSun_up[j][i])**4
                    mu_downs += y[j] * np.cos(thetaNSun_up[j][i])**4

                dispersion_up = np.sqrt(mu_up_2/mu_downs)
                # skewness = (mu_up_3/mu_down) / (dispersion**3)
                kurtosis_up = (mu_up_4/mu_downs) / (dispersion_up**4) - 3 

                mu_up[0].append(mean_up)
                mu_up[1].append(dispersion_up)
                # mu_up[2].append(skewness)
                mu_up[3].append(kurtosis_up)



                mu_up_2,mu_up_3,mu_up_4,mu_downs = 0,0,0,0
                y = sigNS_down[:,cut_n] # sig_0
                x = thetaNS_down[:,cut_n] # tan

                mean_down = np.sum([ x[j] * y[j] * np.cos(thetaNSun_down[j][i])**4  for j in range(0,size_half[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun_down[j][i])**4  for j in range(0,size_half[0]) ]) 

                for j in range(0,size_half[0]):
                    mu_up_2 += (x[j]- mean_down)**2 * y[j] * np.cos(thetaNSun_down[j][i])**4
                    # mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
                    mu_up_4 += (x[j]- mean_down)**4 * y[j] * np.cos(thetaNSun_down[j][i])**4
                    mu_downs += y[j] * np.cos(thetaNSun_down[j][i])**4

                dispersion_down = np.sqrt(mu_up_2/mu_downs)
                # skewness = (mu_up_3/mu_downs) / (dispersion**3)
                kurtosis_down = (mu_up_4/mu_downs) / (dispersion_down**4) - 3 

                mu_down[0].append(mean_down)
                mu_down[1].append(dispersion_down)
                # mu_down[2].append(skewness)
                mu_down[3].append(kurtosis_down)

            # Merge up-half and down-half
            nMap  = np.empty(size)
            nMap[0:25,:] = mu_up[3]
            nMap[24:,:] = mu_down[3]
            nMap[nMap<=4] = 0
            nMap[nMap>4] = 1
            return nMap

        def hyperbollicFilter(self):
            LaNS = self.controller.tools.LaNS
            LoNS = self.controller.tools.LoNS
            thetaNS = self.controller.tools.thetaNS
            sigNS = self.controller.tools.sigNS
            size =  thetaNS.shape
            colFlag = np.zeros((size),dtype = bool)
            self.StatusLabel['text'] = 'Detection started'
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
            
            nMap = np.zeros_like(colFlag)   # This is to copy the array, not link it
            nMap[:] = colFlag[:]            #
            return nMap

        def findEdges(sigNS):

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
            size =  sigNS.shape
            sigNSun = sigNS
            parameters = np.zeros(size)  
            detectorBig = np.zeros(size)
            detectorSmall = np.zeros(size)
            maxs = np.zeros(size)    
            # cvs = np.zeros(size,dtype = bool) 
            sigNS_inv = 10**(0.1*sigNSun) 

            sigNS = sp.ndimage.filters.gaussian_filter(sigNSun, 0.5, mode='constant')

            for i in range(0,size[0]):
                #using function .find_peaks
                detectorBig[i,:] = findEdgesH(sigNS[i,:],6)
                detectorSmall[i,:] = findEdgesH(sigNS[i,:],1.1)
                detectorBig[i,:] = detectorBig[i,:]* 100 / np.amax(detectorBig[i,:])
                detectorSmall[i,:] = detectorSmall[i,:]* 100 / np.amax(detectorSmall[i,:])
            # cvs = detectorSmall
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


            ### OLD ####
            # size = sigNS.shape
            # borders = np.zeros(size,dtype=bool)          
            # for i in range(0,size[0]):
            # # for i in range(25,26):
            #     #using function .find_peaks
            #     detector = findEdgesH(sigNS[i,:])    
                                
            #     peakind,_ = signal.find_peaks( detector )
            #     maxs = detector[peakind]* 100 / np.amax(detector[peakind])
            #     # maxs = detector[peakind]
            #     for j in range(0,len(maxs)):
            #         if maxs[j] > 20:
            #             borders[i][peakind[j]] = True
            return parameters
        
        def fillFlags(borders,ice):
            size = borders.shape
            for i in range(0,size[0]):
                BorderIndex = np.nonzero(borders[i,:])
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
            return nMap

        if not(self.controller.loaded):
            tk.messagebox.showerror('Error', 'No data was loaded')
        else:
            print(self.method.get())
            if self.method.get() == 'By Approximation':
                nMap = hyperbollicFilter(self)
            elif self.method.get() == 'By Kurtosis':
                nMap = kurtosisFilter(self)
                
            
            # Finding border
            borders = findEdges(self.controller.tools.sigNS)          

            # flag filling
            # nMap = fillFlags(borders, nMap)
            
            # Output

            # Make new window
            newwindow = tk.Toplevel() 
            figure = Figure(figsize=(4,3),dpi = 150)
            axis1 = figure.add_subplot(211)
            axis2 = figure.add_subplot(212)
            axis1.imshow(nMap)
            axis1.set_title('icemap_final')
            axis2.imshow(borders)
            axis2.set_title('borders')
            canvas = FigureCanvasTkAgg(figure,newwindow)
            canvas.draw()
            canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
            canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)


    def mass_detection(self):
        FolderPath = fd.askdirectory(title = 'Choose a month folder to scan')
        folders = []
        for i,j,k in os.walk(FolderPath):
            folders.append(i)
        folders.pop(0)
        # print(folders)
        savepath = fd.askdirectory(title = 'Choose where to save the maps')
        MonthYear = FolderPath[len(FolderPath) - 8:len(FolderPath)]
        try:
            os.mkdir(savepath+'/'+ MonthYear)
        except OSError:
            print('Directory already exists')
        savepath = savepath + '/' + MonthYear + '/'



        for currentFolder in folders:
            try:
                print('savepath=%s'%savepath)
                print('currentFolder = %s'%currentFolder)
                pathNS = currentFolder 
                sigNS = np.loadtxt(pathNS+'\SigKu.txt')
                LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
                LoNS =  np.loadtxt(pathNS+'\LoKu.txt')    
                thetaNS = np.loadtxt(pathNS+'\IncKu.txt')
                
                for i in range(0,thetaNS.shape[1]):
                    for j in range(0,math.floor(thetaNS.shape[0]/2)):
                        thetaNS[j][i] = -thetaNS[j][i] 
                size =  thetaNS.shape
                colFlag = np.zeros((size),dtype = bool)
                for i in range(0,size[1]):
                    try:
                        new_parameters, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,5,0],bounds =([150,2,-np.inf],[500,7,np.inf]) )
                        diff = np.subtract( sigNS[:,i], HypApp( thetaNS[:,i],new_parameters[0],new_parameters[1],new_parameters[2] ))
                        err = np.mean( diff**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
                        #print(err)
                        if ( new_parameters[0] < 2000 ) & ( new_parameters[0] > 15 ) & ( new_parameters[2] < 100 ) & ( err < 30 ):
                            colFlag[:,i] = True
                        else: 
                            colFlag[:,i] = False

                    except RuntimeError:
                        pass

                parameters = np.zeros(size,dtype=bool)          
                for i in range(0,size[0]):
                    detector = findEdgesH(sigNS[i,:])    
                    peakind,_ = signal.find_peaks( detector )
                    maxs = detector[peakind]* 100 / np.amax(detector[peakind])
                    for j in range(0,len(maxs)):
                        if maxs[j] > 20:

                            parameters[i][peakind[j]] = True
                nMap = np.zeros_like(colFlag)   # This is to copy the array, not link it
                nMap[:] = colFlag[:]            #

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



                track = pathNS[len(pathNS) - 6:-1]
                # fname = pathNS[len(pathNS) - 18:-1]
                fname = pathNS[len(pathNS) - 15:-1]

                file = open(savepath + fname + '.tsv','w+')
                file.write("La\tLo\tSig\tTheta\tIce")
                LaNS_f = LaNS.flatten()
                LoNS_f = LoNS.flatten()
                sigNS_f = sigNS.flatten()
                thetaNS_f = thetaNS.flatten()
                nMap_f = nMap.flatten()

                for i in range(0,len(LaNS_f)):
                    file.write('\n%f\t'%LaNS_f[i]+'%f\t'%LoNS_f[i]+'%f\t'%sigNS_f[i]+'%f\t'%thetaNS_f[i]+'%d'%nMap_f[i])                 

                file.close()
                print('Done another!')
            except:
                print('Something bad happend, but ill let you through')
            
        print('All done!')    


    def verification(self):
        if not(self.controller.loaded):
            tk.messagebox.showerror('Error', 'No data was loaded')
        else:
            pass


    def getHalfs(self,array):
        # Create two arrays 'up' and 'down' from 'array' which are upper and bottom halfs,
        #  with mirrowed opposite sides. Shapes stay the same.
        import math
        import numpy as np

        size = array.shape
        up = np.zeros(size) - 100
        down = np.zeros(size) -100

        up_limit = math.floor((size[0] + 1)/2)
        down_limit = math.floor((size[0] - 1)/2)

        up[0:up_limit,:] = array[0:up_limit,:]
        up[down_limit:,:] = np.flipud(array[0:up_limit,:])

        down[down_limit:,:] = array[down_limit:,:]
        down[0:up_limit:,:] = np.flipud(array[down_limit:,:])

        return up,down

                      
class GraphSet(tk.Frame):
    
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        self.first_time = True
    

    def show_tracks(self):
        if not(self.controller.loaded):
            tk.messagebox.showerror('Error', 'No data was loaded')
        else:
            trackwindow = tk.Toplevel()
            thetaNS = self.controller.tools.thetaNS
            sigNS = self.controller.tools.sigNS
            size = thetaNS.shape
            figure = Figure(figsize=(8,3), dpi = 150)
            axis = figure.add_subplot(111)
            im = axis.imshow(sigNS,extent=[0, size[1], 0, size[0]], aspect = 'auto',cmap = 'jet')
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax = cax, orientation='vertical')
            cbar.ax.set_title('$\sigma^0,dB$')
            canvas = FigureCanvasTkAgg(figure,trackwindow)
            canvas.draw()
            canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
            canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
    

    def update_map(self):
            
        LaNS = self.controller.tools.LaNS
        LoNS = self.controller.tools.LoNS
        thetaNS = self.controller.tools.thetaNS
        sigNS = self.controller.tools.sigNS
        #Boundries=[64, 168, 44, 132];
        if self.first_time :
            self.figure = Figure()
            self.figure = Figure(figsize=(6,5),dpi = 150)
            self.axis = self.figure.add_subplot(111)
            


            self.main_map = Basemap(llcrnrlon=132, llcrnrlat=44,urcrnrlon=168,urcrnrlat=64, projection='merc', ax = self.axis, resolution='c')
           
           
           
            self.main_map.fillcontinents(zorder = 0)
            self.main_map.drawcoastlines(linewidth = 0.2,color = 'grey', zorder = 3)
            self.main_map.drawmapboundary(linewidth=0.1, zorder=-1)
            x,y = self.main_map(LoNS,LaNS)
            self.scat = self.main_map.scatter(x,y,2,sigNS, marker = '.',alpha =0.7,cmap = 'jet',zorder = 3)
            self.cbar = self.figure.colorbar(self.scat)
            self.cbar.ax.set_title('$\sigma^0,dB$')
            self.canvas = FigureCanvasTkAgg(self.figure,self)
            # self.canvas.get_tk_widget().pack(pady = 10, expand = True)

            self.canvas.draw()
            canvas_toolbar = NavigationToolbar2Tk(self.canvas,self)
            canvas_toolbar.update()
            # self.canvas._tkcanvas.pack(side = tk.BOTTOM,fill = tk.BOTH, expand = True)
            self.canvas.get_tk_widget().pack(side = tk.BOTTOM,fill = tk.BOTH, expand = True)
            # self.canvas._tkcanvas.pack(side = tk.BOTTOM,fill = tk.BOTH, expand = True)
            self.first_time = False
        else: 
            self.scat.remove()
            x,y = self.main_map(LoNS,LaNS)
            self.scat = self.main_map.scatter(x,y,1,sigNS, marker = '.',alpha =0.7,cmap = 'jet')
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cbar = plt.colorbar(im, cax=cax, orientation='vertical')

            # cbar.ax.set_title('$\sigma^0,dB$')

            self.canvas.draw()



gui = IceScanner()
gui.mainloop()