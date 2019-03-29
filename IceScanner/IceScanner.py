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

class IceScanner(tk.Tk):
    #initializaton function
    def __init__(self):
        tk.Tk.__init__(self)
        # tk.Tk.iconbitmap(self,default ='path to icon file.ico ')
        tk.Tk.wm_title(self,'Ice Scanner')
        mainwindow = tk.Frame(self, width = 1000, height = 700)
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
        self.LoadButton = ttk.Button(self, text = 'Load', command = self.loader )
        self.DetectButton = ttk.Button(self, text = 'Detect', command = self.ice_detection )
        self.TrackButton = ttk.Button(self, text = 'Show track', command = self.controller.graphs.show_tracks )
        self.MassButton = ttk.Button(self, text = 'Mass detect', command = self.mass_detection )
        self.LoadLabel = ttk.Label(self,text = 'Load files')
        self.StatusLabel = ttk.Label(self,text = 'Awaiting commands')
        self.LoadLabel.grid(row = 0, column= 1)
        self.StatusLabel.grid(row = 0, column= 2)
        self.LoadButton.grid(row = 0, column = 0, padx=10, pady=10)
        self.MassButton.grid(row = 2, column = 3, padx=10, pady=10)
        self.DetectButton.grid(row = 2, column = 0, padx=10, pady=10)
        self.TrackButton.grid(row = 2, column = 1, padx=10, pady=10)
    

    def loader(self):
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

    
    def ice_detection(self):
        if not(self.controller.loaded):
            tk.messagebox.showerror('Error', 'No data was loaded')
        else:
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

            self.StatusLabel['text'] = 'Flags created' 

            parameters = np.zeros(size,dtype=bool)          
            for i in range(0,size[0]):
            # for i in range(25,26):
                #using function .find_peaks
                detector = findEdgesH(sigNS[i,:])    
                                
                peakind,_ = signal.find_peaks( detector )
                maxs = detector[peakind]* 100 / np.amax(detector[peakind])
                # maxs = detector[peakind]
                for j in range(0,len(maxs)):
                    if maxs[j] > 20:
                        parameters[i][peakind[j]] = True
            
            self.StatusLabel['text'] = 'Peaks found' 

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
            
            newwindow = tk.Toplevel() #make new window
            figure = Figure(figsize=(4,3),dpi = 150)
            axis1 = figure.add_subplot(411)
            axis2 = figure.add_subplot(412)
            axis3 = figure.add_subplot(413)
            axis4 = figure.add_subplot(414)
            axis1.imshow(nMap)
            axis1.set_title('icemap_final')
            axis2.imshow(colFlag)
            axis2.set_title('flags')
            axis3.imshow(parameters)
            axis3.set_title('borders')
            axis4.imshow(sigNS,extent=[0, 500, 0, 49],aspect = 'auto',cmap = 'jet')
            axis4.set_title('map')
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
        print(folders)
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

                file = open(savepath + fname + '.txt','w+')
                file.write("La\tLo\tSig\tTheta\tIce")
                LaNS_f = LaNS.flatten()
                LoNS_f = LoNS.flatten()
                sigNS_f = sigNS.flatten()
                thetaNS_f = thetaNS.flatten()
                nMap_f = nMap.flatten()

                for i in range(0,len(LaNS_f)):
                    if nMap_f[i] > 0:
                        file.write('\n%f\t'%LaNS_f[i]+'%f\t'%LoNS_f[i]+'%f\t'%sigNS_f[i]+'%f\t'%thetaNS_f[i]+'%d'%nMap_f[i])                 

                file.close()
                print('Done another!')
            except:
                print('Something bad happend, but ill let you through')
            
        print('All done!')    


                      
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
            figure = Figure(figsize=(4,3),dpi = 150)
            axis = figure.add_subplot(111)
            axis.imshow(sigNS,extent=[0, 500, 0, 49],aspect = 'auto',cmap = 'jet')
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
            self.figure = Figure(figsize=(7,6),dpi = 150)
            self.axis = self.figure.add_subplot(111)
            
            self.main_map = Basemap(llcrnrlon=132, llcrnrlat=44,urcrnrlon=168,urcrnrlat=64, projection='merc', ax = self.axis, resolution='c')
            self.main_map.drawcoastlines(linewidth = 0.2,color = 'grey')
            self.main_map.drawmapboundary(linewidth=0.1)
            # self.main_map.fillcontinents()
            x,y = self.main_map(LoNS,LaNS)
            self.scat = self.main_map.scatter(x,y,2,sigNS, marker = '.',alpha =0.7,cmap = 'jet')
            self.figure.colorbar(self.scat)
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
            self.canvas.draw()



gui = IceScanner()
gui.mainloop()