import math
from tkinter import ttk
import tkinter.messagebox
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# from matplotlib import style
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
        mainwindow = tk.Frame(self, width = 1000, height = 500)
        self.geometry('1000x700')
        mainwindow.pack(side = 'bottom',fill = 'both',expand = True)
        self.graphs = GraphSet(mainwindow,self)
        self.tools = ToolSet(mainwindow,self)

        self.graphs.grid(row=1,column=0,sticky = 'nwe')
        self.tools.grid(row = 0,column =0,sticky = 'nwe')
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
        self.LoadLabel = ttk.Label(self,text = 'Load files')
        self.StatusLabel = ttk.Label(self,text = 'Awaiting commands')
        self.LoadLabel.grid(row = 0, column= 1)
        self.StatusLabel.grid(row = 0, column= 2)
        self.LoadButton.grid(row = 0, column = 0, padx=10, pady=10)
        self.DetectButton.grid(row = 2, column = 0, padx=10, pady=10)
        self.TrackButton.grid(row = 2, column = 1, padx=10, pady=10)
    
    def loader(self):
        self.pathNS = fd.askdirectory() 
        self.LoadLabel['text'] = 'Loaded: ' + self.pathNS[len(self.pathNS) - 15:-1] 
        self.sigNS = np.loadtxt(self.pathNS+'\SigKu.txt')
        self.LaNS = np.loadtxt(self.pathNS+'\LaKu.txt')     
        self.LoNS =  np.loadtxt(self.pathNS+'\LoKu.txt')    
        self.thetaNS = np.loadtxt(self.pathNS+'\IncKu.txt')
        for i in range(0,self.thetaNS.shape[1]):
            for j in range(0,math.floor(self.thetaNS.shape[0]/2)):
                self.thetaNS[j][i] = -self.thetaNS[j][i]    
        self.controller.loaded = True
        self.StatusLabel['text'] = 'Successfully loaded'
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
            colFlag = np.empty((size))
            self.StatusLabel['text'] = 'Detection started'
            for i in range(1,size[1]):
                try:
                    new_parameters, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,6,-17])
                    diff = np.subtract( sigNS[:,i], HypApp( thetaNS[:,i],new_parameters[0],new_parameters[1],new_parameters[2] ))
                    err = np.mean( diff**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
                    #print(err)
                    if ( new_parameters[0] < 2000 ) & ( new_parameters[0] > 15 ) & ( new_parameters[2] < 100 ) & ( err < 30 ):
                        colFlag[:,i] = True
                    else: 
                        colFlag[:,i] = False

                except RuntimeError:
                    print('Not fitted')

            self.StatusLabel['text'] = 'Flags created'  
            parameters = np.zeros(size)          
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

            # print(colFlag)
            nMap = colFlag

            #interpolating near edges
            for i in range(0,size[1]-1):
            # for i in range(23,24):
                if colFlag[0][i] != colFlag[0][i+1]:
                    d = np.zeros(size)
                    nearEdge = np.zeros((size[0],1))
                    indeces = np.arange(0,size[0])
                    for j in range(0,size[0]):
                        for k in range(0,size[1]):
                            if parameters[j][k] != 0:
                                d[j][k] = k-i
                        temp = np.abs(d[j,:])
                        temp[temp==0] = np.nan
                        temp_min = np.nanargmin(temp)
                        nearEdge[j] = d[j][temp_min]

                    
                    std_dev = np.std(nearEdge)
                    nearEdgeNew = np.array([])
                    indecesNew = np.array([])
                    for j in range(0,size[0]):
                        if np.abs(nearEdge[j] - std_dev) < 2*std_dev:
                            nearEdgeNew = np.append( nearEdgeNew,nearEdge[j] )
                            indecesNew = np.append( indecesNew,indeces[j] )
                    # nearEdgeNew = nearEdge
                    # indecesNew = indeces
                    # for j in range(0,size[0]):
                    #     if np.abs(nearEdge[j] - std_dev) > 2*std_dev:
                    #         nearEdgeNew[j] = np.nan
                    medianed = sp.signal.medfilt(nearEdgeNew,kernel_size = 3)
                    interpolated = sp.interpolate.CubicSpline( indecesNew, nearEdgeNew,extrapolate=False )
                                      

                    if (i<180) & (i>140):
                        newwindow = tk.Toplevel() #make new window
                        figure = Figure(figsize=(4,3),dpi = 150)
                        axis = figure.add_subplot(111)
                        axis.plot(nearEdge,'ro')
                        axis.plot(indecesNew,nearEdgeNew,'g.')
                        axis.plot(indecesNew,medianed,'k--')
                        axis.plot(indeces,interpolated(indeces),'b--')

                        canvas = FigureCanvasTkAgg(figure,newwindow)
                        canvas.draw()
                        canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
                        canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
                               
                

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
            

            # nMap = np.repeat(nMap,size[0],axis = 1)





                    #uncomment for debuging
                # print(new_parameters)
                # newwindow = tk.Toplevel()
                # figure = Figure(figsize=(4,3),dpi = 150)
                # axis = figure.add_subplot(111)
                # axis.plot(thetaNS[:,i],sigNS[:,i],'.')
                # axis.plot(thetaNS[:,i], HypApp( thetaNS[:,i],new_parameters[0],new_parameters[1],new_parameters[2] ))
                # axis.set_title(str(i))
                # canvas = FigureCanvasTkAgg(figure,newwindow)
                # canvas.draw()
                # canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
                # canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
            newwindow = tk.Toplevel() #make new window
            figure = Figure(figsize=(4,3),dpi = 150)
            axis = figure.add_subplot(111)
            axis.imshow(nMap)
            canvas = FigureCanvasTkAgg(figure,newwindow)
            canvas.draw()
            canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
            canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
            
             


class GraphSet(tk.Frame):
    
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        self.first_time = True
        #Somelabel = tk.Label(self, text = 'Graphs' )
        #Somelabel.pack(padx=10,pady=10)
    
    def show_tracks(self):
        if not(self.controller.loaded):
            tk.messagebox.showerror('Error', 'No data was loaded')
        else:
            trackwindow = tk.Toplevel()
            thetaNS = self.controller.tools.thetaNS
            sigNS = self.controller.tools.sigNS
            figure = Figure(figsize=(4,3),dpi = 150)
            axis = figure.add_subplot(111)
            axis.imshow(sigNS,extent=[0, 500, 0, 49],aspect = 'auto',cmap = 'ocean')
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
            self.figure = Figure(figsize=(2,3),dpi = 150)
            self.axis = self.figure.add_subplot(111)
            
            self.main_map = Basemap(llcrnrlon=132, llcrnrlat=44,urcrnrlon=168,urcrnrlat=64, projection='merc', ax = self.axis, resolution='c')
            self.main_map.drawcoastlines(linewidth = 0.2,color = 'grey')
            self.main_map.drawmapboundary(linewidth=0.1)
            self.main_map.fillcontinents()
            x,y = self.main_map(LoNS,LaNS)
            self.scat = self.main_map.scatter(x,y,1,sigNS, marker = '.',alpha =0.7,cmap = 'ocean')
            self.figure.colorbar(self.scat)
            self.canvas = FigureCanvasTkAgg(self.figure,self)
            self.canvas.draw()
            canvas_toolbar = NavigationToolbar2Tk(self.canvas,self)
            canvas_toolbar.update()
            self.canvas._tkcanvas.pack(side = tk.BOTTOM,fill = tk.BOTH, expand = True)
            self.canvas.get_tk_widget().pack(side = tk.BOTTOM,fill = tk.BOTH, expand = True)
            self.first_time = False
        else: 
            self.scat.remove()
            x,y = self.main_map(LoNS,LaNS)
            self.scat = self.main_map.scatter(x,y,1,sigNS, marker = '.',alpha =0.7,cmap = 'ocean')
            self.canvas.draw()



gui = IceScanner()
gui.mainloop()