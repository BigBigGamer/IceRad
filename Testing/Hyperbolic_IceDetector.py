import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# pathNS = 'E:\Work\GitHub\IceRad\Testing\d01m01y2017S003600'
pathNS = 'E:\Work\GitHub\IceRad\Testing\d27m12y2016S014815'
# pathNS = 'E:\Work\GitHub\IceRad\Testing\d03m01g17S172400'
# pathNS = 'E:\Work\GitHub\IceRad\Testing\d31m03y2017S213254'


# File reading
sigNS = np.loadtxt(pathNS+'\SigKu.txt')
LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
LoNS =  np.loadtxt(pathNS+'\LoKu.txt')    
thetaNS = np.loadtxt(pathNS+'\IncKu.txt')
size =  thetaNS.shape

plt.figure(1)
plt.subplot(3,1,1)
plt.imshow(sigNS,extent=[0,size[1], 0,size[0]],aspect = 'auto',cmap = 'jet')
plt.title('Base Data')

plt.figure(2)
plt.title('HParameters')

# Ice Detecting
def HypApp(xdata,p1,p2,p3):
    return p1 * abs( 1/(abs(xdata) + p2) ) + p3

colFlag_h = np.zeros((size),dtype = bool)
errh = -1*np.ones(size[1],dtype = np.float32)
for i in range(0,size[1]):
    try:
        # new_psh, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,5,0],bounds =([150,2,-np.inf],[500,7,np.inf]) )
        new_psh, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,5,0])
        diff_h = np.subtract( sigNS[:,i], HypApp( thetaNS[:,i],new_psh[0],new_psh[1],new_psh[2] ))
            
        errh[i] = np.mean( diff_h**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
        if  ( abs(new_psh[0]) < 2000 ) & (errh[i]<30) :        
            
            plt.figure(2)
            plt.plot(i,new_psh[0],'ro')
            plt.plot(i,new_psh[1],'go')
            plt.plot(i,new_psh[2],'b.')
            colFlag_h[:,i] = True
        else: 
            colFlag_h[:,i] = False
    except RuntimeError:
        print('Not fitted_1')

plt.figure(1)
plt.subplot(3,1,2)
plt.imshow(colFlag_h,extent=[0, 500, 0, 49],aspect = 'auto')
# plt.title('HypApprox')
# plt.axhline(40)
plt.subplot(3,1,3)
plt.axhline(30)

plt.grid(which = 'major')
plt.plot(errh,'r-')
plt.title('Approximation Errors, %')
plt.ylim(0,100)
plt.xlim(0,size[1])

# x = np.arange(-20,20,0.03)

# for i in range(1,10):
#     plt.plot(x,SqApp(x,i,1,0))   



plt.show()
print('Done!')