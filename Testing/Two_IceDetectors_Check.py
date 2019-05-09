import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import toolbar

pathNS = 'E:\Work\GitHub\IceRad\Testing\d01m01y2017S003600'
# pathNS = 'E:\Work\GitHub\IceRad\Testing\d27m12y2016S014815'

# File reading
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)
size =  thetaNS.shape

plt.figure(1)
plt.subplot(3,1,1)
plt.imshow(sigNS,extent=[0,size[1], 0,size[0]],aspect = 'auto',cmap = 'jet')
plt.title('Base')

plt.figure(2)
plt.title('HParameters')

plt.figure(3)
plt.title('SParameters')

plt.figure(4)
plt.title('S2Parameters')

# Ice Detecting
def HypApp(xdata,p1,p2,p3):
    return p1 * abs( 1/(abs(xdata) + p2) ) + p3

def SqApp(xdata,p1,p2,p3):
    return  p1*1/((xdata )**2+ p2 ) + p3

def SqApp2(xdata,p1,p2,p3):
    return  p1*1/((xdata + p2 )**2 ) + p3

colFlag_h = np.zeros((size),dtype = bool)
errh = -1*np.ones(size[1],dtype = np.float32)
errs = -1*np.ones(size[1],dtype = np.float32)
errs2 = -1*np.ones(size[1],dtype = np.float32)
for i in range(0,size[1]):
    try:
        new_psh, covariance = curve_fit(HypApp,thetaNS[:,i],sigNS[:,i], [200,6,-17])

        diff_h = np.subtract( sigNS[:,i], HypApp( thetaNS[:,i],new_psh[0],new_psh[1],new_psh[2] ))
        

        errh[i] = np.mean( diff_h**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )

        
        plt.figure(2)
        plt.plot(i,new_psh[0],'ro')
        plt.plot(i,new_psh[1],'go')
        plt.plot(i,new_psh[2],'b.')

        if ( new_psh[0] < 2000 ) & ( new_psh[0] > 15 ) & ( new_psh[2] < 100 ) & ( errh[i] < 30 ):
            colFlag_h[:,i] = True
        else: 
            colFlag_h[:,i] = False
    except RuntimeError:
        print('Not fitted_1')
    
    try:
        new_pss, covariance = curve_fit(SqApp,thetaNS[:,i],sigNS[:,i], [1,1,0])    
        if (new_pss[0]<25000) & (new_pss[0]>200):
             
            
            diff_s = np.subtract( sigNS[:,i], SqApp( thetaNS[:,i],new_pss[0],new_pss[1],new_pss[2] ))
            errs[i] = np.mean( diff_s**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
            # errs[i] = np.mean( diff_s**2 )

            plt.figure(3)
            plt.plot(i,new_pss[0],'ro')
            plt.plot(i,new_pss[1],'go')
            plt.plot(i,new_pss[2],'b.')
            # if ( new_pss[0] < 2000 ) & ( new_pss[0] > 15 ) & ( new_pss[2] < 100 ) & ( errs[i] < 30 ):
            #     colFlag_s[:,i] = True
            # else: 
            #     colFlag_s[:,i] = False
    except RuntimeError:
        print('Not fitted_2')
    
    try:
        new_pss2, covariance = curve_fit(SqApp2,thetaNS[:,i],sigNS[:,i], [1,1,0])    
        diff_s2 = np.subtract( sigNS[:,i], SqApp2( thetaNS[:,i],new_pss2[0],new_pss2[1],new_pss2[2] ))
        errs2[i] = np.mean( diff_s2**2 ) * 100 / ( np.amax(sigNS[:,i]) - np.amin(sigNS[:,i]) )
        # errs[i] = np.mean( diff_s**2 )

        plt.figure(4)
        plt.plot(i,new_pss2[0],'ro')
        plt.plot(i,new_pss2[1],'go')
        plt.plot(i,new_pss2[2],'b.')
        # if ( new_pss[0] < 2000 ) & ( new_pss[0] > 15 ) & ( new_pss[2] < 100 ) & ( errs[i] < 30 ):
        #     colFlag_s[:,i] = True
        # else: 
        #     colFlag_s[:,i] = False
    except RuntimeError:
        print('Not fitted_2')


plt.figure(1)
plt.subplot(3,1,3)
plt.plot(errh,'r-')
plt.plot(errs,'b-')
plt.plot(errs2,'g-')
plt.title('Errors')

plt.show()
print('Done!')