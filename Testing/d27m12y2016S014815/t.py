import matplotlib.pyplot as plt 
import numpy as np 
import statsmodels.api as sm 
import sklearn.linear_model as lm 
import math 


sigNS = np.loadtxt('SigKu.txt') 

thetaNS = np.loadtxt('IncKu.txt') 
theta1NS = thetaNS 
thetaNS = np.power(np.tan(theta1NS/180*math.pi),2) 
sigNS = np.power(10,sigNS*0.1) 


nc = 162 

ns = 0 
nf = 49 
npt = nf-ns 

x1 = [[0 for i in range(2)] for j in range(npt)] 
z = [0 for j in range(npt)] 


th = (np.array(thetaNS[ns:nf,nc])) 
sig = (np.array(sigNS[ns:nf,nc])) 
print(np.shape(th)) 
for i in range(npt): 
    x1[i][0] = th[i] 
    x1[i][1] = sig[i] 
    z[i] = sig[i] * th[i] 


x = np.array(x1) 
y = (np.array(z)) 

# добавим фиктивную переменную для расчета intercept'а 
x_ = sm.add_constant(x) 

# создаем модель для метода обычных наименьших квадратов (Ordinary Least Squares) 
smm = sm.OLS(y, x_) 
# запускаем расчет модели 
res = smm.fit() 
# теперь выведем параметры рассчитанной модели 
print(res.params) 

plt.plot(theta1NS[ns:nf,nc],10*np.log10(sigNS[ns:nf,nc]),'.') 
c = res.params[1] 
b = -res.params[2] 
a = res.params[0] - b*c 
plt.plot(theta1NS[ns:nf,nc], 10*np.log10(a/(thetaNS[ns:nf,nc]+b)+c))
plt.show()