import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import itertools
import sklearn.linear_model
data = np.loadtxt("data_to_fit.txt", skiprows=1)
x=data[:,0]
def model_A(x,params):
    y=params[0]+x*params[1]+params[2]*x**2
    return y
def model_B(x,params):
    y=params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    return y
def model_C(x,params):
    y=params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    y+=params[0]*(np.exp(-0.5*(x-params[3])**2/params[4]**2))
    return y
def prob(y_m):
    P=np.sum(-1/(2*data[:,2]**2) *(y_m-data[:,1])**2)
    return P
def metro(PAR,model,gauss,sig):
    PARN=PAR+np.random.normal(loc=0.0,scale=sig,size=PAR.shape[0])
    if gauss:
        while np.sum(PARN<0)>0:
            PARN=PAR+np.random.normal(loc=0.0,scale=sig,size=PAR.shape[0])  
    A=np.min([1,np.exp(prob(model(x,PARN))-prob(model(x,PAR)))])
    #print(prob(PARN) , prob(PAR))
    r=np.random.random()
    if r<A:
       # print(PARN)
        PAR=PARN
        Prob=prob(model(x,PARN))
    else:
        Prob=prob(model(x,PAR))
    return PAR,Prob
N=100000
PARA=np.array([-8,0,0]) #np.random.random(size=3)
PARB=np.random.random(size=3)
PARC=np.random.random(size=5)
ProbA=np.zeros(N)
ProbB=np.zeros(N)
ProbC=np.zeros(N)
PARAL=np.zeros((N,3))
PARBL=np.zeros((N,3))
PARCL=np.zeros((N,5))
for i in range(N):
    [PARAL[i,:],ProbA[i]]=metro(PARA,model_A,False,0.05) 
    [PARBL[i,:],ProbB[i]]=metro(PARB,model_B,True,0.02) 
    [PARCL[i,:],ProbC[i]]=metro(PARC,model_C,True,0.02) 
    PARA=PARAL[i,:]
    PARB=PARBL[i,:]
    PARC=PARCL[i,:]
MPARAL=PARAL[int(N/2):,:]
MPARBL=PARBL[int(N/2):,:]
MPARCL=PARCL[int(N/2):,:]
def BIC(model,MPARL):
    BI2=-prob(model(x,np.mean(MPARL,axis=0)))+MPARL.shape[1]*np.log(x.shape[0])/2
    return BI2
BICA=BIC(model_A,MPARAL)
BICB=BIC(model_B,MPARBL)
BICC=BIC(model_C,MPARCL)
xl=np.linspace(3,7,100)
def Plot(model,MPARL,BIC,Name):
    plt.figure(figsize=(12,12))
    for i in range(MPARL.shape[1]):
        plt.subplot(2,int(MPARL.shape[1]+1)/2,i+1)
        plt.hist(MPARL[:,i])
        plt.title(r'$\beta_%1.0f= $'%i +'%4.3f'%np.mean(MPARL[:,i])+'$\pm $ %4.3f' %np.std(MPARL[:,i]))
        plt.xlabel(r'$\beta_%1.0f$'%i)
    plt.subplot(2,int(MPARL.shape[1]+1)/2,i+2)
    plt.errorbar(x,data[:,1],yerr=data[:,2],xerr=0,fmt='.')
    plt.plot(xl,model(xl,np.mean(MPARL,axis=0)))
    plt.title(r'BIC= %4.3f'%(BIC*2))
    plt.savefig(Name)
    plt.show()
Plot(model_A,MPARAL,BICA,'model_A.png')
Plot(model_B,MPARBL,BICB,'model_B.png')
Plot(model_C,MPARCL,BICC,'model_C.png')