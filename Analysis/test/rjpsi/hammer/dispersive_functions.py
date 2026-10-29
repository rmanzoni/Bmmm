import matplotlib
import lsqfit
import gvar
import numpy as np
from poly import *
import CC_fit_parameters as ccp

ORTHOPOLS = [0,1,2,3,4] #,5]                # Orthonormal polynomials to use in main analysis.
    
def pn(n,alpha,z):  
    ret = 0
    if n==0:ret = p0(alpha,z)
    if n==1:ret = p1(alpha,z)
    if n==2:ret = p2(alpha,z)
    if n==3:ret = p3(alpha,z)
    if n==4:ret = p4(alpha,z)
    if n==5:ret = p5(alpha,z)
    return ret

def zz(s,sG,s0):
    return (np.sqrt(sG-s)-np.sqrt(sG-s0))/(np.sqrt(sG-s)+np.sqrt(sG-s0))
    
def phi(F,s,sG,s0,chi,eta,MB,MM):
    sp=(MB+MM)**2
    sm=(MB-MM)**2
    
    if F=='V':
        Nf=2.0/sp
        p=1
        n=2
        m=3        
    if F=='A0':
        Nf=1.0
        p=2
        n=1
        m=3
    if F=='A1':
        Nf=2.0*sp
        p=1
        n=2
        m=1
    if F=='A12':
        Nf=64*MB**2*MM**2
        p=2
        n=2
        m=1
    if F=='T1':
        Nf=2.0
        p=1
        n=3
        m=3
    if F=='T2':
        Nf=2.0*sp*sm
        p=1
        n=3
        m=1
    if F=='T23':
        Nf=16*MB**2*MM**2/sp
        p=0
        n=3
        m=1

    lambdakin=(MB**2-MM**2-s)**2-4*MM**2*s
    zi=zz(s,sG,s0)
    rootpart=Nf*eta/(32*np.pi**2*chi)*(lambdakin/(zz(s,sG,sm)))**(m/2.0)*(-zz(s,sG,0)/s)**((n+p+1))*(4*(1+zi)*(sG-s0)/(1-zi)**3)
    return np.sqrt(rootpart)
        
def alphaBM(sG,s0,MB,MM):
    sp=(MB+MM)**2
    sm=(MB-MM)**2
    return np.pi-np.arctan(np.sqrt(sp-sG)/np.sqrt(sG-s0))
    
def blaschke_factor_1p(s,sG,Mres_list):
    Pres=1
    for Mres in Mres_list:
        Pres=Pres*zz(s,sG,Mres**2)
    return Pres

def disp_ff(F,s,sG,s0,params,MB,MM,Mres_list,chi): 
    
    for mres in Mres_list:
        if mres > np.sqrt(sG):
            print("Warning: resonance has crossed threshold")
            print(Mres_list,np.sqrt(sG))
    
    result=0
    w=(MB**2+MM**2-s)/(2*MB*MM)
    ss=s
    if gvar.mean(w-1.0)<=ccp.deltasafe:
        ss =MB**2+MM**2- 2*MB*MM*(1+ccp.deltasafe)
    if ss==0:
        ss=ccp.deltasafe
    smax = (MB-MM)**2
    zi=zz(s,sG,s0)
    zismax=zz(smax,sG,s0)
    FalphaBM=alphaBM(sG,s0,MB,MM)
    for n in ORTHOPOLS:
        result = result + params['a^'+str(F)+'_'+str(n)]*pn(n,FalphaBM,zi)  
                         
    result=result/(blaschke_factor_1p(ss,sG,Mres_list)*phi(F,ss,sG,s0,chi,1.0,MB,MM))
    return result
    
    
    
    
print('Checking our orthonormal polynomial implementation against reference values from EOS, szego-polynomial_TEST.cc, up to O(z^5)...')

eosP0p1=[+0.6351351032391984, -0.651261, +1.09668 , -1.76188 , +2.82970 , -4.54691  ] 
eosZ0p0=[+0.6351351032391984, -0.749503, +1.304458, -2.237009, +3.834085, -6.577731 ]    
eosM0p1=[+0.6351351032391984, -0.847745, +1.54489 , -2.82388 , +5.166081, -9.464102 ]    
  

failed = False
polychecktol = 1e-4
  
checkz=0.1
print("z = "+str(checkz)+" @ alpha =1.239475...")
for n in range(6):
    if abs(pn(n,1.239475,checkz)-eosP0p1[n])>polychecktol:
        failed=True
        print('failed z=',checkz,'n = ',n,':',pn(n,1.239475,checkz),'!=',eosP0p1[n])
if not failed: print('passed.')
checkz=0.0
print("z = "+str(checkz)+" @ alpha =1.239475 :")
for n in range(6):
    if abs(pn(n,1.239475,checkz)-eosZ0p0[n])>polychecktol:
        failed=True
        print('failed z=',checkz,'n = ',n,':',pn(n,1.239475,checkz),'!=',eosZ0p0[n])
if not failed: print('passed.')
checkz=-0.1
print("z = "+str(checkz)+" @ alpha =1.239475 :")
for n in range(6):
    if abs(pn(n,1.239475,checkz)-eosM0p1[n])>polychecktol:
        failed=True
        print('failed z=',checkz,'n = ',n,':',pn(n,1.239475,checkz),'!=',eosM0p1[n])
if not failed: print('passed.')
    
    
    
    
    
    
    
    

