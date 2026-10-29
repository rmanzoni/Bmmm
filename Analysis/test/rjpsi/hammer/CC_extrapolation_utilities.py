
import gvar
from numpy import *
import numpy as np
import random

import load_chi_u12 as CHI
import dispersive_functions as df
from CC_fit_parameters import *

               
def MBDEPFUNC(j,lambdaqcdphys,mH):
        DELTAMB = 1.0#gvar.sqrt(metab/METABPHYS)
        MHapprox = mH-MBCPHYS+MBPHYS
        if not (j==0):DELTAMB = DELTAMB*((lambdaqcdphys/(MHapprox))**(j)-(lambdaqcdphys/(MBPHYS))**(j))
        return DELTAMB

def MRESFUNC(p,MHc):              # List of resonances, as a function of MHc 
    resonances = {}
    for curr in RESONANCES:
        resonances[curr]=[(MHc+(MBcr-MBCPHYS)) for MBcr in RESONANCES[curr]]
    return resonances
    
def S_G(p,MHc,MJpsi):                  # Start of the branch cut (M_H+M_D*)^2
    return (MHc+(MBPHYS+MDSPHYS-MBCPHYS))**2

def lambda_kin(MHc,MJpsi,s):                                           # Kallen function           
    return MHc**4+MJpsi**4+s**2-2*(MHc**2*MJpsi**2+MHc**2*s+MJpsi**2*s)    
        
########################################################################################################################################################################

def ChiContFF(p,CURR,deltamc,deltamssea,deltamcsea,deltachi,MHc,MJpsi,lambdaqcdphys,mHforDEL,chisusc,mhbar,qsq):
        dfparams={}  
        resonances = MRESFUNC(p,MHc)
        for curr in RESONANCES:
          NLIST=[0,1,2,3,4]
          if (curr=='T23' or curr=='A12' or curr=='A0' or curr=='T1' ):NLIST=[1,2,3,4]       
          for n in NLIST:  
            mhdependencefac=1
            for j in range(1,NEXPANSION1):
                mhdependencefac=mhdependencefac+p[curr+str(n)+'_phys_factor_'+str(j)]*MBDEPFUNC(j,lambdaqcdphys,mHforDEL)  
            dfparams['a^'+str(curr)+'_'+str(n)] = p['a^'+str(curr)+'_'+str(n)]*mhdependencefac#*deltacfactor
                
        s=qsq
        sG=S_G(p,MHc,MJpsi)           # s_Gamma, the momentum transfer squared of the start of the branch cut - i.e. M_B+M_D*
        s0=(MHc-MJpsi)**2    
        
        dfparams['a^A12_0']=geta0A12qsqmaxcond(sG,s0,dfparams,MHc,MJpsi,resonances,chisusc,mhbar)
        dfparams['a^T23_0']=geta0T23qsqmaxcond(sG,s0,dfparams,MHc,MJpsi,resonances,chisusc,mhbar)
        
        dfparams['a^A0_0']=geta0A0qsqzerocond(sG,s0,dfparams,MHc,MJpsi,resonances,chisusc,mhbar)        
        dfparams['a^T1_0']=geta0T1qsqzerocond(sG,s0,dfparams,MHc,MJpsi,resonances,chisusc,mhbar)  
        
        for curr in RESONANCES:
          NLIST=[0,1,2,3,4]     
          for n in NLIST:
            deltacfactor = 1              
            dfparams['a^'+str(curr)+'_'+str(n)] = dfparams['a^'+str(curr)+'_'+str(n)]*deltacfactor    
        
        mhpart=mhbar**(2*chimfac[CURR]) 
        
        return  df.disp_ff(CURR,s,sG,s0,dfparams,MHc,MJpsi,resonances[CURR],chisusc[CURR]/mhpart) 
        
        
        
def geta0A12qsqmaxcond(sG,s0,params,MB,MM,Mres_list,chi,mhbar):   
        smax = (MB-MM)**2-deltasafe
        zismax=df.zz(smax,sG,s0)
        FalphaBM=df.alphaBM(sG,s0,MB,MM) 
        
        
        PN=[df.pn(n,FalphaBM,zismax) for n in  [0,1,2,3,4]]
        
        A12smaxfactor=(MB+MM)*(MB**2-MM**2-smax)/(16*MB*MM**2)
        A1smax=0
        A12smaxrem=0
        
        A1Pphimax  = (df.blaschke_factor_1p(smax,sG,Mres_list['A1'])*df.phi('A1' ,smax,sG,s0,chi['A1']/mhbar**2,1.0,MB,MM))
        A12Pphimax = (df.blaschke_factor_1p(smax,sG,Mres_list['A12'])*df.phi('A12',smax,sG,s0,chi['A12']/mhbar**2,1.0,MB,MM))
        
        for n in [0,1,2,3,4]:
          A1smax = A1smax + params['a^A1_'+str(n)]*PN[n]
        A1smax=A1smax/A1Pphimax
         
        for n in [1,2,3,4]:
          A12smaxrem = A12smaxrem + params['a^A12_'+str(n)]*PN[n]
          
        # set a_0^A12 = Ppi(A12,t_-)*A1(t_-)*(MB+MM)*(MB**2-MM**2-t_-)/(16*MB*MM**2) - sum_1^N a_n^A12 p_n(z(t_-))  
        
        result = (A1smax*A12smaxfactor*A12Pphimax-A12smaxrem)/PN[0]
        
        return result        
        
def geta0A0qsqzerocond(sG,s0,params,MB,MM,Mres_list,chi,mhbar):   
        szero = deltasafe 
        ziszero=df.zz(szero,sG,s0)
        FalphaBM=df.alphaBM(sG,s0,MB,MM) 
        
        
        PN=[df.pn(n,FalphaBM,ziszero) for n in  [0,1,2,3,4]]
        
        A1szero=0
        A12szero=0
        
        A0Pphizero  = (df.blaschke_factor_1p(szero,sG,Mres_list['A0'])*df.phi('A0' ,szero,sG,s0,chi['A0'],1.0,MB,MM))
        A1Pphizero  = (df.blaschke_factor_1p(szero,sG,Mres_list['A1'])*df.phi('A1' ,szero,sG,s0,chi['A1']/mhbar**2,1.0,MB,MM))
        A12Pphizero = (df.blaschke_factor_1p(szero,sG,Mres_list['A12'])*df.phi('A12',szero,sG,s0,chi['A12']/mhbar**2,1.0,MB,MM))
        
        
        for n in [0,1,2,3,4]:
          A1szero  = A1szero  + params['a^A1_'+str(n)]*PN[n]
          A12szero = A12szero + params['a^A12_'+str(n)]*PN[n]
          
        A1szero=A1szero/A1Pphizero
        A12szero=A12szero/A12Pphizero
                                
        lambdabdstar = lambda_kin(MB,MM,szero)              
        A2szero = ((MB+MM)**2*(MB**2-MM**2-szero)*A1szero- (16*MB*MM**2*(MB+MM))*A12szero)/lambdabdstar        
         
        A0szerorem=0
        for n in [1,2,3,4]:
          A0szerorem = A0szerorem + params['a^A0_'+str(n)]*PN[n]    
        
        result = ((A1szero*(MB+MM)/(2*MM)-A2szero*(MB-MM)/(2*MM) )*A0Pphizero-A0szerorem)/PN[0]
        
        return result
        
        
def geta0T23qsqmaxcond(sG,s0,params,MB,MM,Mres_list,chi,mhbar):   
        smax = (MB-MM)**2-deltasafe
        zismax=df.zz(smax,sG,s0)
        FalphaBM=df.alphaBM(sG,s0,MB,MM) 
        
        PN=[df.pn(n,FalphaBM,zismax) for n in  [0,1,2,3,4]]
        
        T23smaxfactor=(MB+MM)*(MB**2+3*MM**2-smax)/(8*MB*MM**2)
        T2smax=0
        T23smaxrem=0
        
        T2Pphimax  = (df.blaschke_factor_1p(smax,sG,Mres_list['T2'])*df.phi('T2' ,smax,sG,s0,chi['T2']/mhbar**2,1.0,MB,MM))
        T23Pphimax = (df.blaschke_factor_1p(smax,sG,Mres_list['T23'])*df.phi('T23',smax,sG,s0,chi['T23']/mhbar**2,1.0,MB,MM))
        
        for n in [0,1,2,3,4]:
          T2smax = T2smax + params['a^T2_'+str(n)]*PN[n]
        T2smax=T2smax/T2Pphimax
         
        for n in [1,2,3,4]:
          T23smaxrem = T23smaxrem + params['a^T23_'+str(n)]*PN[n]
          
        # set a_0^T23 = Ppi(T23,t_-)*T2(t_-)*(MB+MM)*(MB**2+3*MM**2-t_-)/(8*MB*MM**2) - sum_1^N a_n^T23 p_n(z(t_-))  
        
        result = (T2smax*T23smaxfactor*T23Pphimax-T23smaxrem)/PN[0]
        return result
        
def geta0T1qsqzerocond(sG,s0,params,MB,MM,Mres_list,chi,mhbar):   
        szero = deltasafe #(MB-MM)**2
        ziszero=df.zz(szero,sG,s0)
        FalphaBM=df.alphaBM(sG,s0,MB,MM) 
        
        PN=[df.pn(n,FalphaBM,ziszero) for n in  [0,1,2,3,4]]
        
        T1szerorem=0
        T2szero=0
        
        T1Pphizero  = (df.blaschke_factor_1p(szero,sG,Mres_list['T1'])*df.phi('T1' ,szero,sG,s0,chi['T1']/mhbar**2,1.0,MB,MM))
        T2Pphizero  = (df.blaschke_factor_1p(szero,sG,Mres_list['T2'])*df.phi('T2' ,szero,sG,s0,chi['T2']/mhbar**2,1.0,MB,MM))
        
        for n in [0,1,2,3,4]:
          T2szero  = T2szero  + params['a^T2_'+str(n)]*PN[n]
        T2szero=T2szero/T2Pphizero
        
        lambdabdstar = lambda_kin(MB,MM,szero)                      
         
        for n in [1,2,3,4]:
          T1szerorem = T1szerorem + params['a^T1_'+str(n)]*PN[n]      
        
        result = (T2szero*T1Pphizero-T1szerorem)/PN[0]  
        
        return result        
        
def FF_to_Mels(curr,QCDbasisFFs,MHc,MJpsi,amc,amh,pp):

        EJpsi = gvar.sqrt(MJpsi**2+2*pp*pp)
        qsq = (MHc-EJpsi)**2-2*pp*pp
        w=EJpsi/MJpsi
        ksq = 2*pp*pp                       
        
        r=MJpsi/MHc
        relnorm = gvar.sqrt(2.0*EJpsi*2.0*MHc*(1.0+pp*pp/(MJpsi**2)))
        relnormZ = gvar.sqrt(2.0*EJpsi*2.0*MHc)
        
        FACTORA1 = (MJpsi+MHc) 
        FACTORA0 = (2.0*EJpsi*MHc)/((amc+amh)*MJpsi)                                                       #factor of 1/k in fits
        FACTORV = (2.0*MHc)/(MHc+MJpsi)                                                                    #factor of 1/k in fits
        
        AZEROFACTOR = -2.0*pp*pp*EJpsi*MHc/(qsq*MJpsi)
        AONEFACTOR = (MHc+MJpsi)*(1.0+(pp*pp/(MJpsi*MJpsi))    + (EJpsi*MHc*pp*pp)/(MJpsi*MJpsi*qsq)  )
        ATWOFACTOR = ((pp*pp*EJpsi*MHc)/(MJpsi*MJpsi*(MHc+MJpsi)))*(1.0+(MHc**2-MJpsi**2)/qsq)
        
        
        A1q=QCDbasisFFs["A1"]
        MA1ind=FACTORA1*A1q/relnormZ
        MA1 = MA1ind
        
        if not curr == 'A1':
                A0q =QCDbasisFFs["A0"]
                A12q=QCDbasisFFs["A12"]
                Vq  =QCDbasisFFs["V"]
        
                                      
                T1q =QCDbasisFFs["T1"]
                T2q =QCDbasisFFs["T2"]
                T23q =QCDbasisFFs["T23"]
                
                                
                lambdabdstar = lambda_kin(MHc,MJpsi,qsq)

                A2q = ((MHc+MJpsi)**2*(MHc**2-MJpsi**2-qsq)*A1q- (16*MHc*MJpsi**2*(MHc+MJpsi))*A12q)/lambdabdstar
                T3q = ((MHc**2+3*MJpsi**2-qsq)*(MHc+MJpsi)*T2q - T23q/(1.0/(8*MHc*MJpsi**2)))*(MHc-MJpsi)/lambdabdstar  
                
                
                #if curr == 'T1':print("Tensor FFs 1,2,23,3:\t",T1q,"\t",T2q,"\t",T23q,"\t",T3q)                       
                
                hT1 = ((1+r)*(T2q+r**2*T2q-2*r*(T2q+T1q*(-1+w))))/(2*np.sqrt(r)*(1+r**2-2*r*w))
                hT2 = -(((-1+r)*(T2q+r**2*T2q-2*r*(T1q-T2q+T1q*w)))/(2*np.sqrt(r)*(1+r**2-2*r*w)))
                hT3 = -((2*np.sqrt(r)*((-1+r**2)*T1q+T2q-r**2*T2q+T3q*(1+r**2-2*r*w)))/((-1+r**2)*(1+r**2-2*r*w)))
                
                
                #if curr == 'T1':print("HQET FFs 1,2,3:\t",hT1,"\t",hT2,"\t",hT3)      
              
                hT3red   =     hT3*(w*pp**2/MJpsi**2)                                                                       #
                hT1mhT2  =     hT1 -   hT2                                                                                  #
                hT1phT2  =   2*hT1 - (1-w)*hT1mhT2                                                                          #
                
                #if curr == 'T1':print("HQET reduced FFs 1,2,3:\t",hT3red,"\t",hT1mhT2,"\t",hT1phT2)     
                                                                  
                gppgm     =    hT1phT2*gvar.sqrt(MHc*MJpsi)                                                                 #
                gpmgm     =    hT1mhT2*gvar.sqrt(MHc*MJpsi)/MJpsi                                                           #factor of 1/k in fits
                gpmgmpg0  = -( hT3red + hT1mhT2*w*pp**2/MJpsi**2 - hT1phT2*(1+pp**2/MJpsi**2))*gvar.sqrt(MHc*MJpsi)         #
                
                
                #if curr == 'T1':print("gXXXXs 1,2,3:\t",gppgm,"\t",gpmgm,"\t",gpmgmpg0)  
                                
                
                MT1= gpmgm/relnormZ
                MT2= gppgm/relnormZ
                MT23= gpmgmpg0/relnorm 
                
                MA0ind=FACTORA0*A0q/relnorm                                                        
                MA0=MA0ind
                MA2ind=(A1q*AONEFACTOR + A0q*AZEROFACTOR-A2q*ATWOFACTOR )/relnorm
                MA2=MA2ind
                
                #by convention, we put the factor of k back into the matrix element for V
                
                MV=FACTORV*Vq*pp/relnorm                                                                                     
                
        if curr == 'A0' :ret = MA0
        if curr == 'A1' :ret = MA1
        if curr == 'A2': ret = MA2
        if curr == 'V'  :ret = MV 
        
        if curr == 'T1' :ret = MT1
        if curr == 'T2' :ret = MT2
        if curr == 'T23':ret = MT23
        
        return ret
        

def Mels_to_FFs(curr,Mels,MHc,MJpsi,amc,amh,pp):  
    EJpsi = gvar.sqrt(MJpsi**2+2*pp*pp)
    qsq = (MHc-EJpsi)**2-2*pp*pp
    w=EJpsi/MJpsi
    ksq = 2*pp*pp
    MDp = MJpsi
    MHc = MHc
    EJpsi = EJpsi
        
    r=MJpsi/MHc
    relnorm = gvar.sqrt(2.0*EJpsi*2.0*MHc*(1.0+pp*pp/(MJpsi**2)))
    relnormZ = gvar.sqrt(2.0*EJpsi*2.0*MHc)
                          
    FACTORA1 = (MJpsi+MHc) 
    FACTORA0 = (2.0*EJpsi*MHc)/((amc+amh)*MJpsi)                                                                      #factor of 1/k in fits
    FACTORV = (2.0*MHc)/(MHc+MJpsi)                                                                                   #factor of 1/k in fits
     
    AZEROFACTOR = -2.0*pp*pp*EJpsi*MHc/(qsq*MJpsi)
    AONEFACTOR = (MHc+MJpsi)*(1.0+(pp*pp/(MJpsi*MJpsi))    + (EJpsi*MHc*pp*pp)/(MJpsi*MJpsi*qsq)  )
    ATWOFACTOR = ((pp*pp*EJpsi*MHc)/(MJpsi*MJpsi*(MHc+MJpsi)))*(1.0+(MHc**2-MJpsi**2)/qsq)
        
    MA1 =Mels["MA1"]
    A1q=MA1/FACTORA1*relnormZ    
    
    if curr=='A1': ret = A1q
    
    
    if not curr=='A1': 
                                                         
        
        MA0 =Mels["MA0"]
        MA2 =Mels["MA2"]
        MV  =Mels["MV"]
        
        MT1 =Mels["MT1"]
        MT2 =Mels["MT2"]
        MT23 =Mels["MT23"]
        
        ####################
                             
        gpmgm=MT1*relnormZ
        gppgm=MT2*relnormZ
        gpmgmpg0=MT23*relnorm 
                                                        
        #if curr == 'T1':print("gXXXXs 1,2,3:\t",gppgm,"\t",gpmgm,"\t",gpmgmpg0)  
                
        hT1phT2 = gppgm/gvar.sqrt(MHc*MJpsi)                                
        hT1mhT2 = MJpsi*gpmgm/gvar.sqrt(MHc*MJpsi)                                                        #factor of 1/k in fits
        hT3red = -gpmgmpg0/gvar.sqrt(MHc*MJpsi)-hT1mhT2*w*pp**2/MJpsi**2 + hT1phT2*(1+pp**2/MJpsi**2)
        
        #if curr == 'T1':print("HQET reduced FFs 1,2,3:\t",hT3red,"\t",hT1mhT2,"\t",hT1phT2)     
        
        hT1 = ((1-w)*hT1mhT2 + hT1phT2)/2.0                                               
        hT2 = hT1-hT1mhT2                                                                
        hT3 = hT3red/(w*pp**2/MJpsi**2)                                                  
                                        
        #if curr == 'T1':print("HQET FFs 1,2,3:\t",hT1,"\t",hT2,"\t",hT3)                                   
        
        T1q = -(1/(2*sqrt(r)))*(  (1-r)*hT2 - (1+r)*hT1  )                              
        T2q = (1/(2*sqrt(r)))*(  2*r*(w+1)*hT1/(1+r) - 2*r*(w-1)*hT2/(1-r))          
        T3q = (1/(2*sqrt(r)))*(  (1-r)*hT1 - (1+r)*hT2 + (1-r**2)*hT3  )
        
        lambdabdstar = lambda_kin(MHc,MJpsi,qsq)
        T23q = (1.0/(8*MHc*MJpsi**2))*((MHc**2+3*MJpsi**2-qsq)*(MHc+MJpsi)*T2q - lambdabdstar*T3q/(MHc-MJpsi))
        
                
        #if curr == 'T1':print("Tensor FFs 1,2,23,3:\t",T1q,"\t",T2q,"\t",T23q,"\t",T3q)

        ####################
        
        #by convention, we put the factor of k back into the matrix element for V
        Vq=MV/(FACTORV*pp/relnorm)
        
        A0q=MA0*relnorm/FACTORA0   
        A2q=-(MA2*relnorm-A0q*AZEROFACTOR-A1q*AONEFACTOR)/ATWOFACTOR
        A12q = ((MHc+MJpsi)**2*(MHc**2-MJpsi**2-qsq)*A1q-  A2q*lambdabdstar )/(16*MHc*MJpsi**2*(MHc+MJpsi))
        
        
        if curr == 'A0':  ret  = A0q 
        if curr == 'A12': ret  = A12q
        if curr == 'V':   ret  = Vq  
        
        if curr == 'T1':  ret  = T1q 
        if curr == 'T2':  ret  = T2q 
        if curr == 'T23': ret  = T23q      
        
    return ret
########################################################################################################################################################################

if True:   # Check converting from matrix elements to form factors using Mels_to_FFs() and then back again using FF_to_Mels() returns the input values
  print('Checking matrix element to FF conversions...')
  Nchecks = 1000
  failed = False
  for i in range(Nchecks):
    QCDbasisFFs     = {}
    QCDbasisFFstest = {}
    Mels            = {}
    QCDbasisFFs["A0"]    = random.random()
    QCDbasisFFs["A1"]    = random.random()
    QCDbasisFFs["A12"]   = random.random()
    QCDbasisFFs["V"]     = random.random()                    
    QCDbasisFFs["T1"]    = random.random()
    QCDbasisFFs["T2"]    = random.random()
    QCDbasisFFs["T23"]   = random.random()
    MHc   = MBCPHYS*(random.random()+1.0)/2.0
    MJpsi = MJPPHYS*(random.random()+1.0)/2.0
    amc =  deltasafe+random.random()
    amh =  deltasafe+random.random()
    pp  =  deltasafe+random.random()
    for X in ['V','A0','A1','A2','T1','T2','T23']:
        Mels['M'+X]            = FF_to_Mels(X,QCDbasisFFs,MHc,MJpsi,amc,amh,pp)
    #print(Mels)
    for X in ['V','A0','A1','A12','T1','T2','T23']:
        QCDbasisFFstest[X]     = Mels_to_FFs(X,Mels      ,MHc,MJpsi,amc,amh,pp)
    for X in QCDbasisFFs:
        if abs(QCDbasisFFs[X]-QCDbasisFFstest[X])>deltasafe:
            failed=True
            print('Failed Check -',X,':',QCDbasisFFs[X],'!=',QCDbasisFFstest[X])
  if not failed:
      print('Checks completed.')
        


