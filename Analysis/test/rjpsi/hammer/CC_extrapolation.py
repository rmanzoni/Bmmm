import matplotlib
import gvar
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import load_chi_u12 as CHI
import dispersive_functions as df
import CC_extrapolation_utilities as cce
from CC_fit_parameters import *

def fitprintA(curr,p,s,MHc,MJp,mhbar,u):
    chisusc={}
    for CURR in ['A0','A1','A12','V','T1','T2','T23']:
            chisusc[CURR]=CHI.fitprint(p,u,chidict[CURR])
            
    if not curr=='A2':
        retbit=cce.ChiContFF(p,curr,0,0,0,0,MHc,MJp,lambdaqcdphys,MHc,chisusc,mhbar,s)
    else:
        A1q=fitprintA('A1',p,s,MHc,MJp,mhbar,u)
        A12q=fitprintA('A12',p,s,MHc,MJp,mhbar,u)
        
        lambdabdstar = cce.lambda_kin(MHc,MJp,s)
        qsq=s
        A2q = ((MHc+MJp)**2*(MHc**2-MJp**2-qsq)*A1q- (16*MHc*MJp**2*(MHc+MJp))*A12q)/lambdabdstar
        retbit = A2q
    return retbit
    
print("Loading continuum form factor parameters...")
continuum_fit_posteriors  =  gvar.load('continuum_fit_posteriors.pydat')
print("done.")


print("Loading synthetic form factor data...")
synthetic_data   =  gvar.load('HPQCD_BcJpsi_FF.pydat')
print("done.")


print("Checking synthetic form factor data against parameterisation...")
failed=False
for curr in ['A0','A1','A12','V','T1','T2','T23']:   
    ZPLOTS=[] 
    for qsq in [10.0, 6.70, 3.35, 0.01]:
        #print(curr+"("+str(qsq)+") = ",synthetic_data[curr+"("+str(qsq)+")"])
        diffmean = abs(synthetic_data[curr+"("+str(qsq)+")"].mean-fitprintA(curr,continuum_fit_posteriors,qsq,MBCPHYS,MJPPHYS,mbphys,uphys).mean)
        diffsdev = abs(synthetic_data[curr+"("+str(qsq)+")"].sdev-fitprintA(curr,continuum_fit_posteriors,qsq,MBCPHYS,MJPPHYS,mbphys,uphys).sdev)
        if diffmean>1e-4:failed=True
        if diffsdev>1e-5:failed=True
        if failed:print('Failed ',curr+"("+str(qsq)+")",synthetic_data[curr+"("+str(qsq)+")"],fitprintA(curr,continuum_fit_posteriors,qsq,MBCPHYS,MJPPHYS,mbphys,uphys))
print("done.")

print("Loading matrix elements and masses...")
lattice_Mels_Ms_data      =  gvar.load('matrix_elements_and_masses.pydat')
print("done.")


print("Checking data correlation matrix eigenvalues...")

for latt in [0,1,2,3,4,5]:
        failed=False
        data_array=[]        
        data_array.append(lattice_Mels_Ms_data['MJpsi_'+LATTICECHARS[latt]+'c'])                    
        for mh in MASSES[latt]:
                data_array.append(lattice_Mels_Ms_data['MH_mh_'+str(mh)+'_'+LATTICECHARS[latt]+'c'])
                for MEL in ['MA0','MA1','MA2','MV','MT1','MT2','MT23']:
                        momenta = MOMENTA[latt][1:]                         # omit w=1 for matrix elements that go to zero there
                        if curr == 'A1':momenta = MOMENTA[latt]
                        for pp in momenta:
                                data_array.append(lattice_Mels_Ms_data[MEL+'_c_M_'+str(mh)+'_p_'+str(pp)+'_'+LATTICECHARS[latt]])
        evals = np.linalg.eigvalsh((gvar.evalcorr(data_array)))  
        for i in range(len(ENSEMBLE_DATA_EIGVALUES_CHECK[latt])):
            if abs(evals[i]-ENSEMBLE_DATA_EIGVALUES_CHECK[latt][i])>1e-6:
                failed=True
        if failed:
            print('failed correlation matrix eigenvalue check on',LATTICECHARS[latt])
        else:
            print('correlation matrix eigenvalue check on',LATTICECHARS[latt],'passed.')
print('done.')
        












#print("Matrix element and mass values and uncertainties:")
#for X in lattice_Mels_Ms_data:
#    print(X,'=',lattice_Mels_Ms_data[X])


#print("Continuum form factor parameters:")
#for X in continuum_fit_posteriors:
#    print(X,'=',continuum_fit_posteriors[X])


########################################################################################################################################################################################################
########################################################################################################################################################################################################
print("Plotting form factors...")
ylabels={}
ylabels['T1']='{T_1}'
ylabels['T2']='{T_2}'
ylabels['T23']='{T_{23}}'
ylabels['A0']='{A_0}'
ylabels['A1']='{A_1}'
ylabels['A12']='{A_{12}}'
ylabels['V']='{V}'

for curr in ['A0','A1','A12','V','T1','T2','T23']:
    fitprint=[[],[]]
    for q in range(100):
        QSQ = ((MBCPHYS-MJPPHYS)**2.0)*q/99.0
        fitprint[0].append(fitprintA(curr,continuum_fit_posteriors,QSQ,MBCPHYS,MJPPHYS,mbphys,uphys))
        s=QSQ
        sG=cce.S_G(continuum_fit_posteriors,MBCPHYS,MJPPHYS)         
        s0=(MBCPHYS-MJPPHYS)**2  
        fitprint[1].append(df.zz(s,sG,s0))
        
    plt.fill_between(fitprint[1], [x.mean+x.sdev for x in fitprint[0]], [x.mean-x.sdev for x in fitprint[0]], alpha=0.2,color='blue',label=r'$m_h=m_b$')
    plt.plot(fitprint[1], [x.mean for x in fitprint[0]],color='blue')
    
    for i in [2,1,0]:
        MHc=[3.760,4.829,5.895][i]
        mhbar =[1.95951,2.90051,3.85429][i]
        mbovmc=[1.78174,2.91971,4.12371][i]
        COL=['red','orange','green'][i]

        fitprint=[[],[]]
        for q in range(100):
            QSQ = ((MHc-MJPPHYS)**2.0)*q/99.0
            fitprint[0].append(fitprintA(curr,continuum_fit_posteriors,QSQ,MHc,MJPPHYS,mhbar,1/mbovmc))
            s=QSQ
            sG=cce.S_G(continuum_fit_posteriors,MHc,MJPPHYS)         
            s0=(MHc-MJPPHYS)**2
            fitprint[1].append(df.zz(s,sG,s0))
        
        plt.fill_between(
                         fitprint[1],
                         [x.mean+x.sdev for x in fitprint[0]], 
                         [x.mean-x.sdev for x in fitprint[0]], 
                         alpha=0.2,
                         color=COL,
                         label=r'$m_h='+str(mbovmc)[:4]+'\\times m_c$'
                         )
                       
        plt.plot(fitprint[1], [x.mean for x in fitprint[0]],color=COL)

    plt.xlabel(r'$z(q^2,t_0=q^2_\mathrm{max})$')#q^2/$Gev$^2$')
    plt.ylabel(r'$'+ylabels[curr]+'$')
    plt.legend(loc='upper right')
    print('FF = ',curr,':')
    plt.tight_layout()
    plt.show()
print("done.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


