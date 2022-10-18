from math import ceil,floor,atan,pi
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from HJsolve.project.python.level_set import *
from HJsolve.project.python.advection import *

class Scheme():
    def __init__(self,dimensions,F,u,v,dX,dt,order,advection_term,fluxlimiter,noentry_slices):
        
        self._F=F
        self._u=u
        self._v=v
        self._dX=dX
        self._dt=dt
        self._order=order
        self._dims=dimensions
        self._advection_term=advection_term
        self._fluxlimiter=fluxlimiter
        self._timestep_count=0
        self._phi=[]
        self._islices=noentry_slices[0]
        self._jslices=noentry_slices[1]
        self._boundary_points=[]
    def _interpolate(self):
        for pt in self._boundary_points:
            i,j=pt
            if(j>=2 and self._phi[i][j-2]!=float("inf") and self._phi[i][j-1]!=float("inf")):
                Fx_minus=2*self._phi[i][j-1]-self._phi[i][j-2]
            else:
                Fx_minus=float("inf")

            if(j<self._dims[0]-2 and self._phi[i][j+2]!=float("inf") and self._phi[i][j+1]!=float("inf")):
                Fx_plus=2*self._phi[i][j+1]-self._phi[i][j+2]
            else:
                Fx_plus=float("inf") 
            
            
            if(Fx_plus!=float("inf") and Fx_minus!=float("inf")):
                Fx=(Fx_plus+Fx_minus)/2
            elif(Fx_plus==float("inf") and Fx_minus!=float("inf")):
                Fx=Fx_minus
            elif(Fx_minus==float("inf") and Fx_plus!=float("inf")):
                Fx=Fx_plus
            else:
                Fx=float("inf")

            if(i>=2 and self._phi[i-2][j]!=float("inf") and self._phi[i-1][j]!=float("inf")):
                Fy_minus=2*self._phi[i-1][j]-self._phi[i-2][j]
            else:
                Fy_minus=float("inf")

            if(i<self._dims[1]-2 and self._phi[i+2][j]!=float("inf") and self._phi[i+1][j]!=float("inf")):
                Fy_plus=2*self._phi[i+1][j]-self._phi[i+2][j]
            else:
                Fy_plus=float("inf") 
            
            
            if(Fy_plus!=float("inf") and Fy_minus!=float("inf")):
                Fy=(Fy_plus+Fy_minus)/2
            elif(Fy_plus==float("inf") and Fy_minus!=float("inf")):
                Fy=Fy_minus
            elif(Fy_minus==float("inf") and Fy_plus!=float("inf")):
                Fy=Fy_plus
            else:
                Fy=float("inf")

            if(Fy_plus!=float("inf") and Fy_minus!=float("inf")):
                Fy=(Fy_plus+Fy_minus)/2
            elif(Fy_plus==float("inf") and Fy_minus!=float("inf")):
                Fy=Fy_minus
            elif(Fy_minus==float("inf") and Fy_plus!=float("inf")):
                Fy=Fy_plus
            else:
                Fy=float("inf")

            if(Fy!=float("inf") and Fx!=float("inf")):
                self._phi[i][j]=(Fy*self._dX[1]+Fx*self._dX[0])/(self._dX[1]+self._dX[0])
            elif(Fy==float("inf") and Fx!=float("inf")):
                self._phi[i][j]=Fx
            elif(Fx==float("inf") and Fy!=float("inf")):
                self._phi[i][j]=Fy
            else:
                pass
        return self._phi
    def _scheme(self,Fval,i,j):
        levelset_part=self.__level_set(Fval,i,j)
        advection_part=self.__advection(i,j)

        del_phi=levelset_part+advection_part

        phi_level=self._phi[i][j]+del_phi
        
        return phi_level

    def __level_set(self,Fval,i,j):
        if(self._order==2):
            levelset_part=second_order_scheme(self._phi,Fval,i,j,self._dX,self._dt,self._dims)
        elif(self._order==1):
            levelset_part=level_upwind_scheme(self._phi,Fval,i,j,self._dX,self._dt,self._dims)

        
        return levelset_part
    
    def __advection(self,i,j):
        if(self._advection_term==1):

            advection_part=tvd_advection(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        elif(self._advection_term==2):
            advection_part=upwind_scheme(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        elif(self._advection_term==3):
            #print("using lax")
            advection_part=lax_wendroff(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        elif(self._advection_term==4):
            advection_part=tvd_lax_wendroff(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        elif(self._advection_term==5):
            advection_part=nondiffusive_lax_wendroff(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        elif(self._advection_term==6):
            advection_part=lax_wendroff_wiki(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        elif(self._advection_term==7):
            advection_part=lax_wendroff_book(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        elif(self._advection_term==8):
            advection_part=tvd_lax_wendroff_own(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        elif(self._advection_term==9):
            advection_part=lax_wendroff_own(self._phi,self._u[self._timestep_count],self._v[self._timestep_count],i,j,self._dX,self._dt,self._fluxlimiter)
        
        return advection_part
    
    def _level_set_half_step(self):
        self._boundary_points=[]
        
        phi_next=np.ones(self._dims)*float("inf")
        for i,j in self._narrowband:
            if(i< self._dims[0]-self._order-1 and i>=1 and j< self._dims[0]-self._order-1 and j>=1):
            
                Fval=self._F[self._timestep_count][i][j]
                del_phi=self.__level_set(Fval,i,j)
                phi_next[i][j]=self._phi[i][j]+.5*del_phi
                
            else:
                phi_next[i][j]=self._phi[i][j]
        if(self._islices!=[]):
            for islice,jslice in zip(self._islices,self._jslices):
                phi_next[islice,jslice]=float("inf")
        return phi_next
    
    def _advection_half_step(self):
        phi_next=np.ones(self._dims)*float("inf")
        for i,j in self._narrowband:
            if(i< self._dims[0]-self._order-1 and i>=2 and j< self._dims[0]-self._order-1 and j>=2):
            
                phi_next[i][j]=self._phi[i][j]+self.__advection(i,j)
            else:
                phi_next[i][j]=self._phi[i][j]
        if(self._islices!=[]):
            for islice,jslice in zip(self._islices,self._jslices):
                phi_next[islice,jslice]=float("inf")
        return phi_next
