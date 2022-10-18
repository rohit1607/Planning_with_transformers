#!/usr/bin/env python3

# import numpy as np
# import sys
from ctypes import *

import numpy as np

from generate_file import *

#plotting func
@CFUNCTYPE(c_void_p,c_int)
def plot_allcontours_with_path(total_timesteps):
	pass
idim=Grid.shape[0]
jdim=Grid.shape[1]

#ctype wrapper for create grid file 
cfunc=CFUNCTYPE(c_void_p,c_int,POINTER(POINTER(c_int)),POINTER(POINTER(c_double)),POINTER(POINTER(c_double)),POINTER(POINTER(c_double)))
@cfunc
def assign(timestep,nb,F,u,v):
	narrowband=np.empty((idim,jdim))
	for i in range(idim):
		for j in range(jdim):
			narrowband[i][j]=nb[i][j]
	a=np.where(narrowband>0)
	nb_pts=np.asarray([(a[0][i],a[1][i]) for i in range(len(a[0]))])
	Fl,ul,vl=create_gridfile2(timestep,nb_pts)
	for i,j in nb_pts:
		F[i][j]=Fl[i][j]
		u[i][j]=ul[i][j]
		v[i][j]=vl[i][j]

	return

#create velocity for backtracking function
cfunc2=CFUNCTYPE(c_void_p,c_int,POINTER(c_double),POINTER(c_double),POINTER(c_double),c_int,c_int)
@cfunc2
def assign_bvels(timestep,F,u,v,i,j):
	Fl,ul,vl=create_files(timestep,i,j)
	for i in range(4):
		F[i]=Fl[i]
		u[i]=ul[i]
		v[i]=vl[i]

	return

lib=CDLL("/home/revanth/interp/include/libgrid.so")
print("aaa")
#plot_allcontours_with_path(958)#
for threads in [1]:
	print(lib.data_preparation(idim,jdim,assign,assign_bvels,plot_allcontours_with_path,702,155,665,452,10000,5,c_float(.5),c_float(240),c_float(4452.77832),c_float(4422.970908),c_int(threads),c_void_p(0),0))
