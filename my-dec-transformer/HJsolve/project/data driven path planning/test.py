#!/usr/bin/env python3

# import numpy as np
# import sys
from ctypes import *

import numpy as np

from generate_file import *

# import matplotlib.pyplot as plt
# def plot_contour(timestep):
# 	plt.clf()
# 	root="/home/revanth/Mtech/bp/cpp2/openmp4/"
# 	folder=root+"include/testlogs/zcs/"
# 	file="{}.txt".format(timestep)

# 	with open(folder+file,"r") as f:
# 		text=f.readlines()
# 	lines=[i.strip("\n").split(")") for i in text]
# 	contours=[[]]
# 	for line in lines:
# 		if(len(line)==2):
# 			index=int(line[0])
# 			pts=[float(t) for t in line[1].split(",")]
# 			if(len(contours)==index-1):
# 				contours.append([])
# 			contours[index-1].append(pts)
# 	for contour in contours:
# 		ilist=[i[0] for i in contour]
# 		jlist=[i[1] for i in contour]
# 		plt.plot(jlist,ilist)
# 	#plt.show()
# 	plt.savefig(root+"include/figures/{}.png".format(timestep))
# def plot_allcontours(total_timesteps):
# 	plt.clf()
# 	root="/home/revanth/Mtech/bp/cpp2/openmp4/"
# 	folder=root+"include/testlogs/zcs/"
# 	for t in range(total_timesteps):
# 		file="{}.txt".format(t)

# 		with open(folder+file,"r") as f:
# 			text=f.readlines()
# 		lines=[i.strip("\n").split(")") for i in text]
# 		contours=[[]]
# 		for line in lines:
# 			if(len(line)==2):
# 				index=int(line[0])
# 				pts=[float(t) for t in line[1].split(",")]
# 				if(len(contours)==index-1):
# 					contours.append([])
# 				contours[index-1].append(pts)
# 		for contour in contours:
# 			ilist=[i[0] for i in contour]
# 			jlist=[i[1] for i in contour]
# 			plt.plot(jlist,ilist)
# 	plt.savefig(root+"include/figures/all.png")

# def plot_contours_with_path(total_timesteps):
# 	plt.clf()
# 	root="/home/revanth/Mtech/bp/cpp2/openmp4/"
# 	folder=root+"include/testlogs/zcs/"
# 	for t in range(total_timesteps):
# 		file="{}.txt".format(t)

# 		with open(folder+file,"r") as f:
# 			text=f.readlines()
# 		lines=[i.strip("\n").split(")") for i in text]
# 		contours=[[]]
# 		for line in lines:
# 			if(len(line)==2):
# 				index=int(line[0])
# 				pts=[float(t) for t in line[1].split(",")]
# 				if(len(contours)==index-1):
# 					contours.append([])
# 				contours[index-1].append(pts)
# 		for contour in contours:
# 			ilist=[i[0] for i in contour]
# 			jlist=[i[1] for i in contour]
# 			plt.plot(jlist,ilist)
# 	#print("all co")
# 	file="path.txt"
# 	with open(folder+file,"r") as f:
# 		text=f.readlines()
# 	lines=[i.strip("\n").split(",") for i in text]
# 	#print(lines)
# 	path=[]

# 	for line in lines:
# 		if(len(line)==2):
# 			path.append([float(line[0]),float(line[1])])
# 	#print(path)
# 	ilist=[i[0] for i in path]
# 	jlist=[i[1] for i in path]
# 	plt.plot(jlist,ilist)
# 	plt.scatter(jlist,ilist)
# 	plt.savefig(root+"include/figures/path.png")

@CFUNCTYPE(c_void_p,c_int)
def plot_allcontours_with_path(total_timesteps):
	# plot_allcontours(total_timesteps)
	# #print("kkk")
	# plot_contours_with_path(total_timesteps)
	pass
idim=Grid.shape[0]
jdim=Grid.shape[1]


cfunc=CFUNCTYPE(c_void_p,c_int,POINTER(POINTER(c_int)),POINTER(POINTER(c_double)),POINTER(POINTER(c_double)),POINTER(POINTER(c_double)))
@cfunc
def assign(timestep,nb,F,u,v):
	#print("assigning")
	# if (timestep>0):
	# 	plot_contour(timestep-1)
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
#	for i in range(idim):
#		for j in range(jdim):
#			F[i][j]=Fl[i][j]
#			u[i][j]=ul[i][j]
#			v[i][j]=vl[i][j]
	return
cfunc2=CFUNCTYPE(c_void_p,c_int,POINTER(c_double),POINTER(c_double),POINTER(c_double),c_int,c_int)
@cfunc2
def assign_bvels(timestep,F,u,v,i,j):
	Fl,ul,vl=create_files(timestep,i,j)
	for i in range(4):
		F[i]=Fl[i]
		u[i]=ul[i]
		v[i]=vl[i]

	return

lib=CDLL("/home/revanth/Mtech/projct/c++/openmp/include/libgrid.so.1")
#lib2=CDLL("/home/chennamsai/interp/mistake/libmistake.so.1")
# lib.data_preparation.restype=c_void_p
#lib.data_preparation.argtypes=[c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_float,c_float,c_int,c_int,c_int,cfunc]
print("aaa")
#plot_allcontours_with_path(958)#
start=(227,261)
destination=(165,72)
for threads in [1]:
	print(lib.data_preparation(idim,jdim,assign,assign_bvels,plot_allcontours_with_path,c_int(start[0]),c_int(start[1]),c_int(destination[0]),c_int(destination[1]),1000,5,c_float(.5),c_float(time_res_mins*60),c_float(lon_len),c_float(lat_len),c_int(threads),c_void_p(0),0))
# print("ddd")
# print(lib.data_preparation(idim,jdim,assign,20,20,90,90,1000,5,c_float(.5),c_float(.1),1,1,4))
#print(lib2.nonsense(assign_bvels,702,155,1451,1239,4674,c_float(120),c_float(4452.77832),c_float(4422.970908)))
