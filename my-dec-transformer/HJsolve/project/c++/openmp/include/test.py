#!/usr/bin/env python3

# import numpy as np
# import sys
from ctypes import *



# dir="/home/quest1/black_pearl/"
# F=np.load(dir+"f_all1.npy")

# u=np.load(dir+"wa_u_all.npy")
# v=np.load(dir+"wa_v_all.npy")
# land=np.load(dir+"land_0.16.npy")
# # F=np.ones(F.shape)*8
# # u=np.zeros(u.shape)
# # v=np.zeros(v.shape)
# ##########################
# land2=np.empty(land.shape)
# for i in range(land.shape[0]):
#     for j in range(land.shape[1]):
#         if(i> 0 and j>0 and i<land.shape[0]-1 and j<land.shape[1]-1):
#             land2[i][j]=(((land[i-1][j]+land[i+1][j]+land[i][j-1]+land[i][j+1])/4)+land[i][j])/2
#         else:
#             land2[i][j]=land[i][j]
#################################

# s=np.asarray(np.where(land2==0))
# land=np.asarray(s,dtype=np.int)
# F[:,s[0],s[1]]=0
# u[:,s[0],s[1]]=0
# v[:,s[0],s[1]]=0
#land=np.ascontiguousarray(np.asarray(land).T,dtype=np.int)
#print(land.shape)
# for i in [1]:
# 	print("no of threads is {}".format(i))
# 	conv.data_prep(100,100,175,36,363,310,F.shape[0],5,0.5,1800,17811.11328,17691.883632,i,land)
# def main():
# 	idim=jdim=100
# 	F=np.ones((1000,idim,jdim))
# 	u=np.zeros((1000,idim,jdim))
# 	v=np.zeros((1000,idim,jdim))
# 	u[:,30:80,:]=0
# 	land=[]
# 	for i in range(30,35):
# 		for j in range(40,45):
# 			land.append([i,j])
# 	land=np.ascontiguousarray(np.asarray(land).T,dtype=np.int)
# 	print(land.shape)
# 	conv.data_prep(idim,jdim,20,20,90,90,1000,5,.5,.1,1,1,4,land)
import matplotlib.pyplot as plt
import numpy as np

def plot_contour(timestep):
	plt.clf()
	root="/home/revanth/Mtech/projct/c++/openmp/"
	folder=root+"include/testlogs/zcs/"
	file="{}.txt".format(timestep)
	#print("kjjkjj")
	with open(folder+file,"r") as f:
		text=f.readlines()
	lines=[i.strip("\n").split(")") for i in text]

	contours=[[]]
	for line in lines:
		if(len(line)==2):
			index=int(line[0])
			pts=[float(t) for t in line[1].split(",")]
			if(len(contours)==index-1):
				contours.append([])
			contours[index-1].append(pts)

	for contour in contours:
		ilist=[i[0] for i in contour]
		jlist=[i[1] for i in contour]
		plt.plot(jlist,ilist)
	#plt.show()
	plt.savefig(root+"include/figures/{}.png".format(timestep))
def plot_allcontours(total_timesteps):
	plt.clf()
	root="/home/revanth/Mtech/projct/c++/openmp/"
	folder=root+"include/testlogs/zcs/"
	for t in range(total_timesteps):
		file="{}.txt".format(t)

		with open(folder+file,"r") as f:
			text=f.readlines()
		lines=[i.strip("\n").split(")") for i in text]
		contours=[[]]
		for line in lines:
			if(len(line)==2):
				index=int(line[0])
				pts=[float(t) for t in line[1].split(",")]
				if(len(contours)==index-1):
					contours.append([])
				contours[index-1].append(pts)
		for contour in contours:
			ilist=[i[0] for i in contour]
			jlist=[i[1] for i in contour]
			plt.plot(jlist,ilist)
	plt.savefig(root+"include/figures/all.png")

def plot_contours_with_path(total_timesteps):
	plt.clf()
	root="/home/revanth/Mtech/projct/c++/openmp/"
	folder=root+"include/testlogs/zcs/"
	for t in range(total_timesteps):
		file="{}.txt".format(t)

		with open(folder+file,"r") as f:
			text=f.readlines()
		lines=[i.strip("\n").split(")") for i in text]
		contours=[[]]
		for line in lines:
			if(len(line)==2):
				index=int(line[0])
				pts=[float(t) for t in line[1].split(",")]
				if(len(contours)==index-1):
					contours.append([])
				contours[index-1].append(pts)
		for contour in contours:
			ilist=[i[0] for i in contour]
			jlist=[i[1] for i in contour]
			plt.plot(jlist,ilist)
	#print("all co")
	file="path.txt"
	with open(folder+file,"r") as f:
		text=f.readlines()
	lines=[i.strip("\n").split(",") for i in text]
	#print(lines)
	path=[]

	for line in lines:
		if(len(line)==2):
			path.append([float(line[0]),float(line[1])])
	#print(path)
	ilist=[i[0] for i in path]
	jlist=[i[1] for i in path]
	plt.plot(jlist,ilist)
	plt.scatter(jlist,ilist)
	plt.savefig(root+"include/figures/path.png")

@CFUNCTYPE(c_void_p,c_int)
def plot_allcontours_with_path(total_timesteps):
	plot_allcontours(total_timesteps)
	#print("kkk")
	plot_contours_with_path(total_timesteps)

idim=jdim=100
cfunc=CFUNCTYPE(c_void_p,c_int,POINTER(POINTER(c_int)),POINTER(POINTER(c_double)),POINTER(POINTER(c_double)),POINTER(POINTER(c_double)))
@cfunc
def assign(timestep,nb,F,u,v):
	#print("assigning")
	if(timestep>0):
		plot_contour(timestep-1)
	#print("assigning")
	for i in range(idim):
		for j in range(jdim):
			F[i][j]=1
			u[i][j]=0
			v[i][j]=0
	return
cfunc2=CFUNCTYPE(c_void_p,c_int,POINTER(c_double),POINTER(c_double),POINTER(c_double),c_int,c_int)
@cfunc2
def assign_bvels(timestep,F,u,v,i,j):
	F[0]=F[1]=F[2]=F[3]=1
	u[0]=u[1]=u[2]=u[3]=0
	v[0]=v[1]=v[2]=v[3]=0
	return

lib=CDLL("/home/revanth/Mtech/projct/c++/openmp/include/libgrid.so.1")


# lib.data_preparation.restype=c_void_p
#lib.data_preparation.argtypes=[c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_float,c_float,c_int,c_int,c_int,cfunc]
print("aaa")
#plot_allcontours_with_path(958)
print(lib.data_preparation(idim,jdim,assign,assign_bvels,plot_allcontours_with_path,20,20,90,90,1000,5,c_float(.5),c_float(.1),c_float(1),c_float(2),1,c_void_p(0),0))
# print("ddd")
# print(lib.data_preparation(idim,jdim,assign,20,20,90,90,1000,5,c_float(.5),c_float(.1),1,1,4))
