from calendar import monthrange
from datetime import datetime,timedelta
#
#import pandas as pd
import numpy as np

import os
import time
import pickle
from ctypes import *

from joblib import Parallel, delayed
import sklearn
import pickle
from math import sin,cos,pi,floor,ceil
from Grid_generation import Grid_generation

#destination
vladivostok=(43.1198,131.8869)

#source
chennai=(13.0827, 80.2707)

#latitudes and longitudes range
latsrange=(4,14)
lonsrange=(70,90)


res=0.04
lat_len,lon_len,Grid=Grid_generation(lonsrange,latsrange,res)
Grid=np.asarray(Grid)

print(lat_len,lon_len)
folder="0/"
#load the binary map of the geograpgy of the total grid
land=np.load(folder+"land_0.04.npy")
land2=np.empty(land.shape)
#mask the islands and land masses borders to eliminate non rectungular corners
for i in range(land.shape[0]):
    for j in range(land.shape[1]):
        if(i> 0 and j>0 and i<land.shape[0]-1 and j<land.shape[1]-1):
            land2[i][j]=(((land[i-1][j]+land[i+1][j]+land[i][j-1]+land[i][j+1]+land[i-1][j+1]+land[i-1][j-1]+land[i+1][j-1]+land[i+1][j+1])/8)+land[i][j])/2
        else:
            land2[i][j]=land[i][j]
land_ind=np.asarray(np.where(land2>0))


grid_res=(len(Grid),len(Grid[0]))

minlon,maxlon=min(Grid[-1][0][1],Grid[0][0][1]),max(Grid[-1][-1][1],Grid[0][-1][1])
lons=(minlon,maxlon)
lats=(latsrange[0],latsrange[1])


#load the environment files
heightall=np.load(folder+"height.npy").tolist()
winds_u=np.load(folder+"winds_u.npy").tolist()
winds_v=np.load(folder+"winds_v.npy").tolist()
windlats=np.load(folder+"windlats.npy").tolist()
windlons=np.load(folder+"windlons.npy").tolist()
windlatsmap={item:i for i,item in enumerate(windlats)}
windlonsmap={item:i for i,item in enumerate(windlons)}







waterlats=np.load(folder+"water/lats.npy")
waterlats=[round(i,2) for i in waterlats]
waterlatsmap={item:i for i,item in enumerate(waterlats)}



waterlons=np.load(folder+"water/lons_0.npy")
waterlons=[round(i,2) for i in waterlons]
waterlonsmap={item:i for i,item in enumerate(waterlons)}
turl="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z.dods?time"
time_list=list(np.load(folder+"time_list.npy"))
timesmap={t:i for i,t in enumerate(time_list)}


from scipy.interpolate import LinearNDInterpolator


#resolution of the time 
time_res_mins=4

#load the c++ interpolation library
lib=CDLL("./interpolation/libinter.so.1")

#load the Trained ML model to output the ship speed
with open(folder+"model.m","rb") as file:
    model=pickle.load(file)

#times range which encompasses the journey
timesrange=["2018-01-02 00:00:00","2018-01-16 00:00:00"]
ref_time="2000-01-01 00:00:00"
total_iters=(datetime.strptime(timesrange[1],"%Y-%m-%d %H:%M:%S")-datetime.strptime(timesrange[0],"%Y-%m-%d %H:%M:%S")).total_seconds()/(60*time_res_mins)

initial_offset=(datetime.strptime(timesrange[0],"%Y-%m-%d %H:%M:%S")-datetime.strptime(ref_time,"%Y-%m-%d %H:%M:%S")).total_seconds()/3600
#offset of the journey date from the initial time of downloaded data set
dwnld_offset=initial_offset-24

def double_array(M,N,types=c_double):
	double_P = POINTER(types)
	inner_array = (types * N)
	inner_array_P = POINTER(inner_array)
	dest = (double_P * M) ()
	for i in range(M):
		dest[i] = inner_array()
	return dest
def triple_array(P,M,N):
	double_P_P=POINTER(POINTER(c_double))
	dest=(double_P_P*P)()
	for i in range(P):
		dest[i]=double_array(M,N)
	return dest
def copy_dlist(li,types=c_double):
	M,N=len(li),len(li[0])
	dest=double_array(M,N,types)
	for i in range(M):
		for j in range(N):
			if(types==c_int):
				dest[i][j]=int(li[i][j])
			else:
				dest[i][j]=li[i][j]
	return dest
def copy_tlist(li):
	P,M,N=len(li),len(li[0]),len(li[0][0])
	dest=triple_array(P,M,N)
	for i in range(P):
		for j in range(M):
			for k in range(N):
				dest[i][j][k]=li[i][j][k]
	return dest

#copy the python variables to c types for faster processing
c_heightall=copy_tlist(heightall)
c_winds_u=copy_tlist(winds_u)
c_winds_v=copy_tlist(winds_v)
c_Grid=copy_tlist(Grid)
c_wilatsmap=copy_dlist([[item,i] for i,item in enumerate(windlats)])
c_wilonsmap=copy_dlist([[item,i] for i,item in enumerate(windlons)])

# returns the prevoius and next variable value in the grid 
def prev_nxt(x,resolution):
    if(x>0):
        return (round(int(x/resolution)*resolution,2),round((int(x/resolution)+1)*resolution,2))
    else:
        return (round((int(x/resolution)-1)*resolution,2),round((int(x/resolution))*resolution,2))

#@numba.jit

# def trilinear_interpolation(points,values,outpoint):
#     x,y,z=outpoint
#     x0,x1=points[0]
#     y0,y1=points[1]
#     z0,z1=points[2]
#     c000,c001,c010,c011,c100,c101,c110,c111=values
#     denom=(x1-x0)*(y1-y0)*(z1-z0)
#     a0=(c000*x1*y1*z1-c001*x1*y1*z0-c010*x1*y0*z1+c011*x1*y0*z0-c100*x0*y1*z1+c101*x0*y1*z0+c110*x0*y0*z1-c111*x0*y0*z0)/denom
#     a1=(-c000*y1*z1+c001*y1*z0+c010*y0*z1-c011*y0*z0+c100*y1*z1-c101*y1*z0-c110*y0*z1+c111*y0*z0)/denom
#     a2=(-c000*x1*z1+c001*x1*z0+c010*x1*z1-c011*x1*z0+c100*x0*z1-c101*x0*z0-c110*x0*z1+c111*x0*z0)/denom
#     a3=(-c000*x1*y1+c001*x1*y1+c010*x1*y0-c011*x1*y0+c100*x0*y1-c101*x0*y1-c110*x0*y0+c111*x0*y0)/denom
#     a4=(c000*z1-c001*z0-c010*z1+c011*z0-c100*z1+c101*z0+c110*z1-c111*z0)/denom
#     a5=(c000*y1-c001*y1-c010*y0+c011*y1-c100*y1+c101*y1+c110*y0-c111*y0)/denom
#     a6=(c000*x1-c001*x1-c010*x1+c011*x0-c100*x0+c101*x0+c110*x0-c111*x0)/denom
#     a7=(-c000+c001+c010-c011+c100-c101-c110+c111)/denom
#     return a0+a1*x+a2*y+a3*z+a4*x*y+a5*x*z+a6*y*z+a7*x*y*z

# def interp_3d(points,values_list,outpoint):
#     output=[]
#     x,y,z=outpoint
#     x0,x1=points[0]
#     y0,y1=points[1]
#     z0,z1=points[2]
#     #
#     for i in range(len(values_list[0])):
#         output.append(trilinear_interpolation(points,[val[i] for val in values_list],outpoint))

#         c000,c001,c010,c011,c100,c101,c110,c111=[val[i] for val in values_list]
#         output.append(lib.trilinear_interpolation(c_double(x),c_double(y),c_double(z),c_double(x0),
# 	   c_double(x1),c_double(y0),c_double(y1),c_double(z0),c_double(z1),
# 	   c_double(c000),c_double(c001),c_double(c010),c_double(c011),c_double(c100),
# 	   c_double(c101),c_double(c110),c_double(c111)))
#     return output


#function used to create grid of speeds F,u,v at each time step
def create_gridfile2(t,nb_pts):
	#nb=np.asarray(nb)
	s=time.time()
	
	
	Wa_u=np.empty((Grid.shape[0],Grid.shape[1]))
	Wa_v=np.empty((Grid.shape[0],Grid.shape[1]))
	nb_pts=nb_pts.astype(np.int32)
	c_narrowband=nb_pts.ctypes.data_as(POINTER(c_int))


	idate=datetime.strptime(timesrange[0],"%Y-%m-%d %H:%M:%S")+timedelta(minutes=time_res_mins*t)
	imonth=idate.month
	iyear=idate.year
	ihrs_from_month_beginning=(idate-datetime.strptime("{}-{}-01 00:00:00".format(iyear,imonth),"%Y-%m-%d %H:%M:%S")).total_seconds()/3600
	#print(imonth,idate,ihrs_from_month_beginning)
	ihrs=initial_offset+(t*time_res_mins/60)
	prev_t,next_t=prev_nxt(ihrs,3)
	
	#get the index of the prev timestp and next timestep
	try:
		prev_ind=timesmap[prev_t]
		next_ind=timesmap[next_t]
	except:
		close_val=min(timesmap.keys(), key=lambda x:abs(x-ihrs))
		if(close_val<ihrs):
			prev_ind=timesmap[close_val]
			next_ind=prev_ind+1
			prev_t=close_val
			next_t=time_list[next_ind]
		else:
			next_ind=timesmap[close_val]
			prev_ind=next_ind-1
			prev_t=time_list[prev_ind]
			next_t=close_val

	#load the next and prev water uv files and copy to ctype files
	prev_wateruv_file=np.load(folder+"water/{}_0.npy".format(prev_ind))
	next_wateruv_file=np.load(folder+"water/{}_0.npy".format(next_ind))
	prev_wateruv=np.asarray([prev_wateruv_file[0][0],prev_wateruv_file[1][0]])
	prev_wateruv=prev_wateruv.astype(np.float64)

	c_double_p=POINTER(c_double)
	c_prevuv = prev_wateruv.ctypes.data_as(c_double_p)

	next_wateruv=np.asarray([next_wateruv_file[0][0],next_wateruv_file[1][0]])
	next_wateruv=next_wateruv.astype(np.float64)
	c_nextuv=next_wateruv.ctypes.data_as(c_double_p)

	#ctype of model input
	model_input=(c_double*5*len(nb_pts))()
	lib.interpolation(c_narrowband,c_int(len(nb_pts)),c_wilatsmap,
			c_int(len(windlats)),c_wilonsmap,c_int(len(windlons)),
				 c_int(int(dwnld_offset)),c_double(ihrs),c_double(prev_t),c_double(next_t),
				 c_Grid,c_heightall,c_winds_u,c_winds_v,c_prevuv,c_nextuv,
				  c_int(next_wateruv.shape[0]),c_int(next_wateruv.shape[1]),c_int(next_wateruv.shape[2]),model_input)

	model_input=np.copy(model_input,"C")
	
	#run the ML model to get output
	output=model.predict(model_input)
	
	F=np.zeros((Grid.shape[0],Grid.shape[1]))
	
	for k,(i,j) in enumerate(nb_pts):
		F[i][j]=output[k]
		Wa_u[i][j]=model_input[k][3]
		Wa_v[i][j]=model_input[k][4]
	

	F[land_ind[0],land_ind[1]]=0
	Wa_u[land_ind[0],land_ind[1]]=0
	Wa_v[land_ind[0],land_ind[1]]=0

	return F,Wa_u,Wa_v

def create_files(t,i,j):
    H=[]
    Wa_u=[]
    Wa_v=[]
    Wi_u=[]
    Wi_v=[]

    idate=datetime.strptime(timesrange[0],"%Y-%m-%d %H:%M:%S")+timedelta(minutes=time_res_mins*t)
    imonth=idate.month
    iyear=idate.year
    ihrs_from_month_beginning=(idate-datetime.strptime("{}-{}-01 00:00:00".format(iyear,imonth),"%Y-%m-%d %H:%M:%S")).total_seconds()/3600
    ihrs=initial_offset+(t*time_res_mins/60)
    prev_t,next_t=prev_nxt(ihrs,3)
    try:
        prev_ind=timesmap[prev_t]
        next_ind=timesmap[next_t]
    except:
        close_val=min(timesmap.keys(), key=lambda x:abs(x-ihrs))
        if(close_val<ihrs):
            prev_ind=timesmap[close_val]
            next_ind=prev_ind+1
            prev_t=close_val
            next_t=time_list[next_ind]
        else:
            next_ind=timesmap[close_val]
            prev_ind=next_ind-1
            prev_t=time_list[prev_ind]
            next_t=close_val
    prev_wateruv_file=np.load(folder+"water/{}_0.npy".format(prev_ind))
    next_wateruv_file=np.load(folder+"water/{}_0.npy".format(next_ind))
    nb_pts=[(i,j),(i,j+1),(i+1,j+1),(i+1,j)]

#function used in backtracking to estimate speeds at a point i,j
    def each_interp(i,j):
        
        
        ilat,ilon=Grid[i][j]
        
        try:
            variables=[]
            fdata=[]
            for hr in prev_nxt(ihrs,3):
                for la in prev_nxt(ilat,0.5):
                    for lo in prev_nxt(ilon,0.5):
                        hr_ind=int((hr-dwnld_offset)/3)
                        la_ind=windlatsmap[la]
                        lo_ind=windlonsmap[lo]

                        variables.append((hr,la,lo))
                        fdata.append((heightall[hr_ind][la_ind][lo_ind],winds_u[hr_ind][la_ind][lo_ind],winds_v[hr_ind][la_ind][lo_ind]))
            interpolator=LinearNDInterpolator(variables,fdata,rescale=True)
            #print(interpolator(ihrs,ilat,ilon))
            h,wi_u,wi_v=interpolator(ihrs,ilat,ilon)

        except Exception as e:
      #      print(e)
            h,wi_u,wi_v=[0,0,0]

        try:
            variables=[]
            fdata=[]
            for hr in [prev_t,next_t]:
                for la in prev_nxt(ilat,0.08):
                    for lo in prev_nxt(ilon,0.08):
                        if(hr==prev_t):
                            uv_file=prev_wateruv_file
                        else:
                            uv_file=next_wateruv_file

                        la_ind=waterlatsmap[la]
                        lo_ind=waterlonsmap[lo]

                        variables.append((hr,la,lo))
                        fdata.append((uv_file[0][0][la_ind][lo_ind],uv_file[1][0][la_ind][lo_ind]))
            interpolator=LinearNDInterpolator(variables,fdata,rescale=True)
            wa_u,wa_v=interpolator(ihrs,ilat,ilon)
            wa_u=0.001*wa_u
            wa_v=0.001*wa_v
        except Exception as e :
       
            wa_u,wa_v=[0,0]


        H.append(h)
        Wa_v.append(wa_v)
        Wa_u.append(wa_u)
        Wi_u.append(wi_u)
        Wi_v.append(wi_v)

    
    for (i,j) in nb_pts:
        each_interp(i,j)
    
    model_input=[]
    for k,(i,j) in enumerate(nb_pts):
        
        model_input.append([H[k],Wi_u[k],Wi_v[k],Wa_u[k],Wa_v[k]])

    
    output=model.predict(model_input)
    F=[]
    
    for k,(i,j) in enumerate(nb_pts):
        F.append(output[k])

    return F,Wa_u,Wa_v



