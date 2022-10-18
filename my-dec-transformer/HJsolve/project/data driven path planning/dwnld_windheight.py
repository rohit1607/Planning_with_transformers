from calendar import monthrange
from datetime import datetime,timedelta
from pydap.client import open_dods,open_url
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle as pb
from sys import getsizeof
import os
import time

class Wind_Height:
    
    def __map_lats(self,latrange):
        startlat,endlat=latrange
        
        startlat_idx=(self.__maxlat-(int(startlat*2)/2))*2
        endlat_idx=(self.__maxlat-((int(endlat*2)+1)/2))*2
        return int(endlat_idx),int(startlat_idx)
    
    def __map_lons(self,lonrange):
        startlon,endlon=lonrange
        startlon_idx=((int(startlon*2)/2)-self.__minlon)*2
        endlon_idx=(((int(endlon*2)+1)/2)-self.__minlon)*2
        if(startlon<endlon):
            return int(startlon_idx),int(endlon_idx)
        else:
            maxlon_idx=(self.__maxlon-self.__minlon)*2
            
            return (int(startlon_idx),int(maxlon_idx)),(0,int(endlon_idx))
    
    def __map_times(self,timerange):
        starttime,endtime=timerange
        start=datetime.strptime(starttime,"%Y-%m-%d %H:%M:%S")
        end=datetime.strptime(endtime,"%Y-%m-%d %H:%M:%S")
        years_months=[datetime.strftime(start,"%m-%Y")]+pd.date_range(starttime,endtime, 
              freq='MS').strftime("%m-%Y").tolist()

        
        return sorted(list(set(years_months)))
        
    def __init__(self,latrange,lonrange,timerange):
        self.__minlat=-77.5
        self.__maxlat=77.5
        
        self.__maxlon=359.5
        self.__minlon=0
        
        self.__years_months=self.__map_times(timerange)
        lat1,lat2=self.__map_lats(latrange)
        self.__lat_str="[{0}:1:{1}]".format(lat1,lat2)
        self.__lon1,self.__lon2=self.__map_lons(lonrange)
        if(isinstance(self.__lon1,int)):
            self.__lon_str="[{0}:1:{1}]".format(self.__lon1,self.__lon2)
        else:
            self.__lon_str=("[{0}:1:{1}]".format(self.__lon1[0],self.__lon1[1]),"[{0}:1:{1}]".format(self.__lon2[0],self.__lon2[1]))
        
        self.__time_index_str=lambda days_mnth:"[{0}:1:{1}]".format(0,days_mnth*8)
        if(isinstance(self.__lon1,int)):
            self.__varstr=lambda days_mnth:self.__time_index_str(days_mnth)+self.__lat_str+self.__lon_str
        else:
            self.__varstr=(lambda days_mnth:self.__time_index_str(days_mnth)+self.__lat_str+self.__lon_str[0],lambda days_mnth:self.__time_index_str(days_mnth)+self.__lat_str+self.__lon_str[1])
        self.__urlfunc=lambda var,monNee Kannu Neeli Samudramth,year:"https://data.nodc.noaa.gov/thredds/dodsC/ncep/nww3/"+str(year)+"/"+str(month).zfill(2)+"/gribs/multi_1.glo_30m."+var+"."+str(year)+str(month).zfill(2)+".grb2.dods?"
    
    def windspeed(self):
        varu="u-component_of_wind_surface"
        varv="v-component_of_wind_surface"
        uVel=[]
        vVel=[]
        for month_year_str in self.__years_months:
            month,year=month_year_str.split("-")
            days=monthrange(int(year),int(month))[1]
            base_url=self.__urlfunc("wind",month,year)
            if(isinstance(self.__lon1,int)):
                url=base_url+varu+self.__varstr(days)+","+varv+self.__varstr(days)
                print(url)
                request=open_dods(url)
                #print(month,year,time.time()-t)
                #u_vel=request[varu][varu]__calc_maxt_idx__calc_maxt_idx__calc_maxt_idx__calc_maxt_idx
                #v_vel=request[varv][varv]
                u_vel=request[varu]
                v_vel=request[varv]
                uVel.append(u_vel)

                vVel.append(v_vel)
            else:
                url1=base_url+varu+self.__varstr[0](days)+","+varv+self.__varstr[0](days)
                url2=base_url+varu+self.__varstr[1](days)+","+varv+self.__varstr[1](days)
            
                request1=open_dods(url1)
                request2=open_dods(url2)
                
                u_vel1=request1[varu]
                v_vel1=request1[varv]
                u_vel2=request2[varu]
                v_vlatsrangeel2=request2[varv]
                uVel.append((u_vel1,u_vel2))
                vVel.append((v_vel1,v_vel2))
        return uVel,vVel
    
    def waveheight(self):
        var_height="Significant_height_of_combined_wind_waves_and_swell_surface"
        waveheights=[]
        for month_year_str in self.__years_months:
            
            month,year=month_year_str.split("-")
            days=monthrange(int(year),int(month))[1]
            
            base_url=self.__urlfunc("hs",month,year)
            if(isinstance(self.__lon1,int)):
                url=base_url+var_height+self.__varstr(days)
                #print(url)
                print(month_year_str,url)
                request=open_dods(url)

                wave_height=request[var_height]
                waveheights.append(wave_height)
            else:
                url1=base_url+var_height+self.__varstr[0](days)
                url2=base_url+var_height+self.__varstr[1](days)
                #print(url1,url2)
                request1=open_dods(url1)

                request2=open_dods(url2)
                wave_height1=request1[var_height]
                wave_height2=request2[var_height]
                waveheights.append((wave_height1,wave_height2))
        return waveheights


lats=(4,14)
lons=(70,90)
times=["2018-01-01 00:00:00","2018-01-05 11:59:59"]
wh=Wind_Height(lats,lons,times)
height=wh.waveheight()
winds=wh.windspeed()
folder="0/"
heightall=[]
#height_time=[]
# height_lat=[]
# height_lon=[]
for i in range(len(height)):
    for j in range(height[i]["Significant_height_of_combined_wind_waves_and_swell_surface"].shape[0]):
        heightall.append(np.asarray(height[i]["Significant_height_of_combined_wind_waves_and_swell_surface"][j]))
    #height_time.append(height[i]["time"])
#     height_lat.extend(height[i]["lat"])
#     height_lon.extend(height[i]["lon"])
heightall=np.asarray(heightall)
heightall[np.isnan(heightall)]=0
np.save(folder+"height",heightall)

windlats=list(height[0]["lat"])
windlons=list(height[0]["lon"])

np.save(folder+"windlats",np.asarray(windlats))
np.save(folder+"windlons",np.asarray(windlons))


windsu=[]
# winds_time=[]
# winds_lat=[]
# winds_lon=[]
for i in range(len(winds[0])):
    for j in range(winds[0][i]["u-component_of_wind_surface"].shape[0]):
        windsu.append(np.asarray(winds[0][i]["u-component_of_wind_surface"][j]))
    # winds_time.append(winds[0][i]["time"])
    # winds_lat.extend(winds[0][i]["lat"])
    # winds_lon.extend(winds[0][i]["lon"])
windsu=np.asarray(windsu)
windsu[np.isnan(windsu)]=0
np.save(folder+"winds_u",windsu)

windsv=[]
# winds_time=[]
# winds_lat=[]
# winds_lon=[]
for i in range(len(winds[1])):
    for j in range(winds[1][i]["v-component_of_wind_surface"].shape[0]):
        windsv.append(np.asarray(winds[1][i]["v-component_of_wind_surface"][j]))
    # winds_time.append(winds[1][i]["time"])
    # winds_lat.extend(winds[1][i]["lat"])
    # winds_lon.extend(winds[1][i]["lon"])
windsv=np.asarray(windsv)
windsv[np.isnan(windsv)]=0
np.save(folder+"winds_v",windsv)

