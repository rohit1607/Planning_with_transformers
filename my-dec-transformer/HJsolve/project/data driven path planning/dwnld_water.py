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
folder="maersk data/"
file="ship_sample_full.csv"

####replace this with your code
# data=pd.read_csv(folder+file)
# imo_list=set(data["imo"])
# data["lon"]=data["longitude"]%360
# latsrange={}
# lonsrange={}
# for imo in set(data["imo"]):
#     lats=data[data["imo"]==imo]["latitude"]
#     maxlat,minlat=max(lats),min(lats)
#     latsrange[imo]=(minlat,maxlat)
    
#     lons=data[data["imo"]==imo]["longitude"]
#     maxlon,minlon=max(lons)%360,min(lons)%360
#     lonsrange[imo]=(minlon,maxlon)
########################replace#############

class Wave_Speed:
    def __calc_maxt_idx(self):
        return ((datetime.now()-datetime.strptime(self.__start_date,"%Y-%m-%d %H:%M:%S")).days+5)*8 -2726
    
    def __map_lats(self,latrange):
        startlat,endlat=latrange
        #startlat_idx=((int(startlat*25)/25)-self.__minlat)*25
        #endlat_idx=(((int(endlat*25)+1)/25)-self.__minlat)*25
        startidx=max(np.where(self._latsmap<=startlat)[0])
        endidx=min(np.where(self._latsmap>=endlat)[0])
        return int(startidx),int(endidx)
    
    def __map_lons(self,lonrange):
        startlon,endlon=lonrange
        
        startlon_idx=((int(startlon*12.5)/12.5)-self.__minlon)*12.5
        endlon_idx=(((int(endlon*12.5)+1)/12.5)-self.__minlon)*12.5
        maxlon_idx=int((self.__maxlon-self.__minlon)*12.5)
        if(startlon<endlon):
            return int(startlon_idx),int(endlon_idx)
        else:
            return (int(startlon_idx),maxlon_idx),(0,int(endlon_idx))
    
    def _map_times(self,timerange):
        starttime,endtime=timerange
        startdel=int(((datetime.strptime(starttime,"%Y-%m-%d %H:%M:%S")-datetime.strptime(self.__start_date,"%Y-%m-%d %H:%M:%S")).total_seconds())/3600)
        enddel=int(((datetime.strptime(endtime,"%Y-%m-%d %H:%M:%S")-datetime.strptime(self.__start_date,"%Y-%m-%d %H:%M:%S")).total_seconds()+1)/3600)
        startidx=max(np.where(self._timesmap<=startdel)[0])
        endidx=min(np.where(self._timesmap>=enddel)[0])
        return startidx,endidx
    
        
    def __init__(self,):
        self.__base_url="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z.dods?"
        self.__start_date="2000-01-01 00:00:00"
        self.__mindepth=self.__maxdepth=0
        self.__minlat=-80
        self.__minlon=0
        self.__maxlon=359.92
        url=self.__base_url+"time"
        laturl=self.__base_url+"lat"
        print(url)
        self._timesmap=np.asarray(open_dods(url)["time"])
        
        self._latsmap=np.around(np.asarray(open_dods(laturl)["lat"]),2)
        #print(self._latsmap)
        
    def wave_speeds(self,latrange,lonrange,timerange):
        u_name="water_u"
        v_name="water_v"
        minlat,maxlat=self.__map_lats(latrange)
        minlon,maxlon=self.__map_lons(lonrange)
        startidx,endidx=self._map_times(timerange)
        
        time_string="[{}:1:{}]".format(startidx,endidx)
        depth_string="[{}:1:{}]".format(self.__mindepth,self.__maxdepth)
        lat_string="[{}:1:{}]".format(minlat,maxlat)
        if(isinstance(minlon,int)):
            lon_string="[{}:1:{}]".format(minlon,maxlon)
        else:
            lon_string=("[{}:1:{}]".format(minlon[0],minlon[1]),"[{}:1:{}]".format(maxlon[0],maxlon[1]))
        if(isinstance(minlon,int)):
            u_string=u_name+time_string+depth_string+lat_string+lon_string
            v_string=v_name+time_string+depth_string+lat_string+lon_string
            url=self.__base_url+u_string+","+v_string
            req=open_dods(url,timeout=5000)
            u=req[u_name]
            v=req[v_name]
            return u,v
        else:
            u_string1=u_name+time_string+depth_string+lat_string+lon_string[0]
            v_string1=v_name+time_string+depth_string+lat_string+lon_string[0]
            u_string2=u_name+time_string+depth_string+lat_string+lon_string[1]
            v_string2=v_name+time_string+depth_string+lat_string+lon_string[1]
            
            url1=self.__base_url+u_string1+","+v_string1
            url2=self.__base_url+u_string2+","+v_string2
            req1=open_dods(url1,timeout=5000)
            req2=open_dods(url2,timeout=5000)
            u1=req1[u_name]
            v1=req1[v_name]
            u2=req2[u_name]
            v2=req2[v_name]
            u=(u1,u2)
            v=(v1,v2)
            return u,v
    def wave_speeds2(self,latrange,lonrange,timerange,foldername):
        df=pd.DataFrame(columns=["date_time","lat","lon","water_u","water_v"])
        u_name="water_u"
        v_name="water_v"
        minlat,maxlat=self.__map_lats(latrange)
        minlon,maxlon=self.__map_lons(lonrange)
        startidx,endidx=self._map_times(timerange)
        print(startidx,endidx)
        start_date=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S")
        depth_string="[{}:1:{}]".format(self.__mindepth,self.__maxdepth)
        lat_string="[{}:1:{}]".format(minlat,maxlat)
        if(isinstance(minlon,int)):
            lon_string="[{}:1:{}]".format(minlon,maxlon)
        else:
            lon_string=("[{}:1:{}]".format(minlon[0],minlon[1]),"[{}:1:{}]".format(maxlon[0],maxlon[1]))
        if(isinstance(minlon,int)):
            lon_strings=[lon_string]
        else:
            lon_strings=[lon_string[0],lon_string[1]]
        
        for lonid,lon_string in enumerate(lon_strings):
            for idx in range(startidx,endidx+1):
                time_string="[{}:1:{}]".format(idx,idx)
                if(not os.path.exists(foldername+"/"+str(idx)+"_"+str(lonid)+".npy")):
                    u_string=u_name+time_string+depth_string+lat_string+lon_string
                    v_string=v_name+time_string+depth_string+lat_string+lon_string
                    url=self.__base_url+u_string+","+v_string
                    req=open_dods(url,timeout=50000)
                    u=req[u_name]
                    v=req[v_name]
                    times=int(np.asarray(u["time"])[0])
                    lats=u["lat"]
                    lons=u["lon"]
                    uv=[u["water_u"][0],v["water_v"][0]]
                    if(idx==startidx):
                        if(lonid==0):
                            np.save(foldername+"/"+"lats",np.asarray(lats))
                        np.save(foldername+"/"+"lons_{}".format(lonid),np.asarray(lons))
                    np.save(foldername+"/"+str(idx)+"_"+str(lonid),np.asarray(uv))
        return True

ws=Wave_Speed()
starttime,endtime=["2018-01-01 12:00:00","2018-01-05 11:59:59"]
start=datetime.strptime(starttime,"%Y-%m-%d %H:%M:%S")
end=datetime.strptime(endtime,"%Y-%m-%d %H:%M:%S")
timerange=(starttime,endtime)
imo_list=[0]
latsrange={0:(4,14)}
lonsrange={0:(70,90)}
for imo in list(imo_list):
    print(imo)
    if(not os.path.exists(str(imo))):
        os.mkdir(str(imo))
    if(not os.path.exists(str(imo)+"/water")):
        os.mkdir(str(imo)+"/water")
    latrange,lonrange =latsrange[imo],lonsrange[imo]
    finished=False
    while(not finished):
	    try:
	        finished=ws.wave_speeds2(latrange,lonrange,timerange,"{}/water".format(imo))
	    except Exception as e:
	        print(e)
	        time.sleep(30)
        
    print("saved")

