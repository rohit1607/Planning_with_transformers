import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,Point
import numpy as np
from Grid_generation import Grid_generation
folder="./gshhg-shp-2.3.7/GSHHS_shp/f/"
lands=gpd.read_file(folder+"GSHHS_f_L1.shp")
# lands.plot()
# plt.show()
latsrange=(4,14)
lonsrange=(70,90)



res=0.04
_,_,Grid=Grid_generation(lonsrange,latsrange,res)
Grid=np.asarray(Grid)

print(Grid.shape)

grid_1=gpd.GeoSeries([Polygon([[lonsrange[0],latsrange[0]],[lonsrange[0],latsrange[1]],[lonsrange[1],latsrange[1]],[lonsrange[1],latsrange[0]]])])
grid_1=gpd.GeoDataFrame({"geometry":grid_1})

grid_1=gpd.overlay(grid_1,lands,how="intersection")
grid_1.plot()
plt.show()



land=[None]*(Grid.shape[0]*Grid.shape[1])

def plot_land(i,p):
    if(i%Grid.shape[0]==0):
        print(int(i/(Grid.shape[0])))
    p=Point(p[1],p[0])
    c=grid_1.contains(p).sum()
    #print(p[0],c)
    if(not c==0):
        land[i]=True
    else: 
        land[i]=False
        
from joblib import Parallel, delayed,parallel_backend
Parallel(n_jobs=8, require='sharedmem')(delayed(plot_land)(i,point) for i,point in enumerate(list(Grid.reshape(-1,2))))

land=np.asarray(land).reshape((Grid.shape[0],Grid.shape[1]))
np.save("0/land_{}".format(res),land)
