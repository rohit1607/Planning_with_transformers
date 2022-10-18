from math import ceil,floor,atan,pi
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class Narrowband:

    def __init__(self,dimensions,nb_width,dz_per,noentry_slices):
        
        self._dims=dimensions
        self._nb_width=nb_width
        self._dz_dist=(1-dz_per)*nb_width
        self._narrowband=[]
        self._danger_zone=[]
        self._phi=float("inf")*np.ones(dimensions)
        self._round_off=lambda t:(round(t[0]),round(t[1]))
        self._islices=noentry_slices[0]
        self._jslices=noentry_slices[1]

    def _ray_intersect_edge(self,point,edge,threshold=1e-8):
        i,j=point
        (ia,ja),(ib,jb)=edge
        #if edge is  almost  vertical
        if(ia>ib):
            ia,ib=ib,ia
            ja,jb=jb,ja
        if(i==ia ):
            i=ia+threshold*10
        if(i==ib):
            i=ib+threshold*10
        intersect=False

        if(i>ib or i<ia) or (j>max(ja,jb)):
            return False
        if(j<min(ja,jb)):
            intersect=True
        else:
            if(abs(ja-jb)) >threshold:
                m_edge=(ib-ia)/float(jb-ja)
            else:
                m_edge=(1/threshold)
            if(abs(ja-j))>threshold:
                m_point=(i-ia)/float(j-ja)
            else:
                m_point=(1/threshold)
            intersect=(m_point>=m_edge)
        return intersect

    def _signed_dist(self,ray_hit_counter):
        signed_dist=float("inf")*np.ones(self._dims)
        for i,j in self._narrowband:

            # if(self._phi[i][j]>self._dz_dist):
            #     self._danger_zone.append((i,j))
            
            signed_dist[i][j]=self._phi[i][j]*(1-2*(ray_hit_counter[i][j]%2))
            if(abs(self._phi[i][j])==1e-5):
                signed_dist[i][j]=1e-5
        #self._danger_zone=np.asarray(self._danger_zone)
        self._phi=signed_dist
        if(self._islices!=[]):
            for islice,jslice in zip(self._islices,self._jslices):
                self._phi[islice,jslice]=float("inf")

    def _narrow_band_construction(self,zero_contours):
        self._phi=float("inf")*np.ones(self._dims)
        self._narrowband=[]
        self._danger_zone=[]
        idim,jdim=self._dims
        nondanger_zone=set()
        self._zero_contours=zero_contours
        samelevel_pts=defaultdict(set)
        for zero_contour in self._zero_contours:
            if (abs(zero_contour[0][0]-zero_contour[-1][0])<1e-3 and abs(zero_contour[0][1]-zero_contour[-1][1])<1e-3):
                length=len(zero_contour)-1
            else:
                length=len(zero_contour)
            #print("leng",length)
            if(length>1):
                for ind in range(length):
                    i,j=zero_contour[ind]
                    imin=max(0,int(ceil(i)-self._nb_width))
                    imax=min(idim,int(floor(i)+self._nb_width+1))

                    for ia in range(imin,imax):
                        
                        jdel=floor(((self._nb_width**2)-((ia-i)**2))**.5)
                        jmin=max(0,int(round(j)-jdel))
                        jmax=min(jdim,int(round(j)+jdel+1))
                        for ja in range(jmin,jmax):
                            samelevel_pts[ia].add(ja) 
                            if(((i-ia)**2+(j-ja)**2)**.5 <self._dz_dist):
                                nondanger_zone.add((ia,ja))

                            
                            dist=((self._dX[1]*(i-ia))**2+(self._dX[0]*(j-ja))**2)**.5
                            self._phi[ia][ja]=min(abs(self._phi[ia][ja]),dist)
                            
                            if(abs(self._phi[ia][ja])==0):
                                self._phi[ia][ja]=1e-5

        for i,jlist in samelevel_pts.items():
            self._narrowband.extend([(i,j) for j in jlist])
        self._danger_zone=np.asarray(list(set(self._narrowband)-nondanger_zone))
        self._narrowband=np.asarray(self._narrowband)
        ray_hit_counters=np.zeros(self._dims)
    #calculating signed component of the distance
    #calculate no of times a ray originating from point hits a contour
    
        for zero_contour in self._zero_contours:
            

            if (abs(zero_contour[0][0]-zero_contour[-1][0])<1e-3 and abs(zero_contour[0][1]-zero_contour[-1][1])<1e-3):
                length=len(zero_contour)-1
            else:
                length=len(zero_contour)
            #print("leng",length)
            if (length >1):
                for ind in range(length):
                    ind_next=(ind+1)%(len(zero_contour))
                    i1,j1=zero_contour[ind]
                    i2,j2=zero_contour[ind_next]
                    edge=((i1,j1),(i2,j2))
                    imin,imax=min(i1,i2),max(i1,i2)
                    #print("edges",edge)
                    for i in range(int(ceil(imin)),int(floor(imax+1))):
                        for j in list(samelevel_pts[i]):
                            ray_hit_counters[i][j]+=self._ray_intersect_edge((i,j),edge)
                            
        self._signed_dist(ray_hit_counters)

