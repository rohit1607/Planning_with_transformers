from scipy.interpolate import LinearNDInterpolator
from HJsolve.project.python.narrowband import *
from HJsolve.project.python.scheme import *

############        #####################################################################



################################################################           ###############################
class Grid(Narrowband,Scheme):

	def __init__(self,dimensions,start,target,F,u,v,nb_width,dz_per,dX,dt,order=2,advection_term=9,fluxlimiter=0,tol=1e-8,noentry_slices=[[],[]]):
	
		Scheme.__init__(self,dimensions,F,u,v,dX,dt,order,advection_term,fluxlimiter,noentry_slices)
		Narrowband.__init__(self,dimensions,nb_width,dz_per,noentry_slices)
		self._start=start
		self._target=target
		
		self._X=np.arange(dimensions[1])
		self._Y=np.arange(dimensions[0])
		self._zerocontourslist=[self.__get_init_zc()]
		self._targetInNarrowband=False
		self._reached=False
		self._tol=tol
		self.__backtracked=False
		
	def __get_init_zc(self):
		#initialise the zero contour 
		Z=np.empty(shape=self._dims)
		for i in range(self._dims[0]):
			for j in range(self._dims[1]):
				# r=(((self._dX[1]*(i-self._start[0]))**2+(self._dX[0]*(j-self._start[1]))**2)**.5)-1*(self._dX[0]**2+self._dX[1]**2)**.5
				r=(((self._dX[1]*(i-self._start[0]))**2+(self._dX[0]*(j-self._start[1]))**2)**.5)-0.3*(self._dX[0]**2+self._dX[1]**2)**.5

				Z[i][j]=r
		self._phi=Z

		if(self._islices!=[]):
			for islice,jslice in zip(self._islices,self._jslices):
				self._phi[islice,jslice]=float("inf")

		return self.__contouring_function()

	def __contouring_function(self,precision=5):
		#extracts the zerocontour from the 2d profile
		contours=plt.contour(self._X,self._Y,self._phi,levels=[0])
		levels=contours._levels
		try:
			zero_index=np.argwhere(levels==0)[0][0]
			zero_contours=contours.allsegs[zero_index]
			zcs=[]
			for zero_contour in zero_contours:
				zero_contour=np.around(zero_contour,decimals=precision)
				zero_contour[:,[0,1]]=zero_contour[:,[1,0]]
				zcs.append(zero_contour)

		except IndexError:
			zcs=[]
		
		return zcs
	def __get_dangerzone_signlist(self):

		dangerzone_signlist=[]
		for point in self._danger_zone:
			x,y=point[0],point[1]
			dangerzone_signlist.append(np.sign(self._phi[x][y]))
		return dangerzone_signlist
	
	def __check_targetinNarrowband(self):
		for grid_point in self._narrowband:
			if(grid_point[0]==self._target[0] and grid_point[1]==self._target[1]):
				self._targetInNarrowband=True

	def _serial_scheme(self):
		phi_next=np.ones(self._dims)*float("inf")
		for i,j in self._narrowband:
			if(i< self._dims[0]-self._order-1 and i>=2 and j< self._dims[0]-self._order-1 and j>=2):
			#print(scheme(prev_phi,F,u,v,n,i,j,dX,dt,order,advection_term,fluxlimiter))
				Fval=self._F[self._timestep_count][i][j]
				phi_next[i][j]=self._scheme(Fval,i,j)
			else:
				phi_next[i][j]=self._phi[i][j]
		return phi_next

	def _serial_scheme2(self):
		#use this
		self._phi=self._level_set_half_step()
		self._phi=self._advection_half_step()
		self._phi=self._level_set_half_step()
		#self._phi=self._interpolate()
		return self._phi
	def _plot(self,zero_contours,folder="figures/"):

		nbX,nbY=self._narrowband[:,1],self._narrowband[:,0]
		dzX,dzY=self._danger_zone[:,1],self._danger_zone[:,0]
		#print(zero_contours[-1])
		#plt.figure()
		plt.scatter(nbX,nbY,color="b")
		#plt.scatter(dzX,dzY,color="r")

		inside_pts=np.argwhere(self._phi<0)
		outside_pts=np.argwhere(self._phi>0)
		#print(inside_pts)
		inx,iny=inside_pts[:,1],inside_pts[:,0]
		ox,oy=outside_pts[:,1],outside_pts[:,0]
		plt.scatter(inx,iny,color="g")
		plt.scatter(ox,oy,color="r")

		# for zc in self._zerocontourslist[-1]:
		#     zcX,zcY=np.asarray(zc)[:,1],np.asarray(zc)[:,0]
		#     plt.scatter(zcX,zcY,color="b")
		
		
		for czc in zero_contours:
			plt.plot(czc[:,1],czc[:,0],color="k")
			plt.title(self._timestep_count)
		#plt.savefig(folder+"{}.png".format(self._timestep_count))
		plt.show()
		plt.clf()
	def plot_contours(self,freq=1):
		for i in range(len(self._zerocontourslist)):
			if(i%freq==0):
				for c in self._zerocontourslist[i]:
					plt.plot(c[:,1],c[:,0])
		plt.show()

	def main(self):
		"""
		THIS FUNCTION COMPUTES THE COLLECTION OF ZERO LEVEL SETS GIVEN 
		THE SPEED AND VELOCITY FIELD PROFILE UNTIL IT REACHES THE DESTINATION.
		"""
		while (not self._reached): 

			self._narrow_band_construction(self._zerocontourslist[-1])
			dangerzone_signlist=self.__get_dangerzone_signlist()
			self.__check_targetinNarrowband()
			reinitialise=False
			while(not reinitialise):
				
				self._phi=self._serial_scheme2()
				self._timestep_count+=1

				self._zerocontourslist.append(self.__contouring_function())
				if(self._targetInNarrowband):
					target_phi=self._phi[self._target[0]][self._target[1]]
					if(target_phi<=0):
						self._reached=True
						reinitialise=True
						break
				for ind in range(len(self._danger_zone)):
					x,y=self._danger_zone[ind][0],self._danger_zone[ind][1]
					sign=np.sign(self._phi[x][y])
					if(not sign==dangerzone_signlist[ind]):
						reinitialise=True
						break

	#FUNCTIONS  RELATED TO BACKTRACKING .##################################
	def __slope(self,pointa,pointb,tol=1e-12):
		ia,ja=pointa
		ib,jb=pointb
		if(abs(ja-jb) >tol):
			m=((ia-ib))/((ja-jb))
		else:
			m=1/tol
		return m

	def __get_intersection_pt(self,edge1,edge2,tol=1e-12):
		m1=self.__slope(edge1[0],edge1[1])
		m2=self.__slope(edge2[0],edge2[1])
		y1,x1=edge1[0][0],edge1[0][1]
		b1,a1=edge2[0][0],edge2[0][1]
		if (abs(m1-m2)>tol):
			x=((b1-y1+m1*x1-m2*a1)/(m1-m2))
			y=(y1+m1*(x-x1))
			inside_edge,is_vertex=self.__check_point_inside_edge((y,x),edge1)
			if(inside_edge):
				#print(edge1,edge2)
				return True,is_vertex,(y,x)
				
			else:
				#print((y,x),edge1,edge2)
				return False,None,(y,x)
				
		else:
			return False,None,None

	def __check_point_inside_edge(self,point,edge):
		vect=lambda point1,point2:((point1[0]-point2[0]),(point1[1]-point2[1]))
		AB=vect(edge[0],point)
		BC=vect(point,edge[1])
		dot_product=AB[0]*BC[0]+AB[1]*BC[1]
		#print(dot_product,point,edge)
		if(dot_product==0):
			return True,True
		elif(dot_product>0):
			return True,False
		else:
			return False,False

	def __next_point_calc(self,xt,f,u,v,f_dir):
		ib,jb=xt
		ia=ib-self._dt*(v+f*f_dir[1])/self._dX[1]
		ja=jb-self._dt*(u+f*f_dir[0])/self._dX[0]
		return ia,ja

	def __get_direction_target(self,zcs,vertex):
		min_dist=float("inf")
		min_point=None
		min_edge=None
		for zc in zcs:
			if(zc[0][0]==zc[-1][0] and zc[0][1]==zc[-1][1]):
				zc=zc[:-1]
			for ind in range(len(zc)-1):
				i,j=zc[ind]
				dist=(((self._dX[1]*(i-vertex[0]))**2+(self._dX[0]*(j-vertex[1]))**2))**.5
				if(dist<min_dist):
					min_dist=dist
					min_ind=ind
					ind_nxt=(min_ind+1)%len(zc)
					ind_prv=(min_ind-1)%len(zc)
					edge=(zc[ind_prv],zc[ind_nxt])
					min_point=zc[ind]
		#print(self.__get_direction(edge,min_point,True))
		return self.__get_direction(edge,min_point,True),min_point

	#def __interpolate(self):

	def __get_direction(self,edge,point,is_vertex):
		if(not is_vertex):
			pointa,pointb=edge
			m=self.__slope(pointa,pointb)
			if(m):
				normal_m=-1/m
			else:
				normal_m=1e12
			
		else:
			#print(edge)
			pointa,pointb=edge
			m1=self.__slope(point,pointa)
			m2=self.__slope(point,pointb)
			if(m1):
				normal_m1=-1/m1
			else:
				normal_m1=1e12
			if(m2):
				normal_m2=-1/m2
			else:
				normal_m2=1e12
			normal_m=(normal_m1+normal_m2)/2
		
		cos_thetha=1/(1+normal_m**2)**.5
		sin_thetha=normal_m/(1+normal_m**2)**.5 
		if(normal_m>=1e10):
			cos_thetha=0
			sin_thetha=1 
		origin_vector=((point[1]-self._start[1])*self._dX[1],(point[0]-self._start[0])*self._dX[0])
		dot_product=origin_vector[0]*cos_thetha+origin_vector[1]*sin_thetha
		#print(cos_thetha,sin_thetha,dot_product)
		if(dot_product<0):
			cos_thetha*=-1
			sin_thetha*=-1
		#print((atan(normal_m)*180/pi)%360 )
		return cos_thetha,sin_thetha

	def __get_min_projection(self,zcs,projected_point):
		ip,jp=projected_point
		min_dist=float("inf")
		min_point=projected_point
		is_vertex=True
		edge=None

		for zc in zcs:

			if(zc[0][0]==zc[-1][0] and zc[0][1]==zc[-1][1]):
				zc=zc[:-1]

			for ind in range(len(zc)-1):
				ind_nxt=(ind+1)%len(zc)
				ia,ja=zc[ind]
				#print(ip,jp,zc[ind])
				ib,jb=zc[ind_nxt]
				ic,jc=(ia+ib)/2,(ja+jb)/2
				dista=(((self._dX[1]*(ip-ia))**2)+(self._dX[0]*(jp-ja))**2)**.5
				distc=(((self._dX[1]*(ip-ic))**2)+(self._dX[0]*(jp-jc))**2)**.5
				#print(min_dist,dista,distc)
				if(min_dist>min(dista,distc)):
					min_dist=min(dista,distc)
					if (abs(distc-min_dist)<1e-5):
						min_point=(ic,jc)
						is_vertex=False
						edge=((ia,ja),(ib,jb))
						#print(edge)
					else:
						min_point=(ia,ja)
						is_vertex=True
						ind_prv=(ind-1)%len(zc)
						iprev,jprev=zc[ind_prv]
						edge=((iprev,jprev),(ib,jb))
						#print(edge)
				#print(min_dist)
		return is_vertex,min_point,edge,min_dist

	def __get_min_projection2(self,zcs,projected_pt,prev_pt):
		dist=lambda point1,point2:((self._dX[1]*(point1[0]-point2[0]))**2+(self._dX[0]*(point1[1]-point2[1]))**2)**.5
		
		path_edge=(prev_pt,projected_pt)
		min_dist=float("inf")
		min_point=projected_pt
		is_vertex=True
		edge=None
		
		for z in zcs:
			zc=z
			if(z[0][0]==z[-1][0] and z[0][1]==z[-1][1]):
				zc=z[:-1]
			
			for ind in range(len(zc)-1):
				ind_nxt=(ind+1)%len(zc)
				cnt_edge=(zc[ind],zc[ind_nxt])
				
				intersects,is_avertex,intersect_pt=self.__get_intersection_pt(cnt_edge,path_edge)
				
				if(intersects):
					
					distance_err=dist(projected_pt,intersect_pt)
					if(min_dist>distance_err):
						#print(intersect_pt,projected_pt,prev_pt,cnt_edge)
						min_point=intersect_pt
						min_dist=distance_err
						is_vertex=is_avertex
						edge=cnt_edge

						if(is_vertex):
							ind_prv=(ind-1)%len(zc)
							edge=(zc[ind_prv],zc[ind_nxt])
		
		# if (edge==None):
		#     return self.__get_min_projection(zcs,projected_pt)
		return is_vertex,min_point,edge,min_dist


	def __approx_vels(self,point,t):
		ip,jp=point
		i_prv,i_nxt=floor(ip),ceil(ip)
		j_prv,j_nxt=floor(jp),ceil(jp)
		var=lambda i,j:(self._F[t-1][i][j],self._u[t-1][i][j],self._v[t-1][i][j])

		if(i_prv!=i_nxt and j_prv!=j_nxt):
			points=[(i_prv,j_prv),(i_prv,j_nxt),(i_nxt,j_prv),(i_nxt,j_nxt)]
			z=[var(i_prv,j_prv),var(i_prv,j_nxt),var(i_nxt,j_prv),var(i_nxt,j_nxt)]
			#print(points)
			interpolator=LinearNDInterpolator(points,z)
			return interpolator((ip,jp))
		else:
			return self._F[t-1][int(ip)][int(jp)],self._u[t-1][int(ip)][int(jp)],self._v[t-1][int(ip)][int(jp)]

	#######################EXPERIMENTAL BACKTRACKING FUNCTIONS###############################

	# def _backtracking(self):
	#     self._path=[]
	#     self._projected_pts=[]
	#     xt=self._target
	#     zcs=self._zerocontourslist[-1]
	#     fdir,min_point=self.__get_direction_target(zcs,xt)

	#     self._path.append(xt)
	#     self._projected_pts.append(xt)
	#     timestep=self._timestep_count
		
	#     #print(fdir)
	#     for zcs in self._zerocontourslist[::-1][1:]:
	#         f,u,v=self.__approx_vels(xt,timestep)
	#         timestep-=1
	#         projected_pt=self.__next_point_calc(xt,f,u,v,fdir)
	#         #print(fdir)
	#         self._projected_pts.append(projected_pt)
	#         #print(projected_pt,xt)
	#         is_vertex,xt,edge,min_dist=self.__get_min_projection2(zcs,projected_pt,xt)

	#         self._path.append(xt)
	#         fdir=self.__get_direction(edge,xt,is_vertex)
	#     self._path.append(self._start)
	#     self._projected_pts.append(self._start)
	#     self._path=np.asarray(self._path)
	#     self._projected_pts=np.asarray(self._projected_pts)
	#     self.__backtracked=True
	#     return self._path

	# def _backtracking2(self):
	#     self._path=[]
	#     self._projected_pts=[]
	#     xt=self._target
	#     zcs=self._zerocontourslist[-1]
	#     fdir,min_point=self.__get_direction_target(zcs,xt)

	#     self._path.append(xt)
	#     self._projected_pts.append(xt)
	#     timestep=self._timestep_count
		
	#     #print(fdir)
	#     for zcs in self._zerocontourslist[::-1][1:]:
	#         f,u,v=self.__approx_vels(xt,timestep)
	#         timestep-=1
	#         projected_pt=self.__next_point_calc(xt,f,u,v,fdir)
	#         self._projected_pts.append(projected_pt)
	#         #print(projected_pt,xt)
	#         is_vertex,xt,edge,min_dist=self.__get_min_projection(zcs,projected_pt)

	#         self._path.append(xt)
	#         fdir=self.__get_direction(edge,xt,is_vertex)
	#     self._path.append(self._start)
	#     self._projected_pts.append(self._start)
	#     self._path=np.asarray(self._path)
	#     self._projected_pts=np.asarray(self._projected_pts)
	#     #self.__backtracked=True
	#     return self._path
	##################################################################################################
	def _backtracking(self,heur=1):
		self._path=[]
		self._projected_pts=[]
		xt=self._target
		zcs=self._zerocontourslist[-1]
		fdir,min_point=self.__get_direction_target(zcs,xt)

		self._path.append(xt)
		self._projected_pts.append(xt)
		timestep=self._timestep_count
		
		
		for zcs in self._zerocontourslist[::-1][1:]:
			f,u,v=self.__approx_vels(xt,timestep)
			timestep-=1
			projected_pt=self.__next_point_calc(xt,f,u,v,fdir)
			self._projected_pts.append(projected_pt)
			
			is_vertex1,xt1,edge1,min_dist1=self.__get_min_projection(zcs,projected_pt)
			is_vertex2,xt2,edge2,min_dist2=self.__get_min_projection2(zcs,projected_pt,xt)
			if((edge2 is None ) or (min_dist2>heur*min_dist1)):
				is_vertex,xt,edge=is_vertex1,xt1,edge1
				#print("hh",timestep)
			else:
				is_vertex,xt,edge=is_vertex2,xt2,edge2
			
			self._path.append(xt)
			fdir=self.__get_direction(edge,xt,is_vertex)
		self._path.append(self._start)
		self._projected_pts.append(self._start)
		self._path=np.asarray(self._path)
		self._projected_pts=np.asarray(self._projected_pts)
		
		return self._path

	# def _plot_contours_with_path(self,freq=1):
	#     if(not self.__backtracked):
	#         self._backtracking()
	#     for i in range(len(self._zerocontourslist)):
	#         if(i%freq==0):
	#             for c in self._zerocontourslist[i]:
	#                 plt.plot(c[:,1],c[:,0])
	#     x,y=self._path[:,1],self._path[:,0]
	#     #xp,yp=self._projected_pts[:,1],self._projected_pts[:,0]
	#     plt.plot(x,y,color="r")
	#     #plt.plot(xp,yp,color="r")
	#     plt.show()

	# def _plot_contours_with_path2(self,freq=1):
	#     #if(not self.__backtracked):
	#     self._backtracking3(2)
	#     for i in range(len(self._zerocontourslist)):
	#         if(i%freq==0):
	#             for c in self._zerocontourslist[i]:
	#                 plt.plot(c[:,1],c[:,0])

	#     x,y=self._path[:,1],self._path[:,0]
	#     #xp,yp=self._projected_pts[:,1],self._projected_pts[:,0]
	#     plt.plot(x,y,color="r")
	#     #plt.plot(xp,yp,color="r")
	#     plt.show()

	def _plot_contours_with_path3(self,heur=1,freq=1,dir_path="/",lims=(50,50), show_contours=False):
		#if(not self.__backtracked):
		plt.clf()
		xlim, ylim = lims
		plt.xlim(0,xlim)
		plt.ylim(0,ylim)
		
		plt.scatter(self._start[1],self._start[0],marker=".",label="start")
		plt.scatter(self._target[1],self._target[0],marker="*",label="target")
		self._backtracking(heur)

		if show_contours:
			for i in range(len(self._zerocontourslist)):
				if(i%freq==0):
					for c in self._zerocontourslist[i]:
						plt.plot(c[:,1],c[:,0])
						plt.legend()
						# plt.savefig(dir_path+"{}.png".format(i))

		x,y=self._path[:,1],self._path[:,0]
		# for i in range(1,len(x)):
		# 	plt.plot([x[i],x[i-1]],[y[i],y[i-1]],color="r")
		# 	plt.legend()
			# plt.savefig(dir_path+"{}.png".format(len(self._zerocontourslist)+i))
		#xp,yp=self._projected_pts[:,1],self._projected_pts[:,0]
		plt.plot(x,y,color="r",label="level_set path")
		plt.legend()
		# plt.savefig("figures/"+"{}.png".format(len(self._zerocontourslist)+len(x)))
		#plt.plot(xp,yp,color="r")
		# plt.show()
############################################################

