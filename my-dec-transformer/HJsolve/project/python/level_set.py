
def level_upwind_scheme(prev_phi,Fij,i,j,dX,dt,dims):
	"""
	first order upwind scheme
	calculate the change in level set function at a point in space for next time step
	Ref:Level Set Methods Evolving Interfaces in Geometry,Fluid Mechanics,Computer Vision and Material science by J.A.Sethian
	pg no 54 ,55,56
	"""
	dx=dX[0]
	dy=dX[1]

	infinity=float("inf")

	if (i<1 or j<1 or i>dims[0]-2 or j>dims[1]-2):
		return 0

	if(prev_phi[i][j]==infinity):
		return 0
	
	ilist=[i-1,i+1]
	jlist=[j-1,j+1]
	
	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return 0
			else:
				if(prev_phi[ia][j]==infinity):
					return 0
		except:
			if(prev_phi[ia][j]==infinity):
				return 0
	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return 0
			else:
				if(prev_phi[i][ja]==infinity):
					return 0
		except:
			if(prev_phi[i][ja]==infinity):
				return 0

	if(prev_phi[i][j]!=infinity and prev_phi[i][j-1]!=infinity and prev_phi[i][j+1]!=infinity):
		Dijk_minusx=(prev_phi[i][j]-prev_phi[i][j-1])/dx
	else:
		Dijk_minusx=0

	if(prev_phi[i][j]!=infinity and prev_phi[i][j-1]!=infinity and prev_phi[i][j+1]!=infinity):
		Dijk_plusx=(prev_phi[i][j+1]-prev_phi[i][j])/dx
	else:
		Dijk_plusx=0
	
	if(prev_phi[i][j]!=infinity and prev_phi[i-1][j]!=infinity and prev_phi[i+1][j]!=infinity):
		Dijk_minusy=(prev_phi[i][j]-prev_phi[i-1][j])/dy
	else:
		Dijk_minusy=0
		
	if(prev_phi[i][j]!=infinity and prev_phi[i-1][j]!=infinity and prev_phi[i+1][j]!=infinity):
		Dijk_plusy=(prev_phi[i+1][j]-prev_phi[i][j])/dy
	else:
		Dijk_plusy=0

	
	

	del_plus=((min(Dijk_plusx,0))**2 + (max(Dijk_minusx,0))**2 +
				(min(Dijk_plusy,0))**2 + (max(Dijk_minusy,0))**2 )**.5

	del_minus=((max(Dijk_plusx,0))**2 + (min(Dijk_minusx,0))**2 +
				(max(Dijk_plusy,0))**2 + (min(Dijk_minusy,0))**2 )**.5

	del_phi=-dt*((max(Fij,0)* del_plus)+ (min(Fij,0)* del_minus))
	# if(del_phi):
	# 	print(del_phi,i,j,prev_phi[i][j],prev_phi[i][j-1],prev_phi[i+1][j],prev_phi[i][j+1])
	# if(del_phi>10):
	# 	print(del_phi,i,j)
	return del_phi

def m(x,y):
	if((x<=0 and y<=0) or (x>=0 and y>=0)):
		if(abs(x)<=abs(y)):
			return x
		else:
			return y
	else:
		return 0

def second_order_scheme(prev_phi,Fij,i,j,dX,dt,dims):
	dx=dX[0]
	dy=dX[1]
	"""
	second order ENO scheme
	calculate the change in level set function at a point in space for next time step
	Ref:Level Set Methods Evolving Interfaces in Geometry,Fluid Mechanics,Computer Vision and Material science by J.A.Sethian
	pg no 56,57
	"""
	infinity=float("inf")
	
	if(prev_phi[i][j]==infinity):
		return 0
	if (i<2 or j<2 or i>dims[0]-3 or j>dims[1]-3):
		return level_upwind_scheme(prev_phi,Fij,i,j,dX,dt,dims)
	ilist=[i-2,i-1,i+1,i+2]
	jlist=[j-2,j-1,j+1,j+2]

	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return level_upwind_scheme(prev_phi,Fij,i,j,dX,dt,dims)
			else:
				if(prev_phi[ia][j]==infinity):
					return level_upwind_scheme(prev_phi,Fij,i,j,dX,dt,dims)
		except:
			if(prev_phi[ia][j]==infinity):
				return level_upwind_scheme(prev_phi,Fij,i,j,dX,dt,dims)
	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return level_upwind_scheme(prev_phi,Fij,i,j,dX,dt,dims)
			else:
				if(prev_phi[i][ja]==infinity):
					return level_upwind_scheme(prev_phi,Fij,i,j,dX,dt,dims)
		except:
			if(prev_phi[i][ja]==infinity):
				return level_upwind_scheme(prev_phi,Fij,i,j,dX,dt,dims)

	Dijk_minusx=(prev_phi[i][j]-prev_phi[i][j-1])/dx
	Dijk_plusx=(prev_phi[i][j+1]-prev_phi[i][j])/dx
	Dijk_minusy=(prev_phi[i][j]-prev_phi[i-1][j])/dy
	Dijk_plusy=(prev_phi[i+1][j]-prev_phi[i][j])/dy

	Dijk_plusx_plusx=(prev_phi[i][j+2] -2* prev_phi[i][j+1] + prev_phi[i][j])/(dx)**2
	Dijk_plusx_minusx=(prev_phi[i][j+1] -2* prev_phi[i][j] + prev_phi[i][j-1])/(dx)**2
	Dijk_minusx_minusx=(prev_phi[i][j] -2* prev_phi[i][j-1] + prev_phi[i][j-2])/(dx)**2
	Dijk_minusx_plusx=Dijk_plusx_minusx

	Dijk_plusy_plusy=(prev_phi[i+2][j] -2* prev_phi[i+1][j] + prev_phi[i][j])/(dy)**2
	Dijk_plusy_minusy=(prev_phi[i+1][j] -2* prev_phi[i][j] + prev_phi[i-1][j])/(dy)**2
	Dijk_minusy_minusy=(prev_phi[i][j] -2* prev_phi[i-1][j] + prev_phi[i-2][j])/(dy)**2
	Dijk_minusy_plusy=Dijk_plusy_minusy


	A=Dijk_minusx + (dx)*m(Dijk_minusx_minusx,Dijk_plusx_minusx)/2
	B=Dijk_plusx -(dx)*m(Dijk_plusx_plusx,Dijk_plusx_minusx)/2

	C=Dijk_minusy + (dy)*m(Dijk_minusy_minusy,Dijk_plusy_minusy)/2
	D=Dijk_plusy -(dy)*m(Dijk_plusy_plusy,Dijk_plusy_minusy)/2

	del_plus=((max(A,0))**2 + (min(B,0))**2 + (max(C,0))**2 + (min(D,0))**2)**.5
	del_minus=((max(B,0))**2 + (min(A,0))**2 + (max(D,0))**2 + (min(C,0))**2)**.5

	del_phi=-dt*((max(Fij,0)* del_plus)+ (min(Fij,0)* del_minus))
	# if(del_phi>10):
	# 	print(del_phi,i,j)
	return del_phi
