#author:revanth

###fluxlimiter functions

def MC(r):
	return max(0,min(min((1+r)/2,2),2*r))

def vanleer(r):
	return (r+abs(r))/(1+abs(r))

def minmod(r):
	return max(0,min(1,r))

def superbee(r):
	return max(0,min(2*r,1),min(r,2))


def upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter):


	dx=dX[0]
	dy=dX[1]

	infinity=float("inf")
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


	uw=u[i][j-1]
	ue=u[i][j]
	
	if(prev_phi[i][j+1]!=infinity and prev_phi[i][j-1]!=infinity):
		phiE=0.5*(ue*(prev_phi[i][j+1]+prev_phi[i][j])-abs(ue)*(prev_phi[i][j+1]-prev_phi[i][j]))
	else:
		phiE=0
	if(prev_phi[i][j+1]!=infinity and prev_phi[i][j-1]!=infinity):
		phiW=0.5*(uw*(prev_phi[i][j]+prev_phi[i][j-1])-abs(uw)*(prev_phi[i][j]-prev_phi[i][j-1]))
	else:
		phiW=0


	vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4

	if(prev_phi[i-1][j]!=infinity and prev_phi[i+1][j]!=infinity):
		phiS=0.5*(vs*(prev_phi[i][j]+prev_phi[i-1][j])-abs(vs)*(prev_phi[i][j]-prev_phi[i-1][j]))
	else:
		phiS=0

	if(prev_phi[i-1][j]!=infinity and prev_phi[i+1][j]!=infinity):
		phiN=0.5*(vn*(prev_phi[i][j]+prev_phi[i+1][j])-abs(vn)*(prev_phi[i+1][j]-prev_phi[i][j]))
	else:
		phiN=0

	del_phi=-dt*(((phiE-phiW)/dx)+((phiN-phiS)/dy))

	return del_phi

def tvd_advection(prev_phi,u,v,i,j,dX,dt,fluxlimiter):
	"""
	Reference:Matlab implemenataion of the advection term from 
	2.29 Finite Volume MATLAB Framework Documentation
		Manual written by: Matt uwckermann, Pierre Lermusiaux
		April 29, 2013
	
	"""
	
	dx=dX[0]
	dy=dX[1]

	if(fluxlimiter==0):
		fluxlimiter=vanleer
	elif(fluxlimiter==1):
		fluxlimiter=minmod
	elif(fluxlimiter==2):
		fluxlimiter=MC
	elif(fluxlimiter==3):
		fluxlimiter=superbee

	infinity=float("inf")
	if(prev_phi[i][j]==infinity):
		return 0
	
	ilist=[i-2,i-1,i+1]
	jlist=[j-2,j-1,j+1]
	
	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return 0
			else:
				if(prev_phi[ia][j]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
		except:
			if(prev_phi[ia][j]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return 0
			else:
				if(prev_phi[i][ja]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
		except:
			if(prev_phi[i][ja]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)


	uw=u[i][j-1]
	ue=u[i][j]

	denominator=uw*(prev_phi[i][j]-prev_phi[i][j-1])
	if(denominator):
		Rx_minus= ((.5* (uw+abs(uw)) * ((prev_phi[i][j-1])+(prev_phi[i][j-2])) ) +(
		       (.5* (uw-abs(uw)) * ((prev_phi[i][j+1])+(prev_phi[i][j])) )))/(denominator)
	else:
		Rx_minus=1
	
	Fx_minus=uw*((prev_phi[i][j]+prev_phi[i][j-1])/2) - abs(
		uw)* (.5*(prev_phi[i][j]-prev_phi[i][j-1]))*(1-(1-abs(uw*dt/dx))*fluxlimiter(Rx_minus))
	
	denominator=ue*(prev_phi[i][j+1]-prev_phi[i][j])
	if(denominator):
		Rx_plus=((.5* (ue+abs(ue)) * ((prev_phi[i][j])+(prev_phi[i][j-1])) )+(
		(.5* (ue-abs(ue)) * ((prev_phi[i][j+1])+(prev_phi[i][j+2]))  )))/(denominator)
	else:
		Rx_plus=1

	
	Fx_plus=ue*((prev_phi[i][j]+prev_phi[i][j+1])/2) - abs(
		ue)* ((prev_phi[i][j+1]-prev_phi[i][j])/2)*(1-(1-abs(ue*dt/dx))*fluxlimiter(Rx_plus))
	
	
	v_minus=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4
	v_plus=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	
	denominator=v_minus*(prev_phi[i][j]-prev_phi[i-1][j])
	if(denominator):
		Ry_minus=((.5* (v_minus+abs(v_minus)) * ((prev_phi[i-1][j])+(prev_phi[i-2][j])) )+(
		(.5* (v_minus-abs(v_minus)) * ((prev_phi[i+1][j])+(prev_phi[i][j]))  )))/(denominator)
	else:
		Ry_minus=1
	
	Fy_minus=v_minus*((prev_phi[i][j]+prev_phi[i-1][j])/2) - abs(
		v_minus)* ((prev_phi[i][j]-prev_phi[i-1][j])/2)*(1-(1-abs(v_minus*dt/dy))*fluxlimiter(Ry_minus))

	
	denominator=v_plus*(prev_phi[i+1][j]-prev_phi[i][j])
	if(denominator):
		Ry_plus=((.5* (v_plus+abs(v_plus)) * ((prev_phi[i][j])+(prev_phi[i-1][j])) )+(
		(.5* (v_plus-abs(v_plus)) * ((prev_phi[i+2][j])+(prev_phi[i+1][j]))) ) )/(denominator)
	else:
		Ry_plus=1
	Fy_plus=v_plus*((prev_phi[i+1][j]+prev_phi[i][j])/2) - abs(
		v_plus)* ((prev_phi[i+1][j]-prev_phi[i][j])/2)*(1-(1-abs(v_plus*dt/dy))*fluxlimiter(Ry_plus))

	del_phi= -dt*(((Fx_plus-Fx_minus)/dx)+((Fy_plus-Fy_minus)/dy))
	
	return del_phi

def tvd_lax_wendroff(prev_phi,u,v,i,j,dX,dt,fluxlimiter):
	"""
	Reference:Matlab implemenataion of the advection term from 
	2.29 Finite Volume MATLAB Framework Documentation
		Manual written by: Matt uwckermann, Pierre Lermusiaux
		April 29, 2013
	
	"""
	
	dx=dX[0]
	dy=dX[1]

	if(fluxlimiter==0):
		fluxlimiter=vanleer
	elif(fluxlimiter==1):
		fluxlimiter=minmod
	elif(fluxlimiter==2):
		fluxlimiter=MC
	elif(fluxlimiter==3):
		fluxlimiter=superbee

	infinity=float("inf")
	if(prev_phi[i][j]==infinity):
		return 0
	
	ilist=[i-2,i-1,i+1]
	jlist=[j-2,j-1,j+1]
	
	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
			else:
				if(prev_phi[ia][j]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
		except:
			if(prev_phi[ia][j]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
			else:
				if(prev_phi[i][ja]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
		except:
			if(prev_phi[i][ja]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)


	uww=u[i][j-2]
	uw=u[i][j-1]
	ue=u[i][j]
	uee=u[i][j+1]
	rx=dt/dx
	ry=dt/dy

	phiE=(ue*(prev_phi[i][j+1]))
	phiW=uw*(prev_phi[i][j])
	phiT=uww*(prev_phi[i][j-1])

	fx_plus_plus=-(phiE-.5*(ue+uee)*prev_phi[i][j+1])
	fx_plus_minus=(phiE-.5*(ue+uw)*prev_phi[i][j])
	fx_minus_plus=-(phiW-.5*(uw+ue)*prev_phi[i][j])
	fx_minus_minus=(phiW-.5*(uw+uww)*prev_phi[i][j-1])
	fx_tminus_plus=-(phiT-.5*(uww+uw)*prev_phi[i][j-1])
	fx_tminus_minus=(phiT-.5*(uww+u[i][j-3])*prev_phi[i][j-2])

	denominator=prev_phi[i][j+1]-prev_phi[i][j]
	
	if(denominator):
		Cx_plus_plus=rx*(fx_plus_plus)/(denominator)
		Cx_plus_minus=rx*(fx_plus_minus)/denominator
	else:
		Cx_plus_plus=Cx_plus_minus=0
	
	denominator=prev_phi[i][j]-prev_phi[i][j-1]
	if(denominator):
		Cx_minus_plus=rx*(fx_minus_plus)/denominator
		Cx_minus_minus=rx*(fx_minus_minus)/denominator
	else:
		Cx_minus_minus=Cx_minus_plus=0

	denominator=prev_phi[i][j-1]-prev_phi[i][j-2]
	if(denominator):
		Cx_tminus_plus=rx*(fx_tminus_plus)/denominator
		Cx_tminus_minus=rx*(fx_tminus_minus)/denominator
	else:
		Cx_tminus_plus=Cx_tminus_minus=0


	alphax_plus_plus=.5*(1-Cx_plus_plus)
	alphax_plus_minus=.5*(1+Cx_plus_minus)
	alphax_minus_plus=.5*(1-Cx_minus_plus)
	alphax_minus_minus=.5*(1+Cx_minus_minus)
	alphax_tminus_plus=.5*(1-Cx_tminus_plus)
	alphax_tminus_minus=.5*(1+Cx_tminus_minus)

	denominator=alphax_plus_plus*fx_plus_plus
	if(denominator):
		rx_plus_plus=(alphax_minus_plus*fx_minus_plus/(denominator))
	else:
		rx_plus_plus=0
	
	denominator=(alphax_minus_minus*fx_minus_minus)
	if(denominator):
		rx_plus_minus=(alphax_plus_minus*fx_plus_minus)/denominator
	else:
		rx_plus_minus=0
	denominator=(alphax_minus_plus*fx_minus_plus)
	if(denominator):
		rx_minus_plus=(alphax_tminus_plus*fx_tminus_plus)/denominator
	else:
		rx_minus_plus=0
	
	denominator=(alphax_tminus_minus*fx_tminus_minus)
	if(denominator):
		rx_minus_minus=(alphax_minus_minus*fx_minus_minus)/denominator
	else:
		rx_minus_minus=0
	
	Fx_plus=(phiE-(fluxlimiter(rx_plus_plus)*alphax_plus_plus*fx_plus_plus-fluxlimiter(rx_plus_minus)*fx_plus_minus*alphax_plus_minus))
	Fx_minus=(phiW-(fluxlimiter(rx_minus_plus)*fx_minus_plus*alphax_minus_plus-fluxlimiter(rx_minus_minus)*fx_minus_minus*alphax_minus_minus))
	

	vnn=(v[i+2][j-1]+v[i+2][j]+v[i+1][j-1]+v[i+1][j])/4
	vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4
	vss=(v[i-2][j-1]+v[i-2][j]+v[i-1][j-1]+v[i-1][j])/4
	phiS=vs*prev_phi[i][j]
	phiN=vn*prev_phi[i+1][j]
	phiSS=vss*prev_phi[i-2][j]

	fy_plus_plus=-(phiN-.5*(vn+vnn)*prev_phi[i+1][j])
	fy_plus_minus=(phiN-.5*(vn+vs)*prev_phi[i][j])
	fy_minus_plus=-(phiS-.5*(vs+vn)*prev_phi[i][j])
	fy_minus_minus=(phiS-.5*(vs+vss)*prev_phi[i-1][j])
	fy_tminus_plus=-(phiSS-.5*(vss+vs)*prev_phi[i-1][j])
	fy_tminus_minus=(phiSS-.5*(vss+u[i-3][j])*prev_phi[i-2][j])

	denominator=prev_phi[i+1][j]-prev_phi[i][j]
	
	if(denominator):
		Cy_plus_plus=ry*(fy_plus_plus)/(denominator)
		Cy_plus_minus=ry*(fy_plus_minus)/denominator
	else:
		Cy_plus_plus=Cy_plus_minus=0
	
	denominator=prev_phi[i][j]-prev_phi[i-1][j]
	if(denominator):
		Cy_minus_plus=ry*(fy_minus_plus)/denominator
		Cy_minus_minus=ry*(fy_minus_minus)/denominator
	else:
		Cy_minus_minus=Cy_minus_plus=0

	denominator=prev_phi[i-1][j]-prev_phi[i-2][j]
	if(denominator):
		Cy_tminus_plus=ry*(fy_tminus_plus)/denominator
		Cy_tminus_minus=ry*(fy_tminus_minus)/denominator
	else:
		Cy_tminus_plus=Cy_tminus_minus=0


	alphay_plus_plus=.5*(1-Cy_plus_plus)
	alphay_plus_minus=.5*(1+Cy_plus_minus)
	alphay_minus_plus=.5*(1-Cy_minus_plus)
	alphay_minus_minus=.5*(1+Cy_minus_minus)
	alphay_tminus_plus=.5*(1-Cy_tminus_plus)
	alphay_tminus_minus=.5*(1+Cy_tminus_minus)

	denominator=(alphay_plus_plus*fy_plus_plus)
	if(denominator):
		ry_plus_plus=(alphay_minus_plus*fy_minus_plus/denominator)
	else:
		ry_plus_plus=0
	denominator=(alphay_minus_minus*fy_minus_minus)
	if(denominator):
		ry_plus_minus=(alphay_plus_minus*fy_plus_minus)/denominator
	else:
		ry_plus_minus=0
	
	denominator=(alphay_minus_plus*fy_minus_plus)
	
	if(denominator):
		ry_minus_plus=(alphay_tminus_plus*fy_tminus_plus)/denominator
	else:
		ry_minus_plus=0

	denominator=(alphay_tminus_minus*fy_tminus_minus)
	if(denominator):
		ry_minus_minus=(alphay_minus_minus*fy_minus_minus)/denominator
	else:
		ry_minus_minus=0
	
	Fy_plus=(phiN-(fluxlimiter(ry_plus_plus)*alphay_plus_plus*fy_plus_plus-fluxlimiter(ry_plus_minus)*alphay_plus_minus*fy_plus_minus))
	Fy_minus=(phiS-(fluxlimiter(ry_minus_plus)*alphay_minus_plus*fy_minus_plus-fluxlimiter(ry_minus_minus)*alphay_minus_minus*fy_minus_minus))
	



	del_phi=-dt*(((Fx_plus-Fx_minus)/dx)+((Fy_plus-Fy_minus)/dy))
	#print(del_phi)
	return del_phi

def lax_wendroff(prev_phi,u,v,i,j,dX,dt,fluxlimiter):


	dx=dX[0]
	dy=dX[1]

	if(fluxlimiter==0):
		fluxlimiter=vanleer
	elif(fluxlimiter==1):
		fluxlimiter=minmod
	elif(fluxlimiter==2):
		fluxlimiter=MC
	elif(fluxlimiter==3):
		fluxlimiter=superbee

	infinity=float("inf")
	ilist=[i-1,i+1]
	jlist=[j-1,j+1]

	if(prev_phi[i][j]==infinity):
		return 0

	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
			else:
				if(prev_phi[ia][j]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
		except:
			if(prev_phi[ia][j]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
			else:
				if(prev_phi[i][ja]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)
		except:
			if(prev_phi[i][ja]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

	rx=dt/dx
	ry=dt/dy

	uww=u[i][j-2]
	uw=u[i][j-1]
	ue=u[i][j]
	uee=u[i][j+1]
	phiE=0.5*(ue*(prev_phi[i][j+1]+prev_phi[i][j])-abs(ue)*(prev_phi[i][j+1]-prev_phi[i][j]))+abs(.5*ue*rx)*((prev_phi[i][j+1]*.5*(ue+uee))-(prev_phi[i][j]*.5*(ue+uw)))
	phiW=0.5*(uw*(prev_phi[i][j]+prev_phi[i][j-1])-abs(uw)*(prev_phi[i][j]-prev_phi[i][j-1]))+abs(.5*uw*rx)*((prev_phi[i][j]*.5*(ue+uw))-(prev_phi[i][j-1]*.5*(uww+uw)))

	vnn=(v[i+2][j-1]+v[i+2][j]+v[i+1][j-1]+v[i+1][j])/4
	vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4
	vss=(v[i-2][j-1]+v[i-1][j]+v[i-1][j-1]+v[i-2][j])/4
	phiS=0.5*(vs*(prev_phi[i][j]+prev_phi[i-1][j])-abs(vs)*(prev_phi[i][j]-prev_phi[i-1][j]))+abs(.5*vs*ry)*((prev_phi[i][j]*.5*(vs+vss))-(prev_phi[i-1][j]*.5*(vs+vn)))
	phiN=0.5*(vn*(prev_phi[i][j]+prev_phi[i+1][j])-abs(vn)*(prev_phi[i+1][j]-prev_phi[i][j]))+abs(.5*vn*ry)*((prev_phi[i+1][j]*.5*(vn+vnn))-(prev_phi[i][j]*.5*(vs+vn)))

	del_phi=-dt*(((phiE-phiW)/dx)+((phiN-phiS)/dy))

	return del_phi

def nondiffusive_lax_wendroff(prev_phi,u,v,i,j,dX,dt,fluxlimiter):
#non diffusive lax wendroff works very well in the forward direction

	dx=dX[0]
	dy=dX[1]

	infinity=float("inf")
	ilist=[i-1,i,i+1]
	jlist=[j-1,j,j+1]
	if(prev_phi[i][j]==infinity):
		return 0

	for ia in ilist:
		for ja in jlist:
			try:
				if ia not in prev_phi.keys():
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

				else:
					if ja not in prev_phi[ia].keys():
						return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

					else:
						if(prev_phi[ia][ja]==infinity):
							return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			except:
				if(prev_phi[ia][ja]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)


	rx=dt/dx
	ry=dt/dy

	uww=u[i][j-2]
	uw=u[i][j-1]
	ue=u[i][j]
	uee=u[i][j+1]
	phiE=0.5*(ue*(prev_phi[i][j+1]+prev_phi[i][j])-abs(ue)*(prev_phi[i][j+1]-prev_phi[i][j]))+(.5*(1-ue*rx))*((prev_phi[i][j+1]*.5*(ue+uee))-(prev_phi[i][j]*.5*(ue+uw)))
	phiW=0.5*(uw*(prev_phi[i][j]+prev_phi[i][j-1])-abs(uw)*(prev_phi[i][j]-prev_phi[i][j-1]))+(.5*(1-uw*rx))*((prev_phi[i][j]*.5*(ue+uw))-(prev_phi[i][j-1]*.5*(uww+uw)))

	vnn=(v[i+2][j-1]+v[i+2][j]+v[i+1][j-1]+v[i+1][j])/4
	vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4
	vss=(v[i-2][j-1]+v[i-1][j]+v[i-1][j-1]+v[i-2][j])/4
	phiS=0.5*(vs*(prev_phi[i][j]+prev_phi[i-1][j])-abs(vs)*(prev_phi[i][j]-prev_phi[i-1][j]))+(.5*(1-vs*ry))*((prev_phi[i][j]*.5*(vs+vss))-(prev_phi[i-1][j]*.5*(vs+vn)))
	phiN=0.5*(vn*(prev_phi[i][j]+prev_phi[i+1][j])-abs(vn)*(prev_phi[i+1][j]-prev_phi[i][j]))+(.5*(1-vn*ry))*((prev_phi[i+1][j]*.5*(vn+vnn))-(prev_phi[i][j]*.5*(vs+vn)))

	del_phi=-dt*(((phiE-phiW)/dx)+((phiN-phiS)/dy))

	return del_phi

def lax_wendroff_wiki(prev_phi,u,v,i,j,dX,dt,fluxlimiter):

#lax wendroff tvd taken from wikipedia
	dx=dX[0]
	dy=dX[1]

	if(fluxlimiter==0):
		fluxlimiter=vanleer
	elif(fluxlimiter==1):
		fluxlimiter=minmod
	elif(fluxlimiter==2):
		fluxlimiter=MC
	elif(fluxlimiter==3):
		fluxlimiter=superbee

	infinity=float("inf")
	ilist=[i-2,i-1,i+1,i+2]
	jlist=[j-2,j-1,j+1,j+2]

	if(prev_phi[i][j]==infinity):
		return 0
	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			else:
				if(prev_phi[ia][j]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

		except:
			if(prev_phi[ia][j]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			else:
				if(prev_phi[i][ja]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

		except:
			if(prev_phi[i][ja]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)


	rx=dt/dx
	ry=dt/dy

	uww=u[i][j-2]
	uw=u[i][j-1]
	ue=u[i][j]
	uee=u[i][j+1]
	Aplus=.5*(prev_phi[i][j]+prev_phi[i][j+1])
	Aminus=.5*(prev_phi[i][j]+prev_phi[i][j-1])
	Fx_plus=.5*(ue+uee)*prev_phi[i][j+1]-rx*Aplus*(.5*(ue+uee)*prev_phi[i][j+1]-.5*(ue+uw)*prev_phi[i][j])
	Fx_minus=.5*(uww+uw)*prev_phi[i][j-1]-rx*Aminus*(.5*(ue+uw)*prev_phi[i][j]-.5*(uw+uww)*prev_phi[i][j-1])

	vnn=(v[i+2][j-1]+v[i+2][j]+v[i+1][j-1]+v[i+1][j])/4
	vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4
	vss=(v[i-2][j-1]+v[i-1][j]+v[i-1][j-1]+v[i-2][j])/4

	Bplus=.5*(prev_phi[i][j]+prev_phi[i+1][j])
	Bminus=.5*(prev_phi[i][j]+prev_phi[i-1][j])
	Fy_plus=.5*(vn+vnn)*prev_phi[i+1][j]-ry*Bplus*(.5*(vn+vnn)*prev_phi[i+1][j]-.5*(vn+vs)*prev_phi[i][j])
	Fy_minus=.5*(vss+vs)*prev_phi[i-1][j]-ry*Bminus*(.5*(vn+vs)*prev_phi[i][j]-.5*(vs+vss)*prev_phi[i-1][j])

	del_phi=-.5*(rx*(Fx_plus-Fx_minus)+ry*(Fy_plus-Fy_minus))

	return del_phi

def lax_wendroff_book(prev_phi,u,v,i,j,dX,dt,fluxlimiter):

	"""tvd laxwendroff as stated in Gilbert strang's book titled 'Mathematics for engineers part II ' """
	dx=dX[0]
	dy=dX[1]

	if(fluxlimiter==0):
		fluxlimiter=vanleer
	elif(fluxlimiter==1):
		fluxlimiter=minmod
	elif(fluxlimiter==2):
		fluxlimiter=MC
	elif(fluxlimiter==3):
		fluxlimiter=superbee

	infinity=float("inf")
	ilist=[i-2,i-1,i+1,i+2]
	jlist=[j-2,j-1,j+1,j+2]
	if(prev_phi[i][j]==infinity):
		return 0

	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			else:
				if(prev_phi[ia][j]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

		except:
			if(prev_phi[ia][j]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			else:
				if(prev_phi[i][ja]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

		except:
			if(prev_phi[i][ja]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)


	rx=dt/dx
	ry=dt/dy

	uww=u[i][j-2]
	uw=u[i][j-1]
	ue=u[i][j]
	uee=u[i][j+1]

	denominator=(prev_phi[i][j+1]-prev_phi[i][j])
	if(denominator):
		Aplus=.5*(-(ue+uw)*prev_phi[i][j]+(ue+uee)*prev_phi[i][j+1])/denominator
	else:
		Aplus=0
	
	denominator=(prev_phi[i][j]-prev_phi[i][j-1])
	if(denominator):
		Aminus=.5*((ue+uw)*prev_phi[i][j]-(uww+uw)*prev_phi[i][j-1])/denominator
	else:
		Aminus=0


	Fx_plus=.5*(ue+uee)*prev_phi[i][j+1]-((rx)*(Aplus**2))*(prev_phi[i][j+1]-prev_phi[i][j])
	Fx_minus=.5*(uww+uw)*prev_phi[i][j-1]-((rx)*(Aminus**2))*(prev_phi[i][j]-prev_phi[i][j-1])

	vnn=(v[i+2][j-1]+v[i+2][j]+v[i+1][j-1]+v[i+1][j])/4
	vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4
	vss=(v[i-2][j-1]+v[i-1][j]+v[i-1][j-1]+v[i-2][j])/4

	denominator=prev_phi[i+1][j]-prev_phi[i][j]
	if(denominator):
		Bplus=.5*(-(vn+vs)*prev_phi[i][j]+(vn+vnn)*prev_phi[i+1][j])/denominator
	else:
		Bplus=0
	
	denominator=prev_phi[i][j]-prev_phi[i-1][j]
	if(denominator):
		Bminus=.5*((vn+vs)*prev_phi[i][j]-(vs+vss)*prev_phi[i-1][j])/denominator
	else:
		Bminus=0

	Fy_plus=.5*(vn+vnn)*prev_phi[i+1][j]-(ry*(Bplus**2))*(prev_phi[i+1][j]-prev_phi[i][j])
	Fy_minus=.5*(vss+vs)*prev_phi[i-1][j]-(ry*(Bminus**2))*(prev_phi[i][j]-prev_phi[i-1][j])

	del_phi=-.5*(rx*(Fx_plus-Fx_minus)+ry*(Fy_plus-Fy_minus))

	return del_phi

def tvd_lax_wendroff_own(prev_phi,u,v,i,j,dX,dt,fluxlimiter):
	
	"""My own version of tvd laxwendroff based on the lectures og  Gilbert strang's'Mathematics for engineers part II ' lecture series """
	dx=dX[0]
	dy=dX[1]

	if(fluxlimiter==0):
		fluxlimiter=vanleer
	elif(fluxlimiter==1):
		fluxlimiter=minmod
	elif(fluxlimiter==2):
		fluxlimiter=MC
	elif(fluxlimiter==3):
		fluxlimiter=superbee

	infinity=float("inf")
	ilist=[i-2,i-1,i+1,i+2]
	jlist=[j-2,j-1,j+1,j+2]
	if(prev_phi[i][j]==infinity):
		return 0

	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			else:
				if(prev_phi[ia][j]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

		except:
			if(prev_phi[ia][j]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			else:
				if(prev_phi[i][ja]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

		except:
			if(prev_phi[i][ja]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)


	rx=dt/dx
	ry=dt/dy

	uww=u[i][j-2]
	uw=u[i][j-1]
	ue=u[i][j]
	uee=u[i][j+1]
	denominator=prev_phi[i][j+1]-prev_phi[i][j]
	if(denominator):
		rjplus=(prev_phi[i][j+2]-prev_phi[i][j+1])/denominator
	else:
		rjplus=0

	denominator=prev_phi[i][j]-prev_phi[i][j-1]
	if(denominator):
		rj=(prev_phi[i][j+1]-prev_phi[i][j])/denominator
	else:
		rj=0

	denominator=prev_phi[i][j-1]-prev_phi[i][j-2]
	if(denominator):
		rjminus=(prev_phi[i][j]-prev_phi[i][j-1])/denominator
	else:
		rjminus=0

	phiE=0.5*(ue*(prev_phi[i][j+1]+prev_phi[i][j])-abs(ue)*(prev_phi[i][j+1]-prev_phi[i][j]))+abs(.5*ue*rx)*(fluxlimiter(rjplus)*(prev_phi[i][j+1]*.5*(ue+uee))-fluxlimiter(rj)*(prev_phi[i][j]*.5*(ue+uw)))
	phiW=0.5*(uw*(prev_phi[i][j]+prev_phi[i][j-1])-abs(uw)*(prev_phi[i][j]-prev_phi[i][j-1]))+abs(.5*uw*rx)*(fluxlimiter(rj)*(prev_phi[i][j]*.5*(ue+uw))-fluxlimiter(rjminus)*(prev_phi[i][j-1]*.5*(uww+uw)))

	vnn=(v[i+2][j-1]+v[i+2][j]+v[i+1][j-1]+v[i+1][j])/4
	vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4
	vss=(v[i-2][j-1]+v[i-1][j]+v[i-1][j-1]+v[i-2][j])/4

	denominator=prev_phi[i+1][j]-prev_phi[i][j]
	if(denominator):
		riplus=(prev_phi[i+2][j]-prev_phi[i+1][j])/denominator
	else:
		riplus=0

	denominator=prev_phi[i][j]-prev_phi[i-1][j]
	if(denominator):
		ri=(prev_phi[i+1][j]-prev_phi[i][j])/denominator
	else:
		ri=0

	denominator=prev_phi[i-1][j]-prev_phi[i-2][j]
	if(denominator):
		riminus=(prev_phi[i][j]-prev_phi[i-1][j])/denominator
	else:
		riminus=0

	phiS=0.5*(vs*(prev_phi[i][j]+prev_phi[i-1][j])-abs(vs)*(prev_phi[i][j]-prev_phi[i-1][j]))+abs(.5*vs*ry)*(fluxlimiter(riplus)*(prev_phi[i][j]*.5*(vs+vss))-fluxlimiter(ri)*(prev_phi[i-1][j]*.5*(vs+vn)))
	phiN=0.5*(vn*(prev_phi[i][j]+prev_phi[i+1][j])-abs(vn)*(prev_phi[i+1][j]-prev_phi[i][j]))+abs(.5*vn*ry)*(fluxlimiter(ri)*(prev_phi[i+1][j]*.5*(vn+vnn))-fluxlimiter(riminus)*(prev_phi[i][j]*.5*(vs+vn)))

	del_phi=-dt*(((phiE-phiW)/dx)+((phiN-phiS)/dy))

	return del_phi

def lax_wendroff_own(prev_phi,u,v,i,j,dX,dt,fluxlimiter):

	#Note:USE THIS FUNCTION .AS IT WORKS BEST.............

	"""My own version of  laxwendroff based on the lectures og  Gilbert strang's'Mathematics for engineers part II ' lecture series """


	dx=dX[0]
	dy=dX[1]

	if(fluxlimiter==0):
		fluxlimiter=vanleer
	elif(fluxlimiter==1):
		fluxlimiter=minmod
	elif(fluxlimiter==2):
		fluxlimiter=MC
	elif(fluxlimiter==3):
		fluxlimiter=superbee

	infinity=float("inf")
	ilist=[i-1,i+1]
	jlist=[j-1,j+1]
	if(prev_phi[i][j]==infinity):
		return 0
		
	for ia in ilist:
		try:
			if ia not in prev_phi.keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			else:
				if(prev_phi[ia][j]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

		except:
			if(prev_phi[ia][j]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

	
	for ja in jlist:
		try:
			if ja not in prev_phi[i].keys():
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

			else:
				if(prev_phi[i][ja]==infinity):
					return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)

		except:
			if(prev_phi[i][ja]==infinity):
				return upwind_scheme(prev_phi,u,v,i,j,dX,dt,fluxlimiter)


	rx=dt/dx
	ry=dt/dy

	uww=u[i][j-2]
	uw=u[i][j-1]
	ue=u[i][j]
	uee=u[i][j+1]

	denominator=(prev_phi[i][j+1]-prev_phi[i][j])
	if(denominator):
		Aplus=.5*(-(ue+uw)*prev_phi[i][j]+(ue+uee)*prev_phi[i][j+1])/denominator
	else:
		Aplus=0
	
	denominator=(prev_phi[i][j]-prev_phi[i][j-1])
	if(denominator):
		Aminus=.5*((ue+uw)*prev_phi[i][j]-(uww+uw)*prev_phi[i][j-1])/denominator
	else:
		Aminus=0

	denominator=prev_phi[i][j]-prev_phi[i][j-1]
	if(denominator):
		term=.5*((ue+uw)*prev_phi[i][j]-(uww+uw)*prev_phi[i][j-1])*(prev_phi[i][j+1]-prev_phi[i][j])/denominator
	else:
		#term=.5*((ue+uw)*prev_phi[i][j]-(uww+uw)*prev_phi[i][j-1])
		term=0
	Fx_plus=.5*(ue+uee)*prev_phi[i][j+1]-((rx)*(Aplus**2))*(prev_phi[i][j+1]-prev_phi[i][j])-rx*(-(ue+uw)*prev_phi[i][j]+(ue+uee)*prev_phi[i][j+1])*.5
	Fx_minus=.5*(uww+uw)*prev_phi[i][j-1]-((rx)*(Aminus**2))*(prev_phi[i][j]-prev_phi[i][j-1])-rx*term

	vnn=(v[i+2][j-1]+v[i+2][j]+v[i+1][j-1]+v[i+1][j])/4
	vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4
	vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4
	vss=(v[i-2][j-1]+v[i-1][j]+v[i-1][j-1]+v[i-2][j])/4

	denominator=prev_phi[i+1][j]-prev_phi[i][j]
	if(denominator):
		Bplus=.5*(-(vn+vs)*prev_phi[i][j]+(vn+vnn)*prev_phi[i+1][j])/denominator
	else:
		Bplus=0
	
	denominator=prev_phi[i][j]-prev_phi[i-1][j]
	if(denominator):
		Bminus=.5*((vn+vs)*prev_phi[i][j]-(vs+vss)*prev_phi[i-1][j])/denominator
	else:
		Bminus=0

	denominator=prev_phi[i][j]-prev_phi[i-1][j]
	if(denominator):
		term=.5*((vn+vs)*prev_phi[i][j]-(vs+vss)*prev_phi[i-1][j])*(prev_phi[i+1][j]-prev_phi[i][j])/denominator
	else:
		#term=.5*((vn+vs)*prev_phi[i][j]-(vs+vss)*prev_phi[i-1][j])
		term=0
	Fy_plus=.5*(vn+vnn)*prev_phi[i+1][j]-(ry*(Bplus**2))*(prev_phi[i+1][j]-prev_phi[i][j])-ry*(-(vn+vs)*prev_phi[i][j]+(vn+vnn)*prev_phi[i+1][j])*.5
	Fy_minus=.5*(vss+vs)*prev_phi[i-1][j]-(ry*(Bminus**2))*(prev_phi[i][j]-prev_phi[i-1][j])-ry*term

	del_phi=-.5*(rx*(Fx_plus-Fx_minus)+ry*(Fy_plus-Fy_minus))

	return del_phi
