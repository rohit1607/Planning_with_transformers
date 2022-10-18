#ifndef ADVECTION_H
#define ADVECTION_H

#include "point.h"


double upwind_scheme(double** prev_phi,double** u,double** v,int i,int j,Point dX,double dt,int fluxlimiter,int idim,int jdim);

double tvd_advection(double** prev_phi,double** u,double** v,int i,int j,Point dX,double dt,int fluxlimiter,int idim,int jdim);
double lax_wendroff_own(double** prev_phi,double** u,double** v,int i,int j,Point dX,double dt,int fluxlimiter,int idim,int jdim);

#endif 