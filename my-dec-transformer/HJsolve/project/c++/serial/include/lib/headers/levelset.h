#ifndef LEVELSET_H
#define LEVELSET_H
#include "point.h"

double level_upwind_scheme(double** prev_phi, double Fij,int i,int j,Point dX,float dt,int idim,int jdim);


double second_order_scheme(double** prev_phi, double Fij,int i,int j,Point dX,float dt,int idim,int jdim);

#endif