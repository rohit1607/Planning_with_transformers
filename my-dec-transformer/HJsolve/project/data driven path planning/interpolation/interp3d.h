#ifndef INTERP_3D_H
#define INTERP_3D_H

#include <iostream>
#include <stdexcept> 
#include <map>
#include <math.h>
#include <cmath>
#include <omp.h>
using namespace std;

extern "C"
{
double trilinear_interpolation(double x,double y,double z,double x0,double x1,double y0,double y1,double z0,double z1,
	double c000,double c001,double c010,double c011,double c100,double c101,double c110,double c111);

void interp3d_fn(double* outputpt,double** points,double** values,int size,double* out);
void dummy_fn(double** arr,int size);
void interp3d_parallel(double** outputpt,double*** points,double*** values,int size,int length,double** out,int num_threads);
void interpolation(int* narrowband_single,int nb_length,double** wilatsmap,
    int latsmap_length,double** wilonsmap,int lonsmap_length,
    int dwnld_offset,double ihrs,double prev_t,double next_t,double*** Grid,
    double*** heightall,double*** winds_u,double*** winds_v,
    double* prev_uv_single,double* next_uv_single,int m,int n,int p,double* output);
}
#endif
