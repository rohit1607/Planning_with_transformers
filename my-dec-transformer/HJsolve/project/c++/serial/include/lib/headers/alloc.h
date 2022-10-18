#ifndef ALLOC_H
#define ALLOC_H


#include <limits>
#include <iomanip>


using namespace std;
double **alloc_2d_int(int rows, int cols);
double*** alloc_3d_arr(int m,int n,int p);

#endif