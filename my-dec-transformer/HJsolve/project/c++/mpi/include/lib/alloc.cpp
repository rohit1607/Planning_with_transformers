#include "alloc.h"

double ** alloc_2d_int(int rows, int cols) {
    double infinity=numeric_limits<double>::infinity();
    double *data=new double[rows*cols];
    fill(data,data+cols*rows,infinity);
    double **arr=new double*[rows];
    for (int i=0; i<rows; i++){
        arr[i] = &(data[cols*i]);

   }
    return arr;
}

double*** alloc_3d_arr(int m,int n,int p){
    
    //double *data=new double[m*n*p];
    double*** arr=new double**[m];
    for(int i=0;i<m;i++){
        arr[i]=new double *[n];
        for(int j=0;j<n;j++){
            arr[i][j]=new double[p];
        }
    }
    return arr;
}