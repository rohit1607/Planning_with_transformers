#ifndef COMMON_H
#define COMMON_H

#include "point.h"
#include "alloc.h"
#include <set>


class Common_Params{
    public:
        int idim,jdim;
        double** phi;
        double* data;
        vector <Point> narrowband;
        pair<double,double> dX;
        vector<pair<int,int> > row_split_vec;
        //int rank,active_proc_size;
        vector<pair<int,int>> noentry_points;
        Common_Params(int dimensions[2],pair<double,double> Dx,vector<pair<int,int>> noentry);
        ~Common_Params();
};
#endif
