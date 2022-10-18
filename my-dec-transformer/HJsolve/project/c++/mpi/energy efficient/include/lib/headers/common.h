#ifndef COMMON_H
#define COMMON_H

#include "point.h"
#include "alloc.h"
#include <set>

class Common_Params{
    public:
        int idim,jdim;
        double** phi;
        set <Point> narrowband;
        Point dX;
        //int rank,active_proc_size;
        Common_Params(int dimensions[2],Point Dx);
        
};
#endif