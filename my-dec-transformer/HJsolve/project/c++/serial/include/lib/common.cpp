
#include "common.h"

Common_Params::Common_Params(int dimensions[2],Point DX){
            idim=dimensions[0];jdim=dimensions[1];
            dX=DX;
            phi=alloc_2d_int(idim,jdim);
            //rank=ranks;
            //active_proc_size=active_proc_sizes;
}