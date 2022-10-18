
#include "common.h"

Common_Params::Common_Params(int dimensions[2],pair<double,double> DX,vector<pair<int,int>> noentry){
            //cout<<"ffffffff"<<endl;
            idim=dimensions[0];jdim=dimensions[1];
            dX=DX;
            double infinity=numeric_limits<double>::infinity();
            data=new double[idim*jdim];
            fill(data,data+idim*jdim,infinity);
            phi=new double*[idim];
            for(int i=0;i<idim;i++){
            	phi[i]=&(data[jdim*i]);
            }
            //phi=alloc_2d_int(idim,jdim);
            noentry_points=noentry;
            //rank=ranks;
            //active_proc_size=active_proc_sizes;
            //cout<<"end common"<<endl;
}
Common_Params::~Common_Params(){
	delete [] data;
	delete phi;
}