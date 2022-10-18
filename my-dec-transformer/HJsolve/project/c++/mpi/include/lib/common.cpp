
#include "common.h"

Common_Params::Common_Params(int dimensions[2],Point DX){
            idim=dimensions[0];jdim=dimensions[1];
            dX=DX;
            //cout<<"before knk"<<endl;
            double infinity=numeric_limits<double>::infinity();
    		 data=new double[idim*jdim];
			fill(data,data+idim*jdim,infinity);
			phi=new double*[idim];
			for (int i=0; i<idim; i++){
			    phi[i] = &(data[jdim*i]);

			}
            //cout<<"afterr knk"<<endl;
}
Common_Params::~Common_Params(){
	//for(int i=0;i<idim;i++){
		delete[] data;
	//}
}