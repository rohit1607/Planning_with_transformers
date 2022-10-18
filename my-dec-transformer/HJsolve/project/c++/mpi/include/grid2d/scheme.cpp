#include "scheme.h"

double Scheme::level_set(double Fval,int i,int j){
    double level_set_part;
    if(order==2){
        level_set_part=second_order_scheme(phi,Fval,i,j,dX,dt,idim,jdim);

    }
    else if(order==1){
        level_set_part=level_upwind_scheme(phi,Fval,i,j,dX,dt,idim,jdim);
    }
    return level_set_part;
}
double Scheme::advection_part(int i,int j){
    double advection_val;
    if(advection_term==1){

        advection_val=upwind_scheme(phi,u[iter_no],v[iter_no],i,j,dX,dt,flux_limiter,idim,jdim);
    }
    else if(advection_term==2){
        advection_val=tvd_advection(phi,u[iter_no],v[iter_no],i,j,dX,dt,flux_limiter,idim,jdim);
    }
    else if(advection_term==9){
        //cout<<"fgggg"<<u[timestep_count][i][j]<<" "<<i<<" "<<j<<endl;
        advection_val=lax_wendroff_own(phi,u[iter_no],v[iter_no],i,j,dX,dt,flux_limiter,idim,jdim);
    }
    return advection_val;
}
void Scheme::level_set_half_step(){
    double** cphi=new double*[idim];
    for(int i=0;i<jdim;i++){
        cphi[i]=new double[jdim];
        fill(cphi[i],cphi[i]+jdim,numeric_limits<double>::infinity());
    }
    int i,j;
    double Fval,val;
    for(set<Point>::iterator it=narrowband.begin();it!=narrowband.end();++it){
        i=it->first;
        j=it->second;

        if(i<idim-order-1 && i>=2 && j<jdim-order-1 && j>=2){
            //cout<<timestep_count<<" "<<i<<" "<<j<<" "<<endl;

            Fval=F[iter_no][i][j];
            val=.5*level_set(Fval,i,j)+phi[i][j];
            //cphi.append(i,j,val);
            // if(i==24 && j==50){
            //     cout<<Fval<<" "<<val<<" "<<level_set(Fval,i,j)<<endl;
            // }
            cphi[i][j]=val;
        }
        else{
            val=phi[i][j];
            //cphi.append(i,j,val);
            cphi[i][j]=val;
        }
    }
    for(int i=0;i<idim;i++){
    for(int j=0;j<jdim;j++){
        phi[i][j]=cphi[i][j];
}
}
for (int i=0;i<idim;i++){
    delete[] cphi[i];
}
    delete cphi;
}
void Scheme::advection_half_step(){
    //double** cphi=alloc_2d_int(idim,jdim);
    double** cphi=new double*[idim];
    for(int i=0;i<jdim;i++){
        cphi[i]=new double[jdim];
        fill(cphi[i],cphi[i]+jdim,numeric_limits<double>::infinity());
    }
    int i,j;
    double val;
    for(set<Point>::iterator it=narrowband.begin();it!=narrowband.end();++it){
        i=it->first;
        j=it->second;

        if(i<idim-order-1 && i>=2 && j<jdim-order-1 && j>=2){

            val=advection_part(i,j)+phi[i][j];

            cphi[i][j]=val;

        }
        else{
            val=phi[i][j];
            cphi[i][j]=val;
        }

    }
    for(int i=0;i<idim;i++){
    for(int j=0;j<jdim;j++){
        phi[i][j]=cphi[i][j];
}
}
    for (int i=0;i<idim;i++){
        delete[] cphi[i];
    }
    delete cphi;
}