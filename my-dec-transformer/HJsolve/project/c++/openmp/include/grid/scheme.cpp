#include "scheme.h"

double Scheme::level_set(double Fval,int i,int j,double** cphi){
    double level_set_part;
    if(order==2){
        level_set_part=second_order_scheme(cphi,Fval,i,j,dX,dt,idim,jdim);

    }
    else if(order==1){
        level_set_part=level_upwind_scheme(cphi,Fval,i,j,dX,dt,idim,jdim);
    }
    return level_set_part;
}
double Scheme::advection_part(int i,int j,double** cphi){
    double advection_val;
    if(advection_term==1){

        advection_val=upwind_scheme(cphi,u,v,i,j,dX,dt,flux_limiter,idim,jdim);
    }
    else if(advection_term==2){
        advection_val=tvd_advection(cphi,u,v,i,j,dX,dt,flux_limiter,idim,jdim);
    }
    else if(advection_term==9){
        advection_val=lax_wendroff_own(cphi,u,v,i,j,dX,dt,flux_limiter,idim,jdim);
    }
    return advection_val;
}
void Scheme::level_set_half_step(set<Point> nb_local){
    
    #pragma omp for schedule(dynamic,2) collapse(2)
    for(int i=0;i<idim;i++){
        for(int j=0;j<jdim;j++){
            cphi[i][j]=phi[i][j];
        }
    }
    for(auto it=nb_local.begin();it!=nb_local.end();++it){
       int  i=it->first;
        int j=it->second;

        if(i<idim-order-1 && i>=2 && j<jdim-order-1 && j>=2){

            double Fval=F[i][j];
            
            double val=.5*level_set(Fval,i,j,cphi)+cphi[i][j];
            
            phi[i][j]=val;
            
        }
        
    }
    
    
   
}
void Scheme::advection_half_step(set<Point> nb_local){
    
    #pragma omp for schedule(dynamic,2) collapse(2)
    for(int i=0;i<idim;i++){
        for(int j=0;j<jdim;j++){
        	cphi[i][j]=phi[i][j];
        }
    }
    for(auto it=nb_local.begin();it!=nb_local.end();++it){
        int i=it->first;
        int j=it->second;
    
        if(i<idim-order-1 && i>=2 && j<jdim-order-1 && j>=2){

            double val=advection_part(i,j,cphi)+cphi[i][j];

            phi[i][j]=val;

        }
        
    }

}