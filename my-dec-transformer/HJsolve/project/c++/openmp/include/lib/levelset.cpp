#include "levelset.h"

double level_upwind_scheme(double** prev_phi, double Fij,int i,int j,Point dX,float dt,int idim,int jdim){

    float dy=dX.second;
    float dx=dX.first;

    double infinity=std::numeric_limits<double>::infinity();
    if(i<1 || j<1 || i>idim-2 || j>jdim-2){
        return 0;
    }
    if(prev_phi[i][j]==infinity){
        return 0;
    }
    const int adjacent_num=2;
    int ilist[adjacent_num]={i-1,i+1};
    int jlist[adjacent_num]={j-1,j+1};

    for(int l=0;l<adjacent_num;l++){
        if(prev_phi[ilist[l]][j]==infinity){
            return 0;
        }
    }

    for(int l=0;l<adjacent_num;l++){
        if(prev_phi[i][jlist[l]]==infinity){
            return 0;
        }
    }

    double Dijk_minusx=(prev_phi[i][j]-prev_phi[i][j-1])/dx;
    double Dijk_plusx=(prev_phi[i][j+1]-prev_phi[i][j])/dx;
    double Dijk_minusy=(prev_phi[i][j]-prev_phi[i-1][j])/dy;
    double Dijk_plusy=(prev_phi[i+1][j]-prev_phi[i][j])/dy;

    double del_plus=pow(pow(min(Dijk_plusx,0.0),2.0) + pow(max(Dijk_minusx,0.0),2.0) +
                pow(min(Dijk_plusy,0.0),2.0) + pow(max(Dijk_minusy,0.0),2.0) ,.5);

    double del_minus=pow(pow(max(Dijk_plusx,0.0),2.0) + pow(min(Dijk_minusx,0.0),2.0) +
                pow(max(Dijk_plusy,0.0),2.0) + pow(min(Dijk_minusy,0.0),2.0) ,.5);

    double del_phi=-dt*((max(Fij,0.0)* del_plus)+ (min(Fij,0.0)* del_minus));

    return del_phi;

}

double m(double x,double y){

    if(((x<=0 and y<=0) || (x>=0 and y>=0))){
        if(abs(x)<=abs(y)){
            return x;
        }
        else{
            return y;
        }
    }
    else{
        return 0;
    }
}
double second_order_scheme(double** prev_phi, double Fij,int i,int j,Point dX,float dt,int idim,int jdim){


    float dx=dX.first;
    float dy=dX.second;
    double infinity=std::numeric_limits<double>::infinity();

    

    if(i<2 || j<2 || i>idim-3 || j>jdim-3){
        return 0;
    }
    if(prev_phi[i][j]==infinity){
        return 0;
    }
    
    const int adjacent_num=4;
    int ilist[adjacent_num]={i-2,i-1,i+1,i+2};
    int jlist[adjacent_num]={j-2,j-1,j+1,j+2};

    for(int l=0;l<adjacent_num;l++){
        if(prev_phi[ilist[l]][j]==infinity){
            
            return level_upwind_scheme(prev_phi, Fij, i,j, dX, dt, idim, jdim);
        }
    }

    for(int l=0;l<adjacent_num;l++){
        if(prev_phi[i][jlist[l]]==infinity){
            return level_upwind_scheme(prev_phi, Fij, i,j, dX, dt, idim, jdim);
        }
    }
    

    double Dijk_minusx=(prev_phi[i][j]-prev_phi[i][j-1])/dx;
    double Dijk_plusx=(prev_phi[i][j+1]-prev_phi[i][j])/dx;
    double Dijk_minusy=(prev_phi[i][j]-prev_phi[i-1][j])/dy;
    double Dijk_plusy=(prev_phi[i+1][j]-prev_phi[i][j])/dy;

    double Dijk_plusx_plusx=(prev_phi[i][j+2] -2 * prev_phi[i][j+1] + prev_phi[i][j])/(dx*dx);
    double Dijk_plusx_minusx=(prev_phi[i][j+1] -2 * prev_phi[i][j] + prev_phi[i][j-1])/(dx*dx);
    double Dijk_minusx_minusx=(prev_phi[i][j] -2 * prev_phi[i][j-1] + prev_phi[i][j-2])/(dx*dx);
    double Dijk_minusx_plusx=Dijk_plusx_minusx;

    double Dijk_plusy_plusy=(prev_phi[i+2][j] -2* prev_phi[i+1][j] + prev_phi[i][j])/(dy*dy);
    double Dijk_plusy_minusy=(prev_phi[i+1][j] -2* prev_phi[i][j] + prev_phi[i-1][j])/(dy*dy);
    double Dijk_minusy_minusy=(prev_phi[i][j] -2* prev_phi[i-1][j] + prev_phi[i-2][j])/(dy*dy);
    double Dijk_minusy_plusy=Dijk_plusy_minusy;


    double A=Dijk_minusx + (dx)*m(Dijk_minusx_minusx,Dijk_plusx_minusx)/2;
    double B=Dijk_plusx -(dx)*m(Dijk_plusx_plusx,Dijk_plusx_minusx)/2;

    double C=Dijk_minusy + (dy)*m(Dijk_minusy_minusy,Dijk_plusy_minusy)/2;
    double D=Dijk_plusy -(dy)*m(Dijk_plusy_plusy,Dijk_plusy_minusy)/2;

    double del_plus=pow(pow(max(A,0.0),2.0) + pow(min(B,0.0),2.0) + pow(max(C,0.0),2.0) + pow(min(D,0.0),2.0),.5);
    double del_minus=pow((max(B,0.0),2.0) + pow(min(A,0.0),2.0) + pow(max(D,0.0),2.0) + pow(min(C,0.0),2.0),0.5);

    double del_phi=-dt*((max(Fij,0.0)* del_plus)+ (min(Fij,0.0)* del_minus));

    return del_phi;

}