
#include "advection.h"

double  MC(double r){
    return max(0.0,min(min((1+r)/2.0,2.0),2.0*r));
}

double vanleer(double r){
    return (r+abs(r))/(1+abs(r));
}

double minmod(double r){
    return max(0.0,min(1.0,r));
}

double superbee(double r){
    return max(max(0.0,min(2*r,1.0)),max(0.0,min(r,2.0)));
}

double upwind_scheme(double** prev_phi,double** u,double** v,int i,int j,Point dX,double dt,int fluxlimiter,int idim,int jdim){


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

    double ue=u[i][j];
    double uw=u[i][j-1];

    double phiE=0.5*(ue*(prev_phi[i][j+1]+prev_phi[i][j])-abs(ue)*(prev_phi[i][j+1]-prev_phi[i][j]));
    double phiW=0.5*(uw*(prev_phi[i][j]+prev_phi[i][j-1])-abs(uw)*(prev_phi[i][j]-prev_phi[i][j-1]));

    double vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4;
    double vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4;

    double phiS=0.5*(vs*(prev_phi[i][j]+prev_phi[i-1][j])-abs(vs)*(prev_phi[i][j]-prev_phi[i-1][j]));
    double phiN=0.5*(vn*(prev_phi[i][j]+prev_phi[i+1][j])-abs(vn)*(prev_phi[i+1][j]-prev_phi[i][j]));

    double del_phi=-dt*(((phiE-phiW)/dx)+((phiN-phiS)/dy));

    return del_phi;
}
double tvd_advection(double** prev_phi,double** u,double** v,int i,int j,Point dX,double dt,int fluxlimiter,int idim,int jdim){
    /*
    Reference:Matlab implemenataion of the advection term from
    2.29 Finite Volume MATLAB Framework Documentation
        Manual written by: Matt Ueckermann, Pierre Lermusiaux
        April 29, 2013

    */

    float dy=dX.second;
    float dx=dX.first;

    double infinity=std::numeric_limits<double>::infinity();
    if(i<1 || j<1 || i>idim-2 || j>jdim-2){
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
            return 0;
        }
    }

    for(int l=0;l<adjacent_num;l++){
        if(prev_phi[i][jlist[l]]==infinity){
            return 0;
        }
    }

    double (*fluxlimiter_fn)(double);
    if(fluxlimiter==0){
        fluxlimiter_fn=vanleer;
    }
    else if(fluxlimiter==1){
        fluxlimiter_fn=minmod;
    }
    else if(fluxlimiter==2){
        fluxlimiter_fn=MC;
    }
    else if(fluxlimiter==3){
        fluxlimiter_fn=superbee;
    }
    else{
        return 0;
    }



    double Rx_minus,Rx_plus,Ry_minus,Ry_plus;

    double denominator=u[i][j-1]*(prev_phi[i][j]-prev_phi[i][j-1]);
    if(denominator!=0){
        Rx_minus= ((.5* (u[i][j-1]+abs(u[i][j-1])) * ((prev_phi[i][j-1])+(prev_phi[i][j-2])) ) +(
               (.5* (u[i][j-1]-abs(u[i][j-1])) * ((prev_phi[i][j+1])+(prev_phi[i][j])) )))/(denominator);
    }
    else{
        Rx_minus=1;
    }

    double Fx_minus=u[i][j-1]*((prev_phi[i][j]+prev_phi[i][j-1])/2) - abs(
        u[i][j-1])* (.5*(prev_phi[i][j]-prev_phi[i][j-1]))*(1-(1-abs(u[i][j-1]*dt/dx))*fluxlimiter_fn(Rx_minus));

    denominator=u[i][j]*(prev_phi[i][j+1]-prev_phi[i][j]);
    if(denominator!=0){
        Rx_plus=((.5* (u[i][j]+abs(u[i][j])) * ((prev_phi[i][j])+(prev_phi[i][j-1])) )+(
        (.5* (u[i][j]-abs(u[i][j])) * ((prev_phi[i][j+1])+(prev_phi[i][j+2]))  )))/(denominator);
    }
    else{
        Rx_plus=1;
    }


    double Fx_plus=u[i][j]*((prev_phi[i][j]+prev_phi[i][j+1])/2) - abs(
        u[i][j])* ((prev_phi[i][j+1]-prev_phi[i][j])/2)*(1-(1-abs(u[i][j]*dt/dx))*fluxlimiter_fn(Rx_plus));


    double v_minus=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4;
    double v_plus=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4;

    denominator=v_minus*(prev_phi[i][j]-prev_phi[i-1][j]);
    if(denominator!=0){
        Ry_minus=((.5* (v_minus+abs(v_minus)) * ((prev_phi[i-1][j])+(prev_phi[i-2][j])) )+(
        (.5* (v_minus-abs(v_minus)) * ((prev_phi[i+1][j])+(prev_phi[i][j]))  )))/(denominator);
    }
    else{
        Ry_minus=1;
    }

    double Fy_minus=v_minus*((prev_phi[i][j]+prev_phi[i-1][j])/2) - abs(
        v_minus)* ((prev_phi[i][j]-prev_phi[i-1][j])/2)*(1-(1-abs(v_minus*dt/dy))*fluxlimiter_fn(Ry_minus));


    denominator=v_plus*(prev_phi[i+1][j]-prev_phi[i][j]);
    if(denominator!=0){
        Ry_plus=((.5* (v_plus+abs(v_plus)) * ((prev_phi[i][j])+(prev_phi[i-1][j])) )+(
        (.5* (v_plus-abs(v_plus)) * ((prev_phi[i+2][j])+(prev_phi[i+1][j]))) ) )/(denominator);
    }
    else{
        Ry_plus=1;
    }
    double Fy_plus=v_plus*((prev_phi[i+1][j]+prev_phi[i][j])/2) - abs(
        v_plus)* ((prev_phi[i+1][j]-prev_phi[i][j])/2)*(1-(1-abs(v_plus*dt/dy))*fluxlimiter_fn(Ry_plus));

    double del_phi= -dt*(((Fx_plus-Fx_minus)/dx)+((Fy_plus-Fy_minus)/dy));

    return del_phi;
}
double lax_wendroff_own(double** prev_phi,double** u,double** v,int i,int j,Point dX,double dt,int fluxlimiter,int idim,int jdim){
    //cout<<"lax wendroff"<<endl;
    float dy=dX.second;
    float dx=dX.first;

    double infinity=std::numeric_limits<double>::infinity();
    if(i<1 || j<1 || i>idim-2 || j>jdim-2){
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
            return 0;
        }
    }

    for(int l=0;l<adjacent_num;l++){
        if(prev_phi[i][jlist[l]]==infinity){
            return 0;
        }
    }

    double rx=dt/dx;
    double ry=dt/dy;

    double uww=u[i][j-2];
    double uw=u[i][j-1];
    double ue=u[i][j];
    double uee=u[i][j+1];

    double Aplus,Aminus,term,Fx_plus,Fx_minus,Bplus,Bminus,Fy_plus,Fy_minus;
    double denominator=(prev_phi[i][j+1]-prev_phi[i][j]);
    if(denominator){
        Aplus=.5*(-(ue+uw)*prev_phi[i][j]+(ue+uee)*prev_phi[i][j+1])/denominator;
    }
    else{
        Aplus=0;
    }

    denominator=(prev_phi[i][j]-prev_phi[i][j-1]);
    if(denominator){
        Aminus=.5*((ue+uw)*prev_phi[i][j]-(uww+uw)*prev_phi[i][j-1])/denominator;
    }
    else{
        Aminus=0;
    }

    denominator=prev_phi[i][j]-prev_phi[i][j-1];
    if(denominator){
        term=.5*((ue+uw)*prev_phi[i][j]-(uww+uw)*prev_phi[i][j-1])*(prev_phi[i][j+1]-prev_phi[i][j])/denominator;
    }
    else{
        term=0;
    }
    Fx_plus=.5*(ue+uee)*prev_phi[i][j+1]-((rx)*(Aplus*Aplus))*(prev_phi[i][j+1]-prev_phi[i][j])-rx*(-(ue+uw)*prev_phi[i][j]+(ue+uee)*prev_phi[i][j+1])*.5;
    Fx_minus=.5*(uww+uw)*prev_phi[i][j-1]-((rx)*(Aminus*Aminus))*(prev_phi[i][j]-prev_phi[i][j-1])-rx*term;

    double vnn=(v[i+2][j-1]+v[i+2][j]+v[i+1][j-1]+v[i+1][j])/4;
    double vn=(v[i+1][j-1]+v[i+1][j]+v[i][j-1]+v[i][j])/4;
    double vs=(v[i-1][j-1]+v[i-1][j]+v[i][j-1]+v[i][j])/4;
    double vss=(v[i-2][j-1]+v[i-1][j]+v[i-1][j-1]+v[i-2][j])/4;

    denominator=prev_phi[i+1][j]-prev_phi[i][j];
    if(denominator){
        Bplus=.5*(-(vn+vs)*prev_phi[i][j]+(vn+vnn)*prev_phi[i+1][j])/denominator;
    }
    else{
        Bplus=0;
    }

    denominator=prev_phi[i][j]-prev_phi[i-1][j];
    if(denominator){
        Bminus=.5*((vn+vs)*prev_phi[i][j]-(vs+vss)*prev_phi[i-1][j])/denominator;
    }
    else{
        Bminus=0;
    }
    denominator=prev_phi[i][j]-prev_phi[i-1][j];
    if(denominator){
        term=.5*((vn+vs)*prev_phi[i][j]-(vs+vss)*prev_phi[i-1][j])*(prev_phi[i+1][j]-prev_phi[i][j])/denominator;
    }
    else{
        term=0;
    }
    Fy_plus=.5*(vn+vnn)*prev_phi[i+1][j]-(ry*(Bplus*Bplus))*(prev_phi[i+1][j]-prev_phi[i][j])-ry*(-(vn+vs)*prev_phi[i][j]+(vn+vnn)*prev_phi[i+1][j])*.5;
    Fy_minus=.5*(vss+vs)*prev_phi[i-1][j]-(ry*(Bminus*Bminus))*(prev_phi[i][j]-prev_phi[i-1][j])-ry*term;

    double del_phi=-.5*(rx*(Fx_plus-Fx_minus)+ry*(Fy_plus-Fy_minus));
    //cout<<"lwo done"<<endl;
    return del_phi;
}
