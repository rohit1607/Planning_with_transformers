#include "interp3d.h"

double trilinear_interpolation(double x,double y,double z,double x0,double x1,double y0,double y1,double z0,double z1,
	double c000,double c001,double c010,double c011,double c100,double c101,double c110,double c111){
    //cout<<c000<<endl;
	double denom=(x1-x0)*(y1-y0)*(z1-z0);
    double a0=(c000*x1*y1*z1-c001*x1*y1*z0-c010*x1*y0*z1+c011*x1*y0*z0-c100*x0*y1*z1+c101*x0*y1*z0+c110*x0*y0*z1-c111*x0*y0*z0)/denom;
    double a1=(-c000*y1*z1+c001*y1*z0+c010*y0*z1-c011*y0*z0+c100*y1*z1-c101*y1*z0-c110*y0*z1+c111*y0*z0)/denom;
    double a2=(-c000*x1*z1+c001*x1*z0+c010*x1*z1-c011*x1*z0+c100*x0*z1-c101*x0*z0-c110*x0*z1+c111*x0*z0)/denom;
    double a3=(-c000*x1*y1+c001*x1*y1+c010*x1*y0-c011*x1*y0+c100*x0*y1-c101*x0*y1-c110*x0*y0+c111*x0*y0)/denom;
    double a4=(c000*z1-c001*z0-c010*z1+c011*z0-c100*z1+c101*z0+c110*z1-c111*z0)/denom;
    double a5=(c000*y1-c001*y1-c010*y0+c011*y1-c100*y1+c101*y1+c110*y0-c111*y0)/denom;
    double a6=(c000*x1-c001*x1-c010*x1+c011*x0-c100*x0+c101*x0+c110*x0-c111*x0)/denom;
    double a7=(-c000+c001+c010-c011+c100-c101-c110+c111)/denom;
    double o=a0+a1*x+a2*y+a3*z+a4*x*y+a5*x*z+a6*y*z+a7*x*y*z;
    //cout<<o<<endl;
    return o;
}
void interp3d_fn(double* outputpt,double** points,double** values,int size,double* out){
    double x=outputpt[0];double y=outputpt[1];double z=outputpt[2];
    double x0=points[0][0];double x1=points[0][1];
    double y0=points[1][0];double y1=points[1][1];
    double z0=points[2][0];double z1=points[2][1];
    //cout<<(x,y,z,x0,y0,z0,x1,y1,z1)<<"tt"<<endl;
    x=(x-x0)/(x1-x0);
    y=(y-y0)/(y1-y0);
    z=(z-z0)/(z1-z0);
    for(int i=0;i<size;i++){
        double c000=values[0][i];
        double c001=values[1][i];
        double c010=values[2][i];
        double c011=values[3][i];
        double c100=values[4][i];
        double c101=values[5][i];
        double c110=values[6][i];
        double c111=values[7][i];
        //cout<<x<<" "<<y<<" "<<z<<" "<<x0<<" "<<x1<<" "<<y0<<" "<<y1<<" "<<z0<<" "<<z1<<" "<<c000<<" "<<c001<<" "<<c010<<" "<<c011<<" "<<c100<<" "<<c101<<" "<<c110<<" "<<c111<<" "<<endl;
        out[i]=trilinear_interpolation(x,y,z,x0,x1,y0,y1,z0,z1,c000,c001,c010,c011,c100,c101,c110,c111);
        
        out[i]=trilinear_interpolation(x,y,z,0,1,0,1,0,1,c000,c001,c010,c011,c100,c101,c110,c111);

        //cout<<i<<" "<<out[i]<<endl;
    }

}

void interp3d_parallel(double** outputpt,double*** points,double*** values,int size,int length,double** out,int num_threads){
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,int(length/num_threads))
        for (int i=0;i<length;i++){
            interp3d_fn(outputpt[i],points[i],values[i],size,out[i]);
        }
    }
    
}

void dummy_fn(double** arr,int size){
    for (int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            arr[i][j]=i*j;
        }
        
    }
}

void prev_nxt(double x,double res,double* out){
    double prev,nxt;
    if(x>0){
        prev=round(int(x/res)*res*100)/100;
        nxt=round(int((x/res)+1)*res*100)/100;
    }
    else{
        prev=round(int((x/res)-1)*res*100)/100;
        nxt=round(int((x/res))*res*100)/100;
    }
    out[0]=prev;
    out[1]=nxt;
    //cout<<out[0]<<" "<<out[1]<<endl;
    return;
}

void interpolation(int* narrowband_single,int nb_length,double** wilatsmap,
    int latsmap_length,double** wilonsmap,int lonsmap_length,
    int dwnld_offset,double ihrs,double prev_t,double next_t,double*** Grid,
    double*** heightall,double*** winds_u,double*** winds_v,
    double* prev_uv_single,double* next_uv_single,int m,int n,int p,double* output){
    //cout<<"ddd"<<endl;
    int** narrowband=new int*[nb_length];
    for(int i=0;i<nb_length;i++){
        narrowband[i]=&(narrowband_single[i*2]);
    }
    //wilonsmap //wilatsmap
    double*** prev_uv=new double**[m];
    double*** next_uv=new double**[m];
    for(int i=0;i<m;i++){
        prev_uv[i]=new double*[n];
        next_uv[i]=new double*[n];
        for(int j=0;j<n;j++){
            prev_uv[i][j]=&(prev_uv_single[i*n*p+j*p]);
            next_uv[i][j]=&(next_uv_single[i*n*p+j*p]);
            // for (int k=0;k<p;k++){
            //     prev_uv[i][j][k]=prev_uv_single[i*n*p+j*p+k];
            // }
            // for (int k=0;k<p;k++){
            //     next_uv[i][j][k]=next_uv_single[i*n*p+j*p+k];
            // }
        }
    }

    map<double,int> windlatsmap,windlonsmap;
    for(int i=0;i<latsmap_length;i++){
        windlatsmap.insert(pair<double,int>(wilatsmap[i][0],int(wilatsmap[i][1])));
    } 
    for(int i=0;i<lonsmap_length;i++){
        windlonsmap.insert(pair<double,int>(wilonsmap[i][0],int(wilonsmap[i][1])));
    }
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,4)
        for (int p=0;p<nb_length;p++){
            int i,j;
            
            i=narrowband[p][0];j=narrowband[p][1];
            double ilat,ilon;
            ilat=Grid[i][j][0];ilon=Grid[i][j][1];
            double h,wi_u,wi_v,wa_u,wa_v;

            double hrs[2];double lats[2];double lons[2];
            hrs[0]=prev_t;hrs[1]=next_t;
            prev_nxt(ilat,0.5,lats);
            prev_nxt(ilon,0.5,lons);
            //cout<<lats[0]<<" "<<lats[1]<<endl;
            //double*** uv_file;

            double ** vars=new double*[3];
            for (int ii=0;ii<3;ii++){
                vars[ii]=new double[2];
            }
            //double vars[3][2];
            //double fdata [8][5];

            double** fdata=new double*[8];
            for(int ii=0;ii<8;ii++){
                fdata[ii]=new double[5];
            }
            try{
                for(int ii=0;ii<2;ii++){
                    for(int jj=0;jj<2;jj++){
                        for(int kk=0;kk<2;kk++){
                            double hr,la,lo;
                            hr=hrs[ii];la=lats[jj];lo=lons[kk];
                            int hr_ind,la_ind,lo_ind;
                            hr_ind=int((hr - dwnld_offset)/3);

                            la_ind=windlatsmap.at(la);
                            lo_ind=windlonsmap.at(lo);
                            //cout<<la_ind<<" "<<lo_ind<<endl;
                            fdata[ii*4+jj*2+kk][0]=heightall[hr_ind][la_ind][lo_ind];
                            fdata[ii*4+jj*2+kk][1]=winds_u[hr_ind][la_ind][lo_ind];
                            fdata[ii*4+jj*2+kk][2]=winds_v[hr_ind][la_ind][lo_ind];
                            if(ii==0){
                                fdata[ii*4+jj*2+kk][3]=0.001*prev_uv[0][la_ind][lo_ind];
                                fdata[ii*4+jj*2+kk][4]=0.001*prev_uv[1][la_ind][lo_ind];

                            }
                            else{
                                fdata[ii*4+jj*2+kk][3]=0.001*next_uv[0][la_ind][lo_ind];
                                fdata[ii*4+jj*2+kk][4]=0.001*next_uv[1][la_ind][lo_ind];
                            }
                            
                            
                            //cout<<fdata[ii*4+jj*2+kk][0]<<" "<<fdata[ii*4+jj*2+kk][4]<<endl;
                        }
                    }
                }
                vars[0][0]=prev_t;vars[0][1]=next_t;
                vars[1][0]=lats[0];vars[1][1]=lats[1];
                vars[2][0]=lons[0];vars[2][1]=lons[1];
                double outputpt[3]={ihrs,ilat,ilon};
                interp3d_fn(outputpt,vars,fdata,5,&(output[p*5]));

            }
            catch(const out_of_range& oor){
                h=0;wi_u=0;wi_v=0;wa_u=0;wa_v=0;
                output[p*5+0]=h;
                output[p*5+1]=wi_u;
                output[p*5+2]=wi_v;
                output[p*5+3]=wa_u;
                output[p*5+4]=wa_v;
            }
            
        
        for (int ii=0;ii<3;ii++){
            delete[] vars[ii];
        }
        delete vars;
        for(int ii=0;ii<8;ii++){
            delete [] fdata[ii];
        }
        delete fdata;
        //delete uv_file;

        }
    }
    delete [] narrowband;
    for(int i=0;i<m;i++){
        delete[] prev_uv[i];
        delete[] next_uv[i];
    }
    delete[] prev_uv;delete[] next_uv;
    
}

