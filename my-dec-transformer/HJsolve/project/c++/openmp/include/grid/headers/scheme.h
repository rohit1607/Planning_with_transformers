#ifndef SCHEME_H
#define SCHEME_H

#include "common.h"
#include "levelset.h"
#include "advection.h"

class Scheme:virtual public Common_Params{
        public:
            float dt;
            int order;
            int advection_term,flux_limiter;
            int timestep_count;
            int iter_no; 
            double level_set(double Fval,int i,int j,double** cphi);
            double advection_part(int i,int j,double** cphi);
            Scheme(int dimensions[2],pair<double,double> dX,float dtl,int orderl,
            int advection_terml,int fluxlimiterl,vector<pair<int,int>> noentry):Common_Params(dimensions,dX, noentry){

                //cout<<"SCHEME_H"<<endl;
                dt=dtl;
                order=orderl;
                advection_term=advection_terml;
                flux_limiter=fluxlimiterl;
                timestep_count=iter_no=0;

                cphi=new double*[idim];
                for(int i=0;i<idim;i++){
                    cphi[i]=new double[jdim];
                }
                F=new double*[idim];
                u=new double*[idim];
                v=new double*[idim];
                for(int i=0;i<idim;i++){
                    F[i]=new double[jdim];
                    
                    u[i]=new double[jdim];
                    v[i]=new double[jdim];
                }
                //cout<<"SCHEME_H"<<endl;
            }
            ~Scheme(){
                for(int i=0;i<idim;i++){
                    delete cphi[i];
                    delete F[i];
                    delete u[i];
                    delete v[i];
                }
                delete cphi;
                delete F,u,v;
            }
            double** F,**u,**v;
            double** cphi;
        public:
            void level_set_half_step(set<Point>);
            void advection_half_step(set<Point>);

};


#endif

