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
            double level_set(double Fval,int i,int j);
            double advection_part(int i,int j);
            Scheme(int dimensions[2],Point dX,float dtl,int orderl,
            int advection_terml,int fluxlimiterl):Common_Params(dimensions,dX){

                dt=dtl;
                order=orderl;
                advection_term=advection_terml;
                flux_limiter=fluxlimiterl;
                timestep_count=iter_no=0;
                //F=Fl;u=ul;v=vl;

            }
            double*** F,***u,***v;
        public:
            void level_set_half_step();
            void advection_half_step();

};


#endif
