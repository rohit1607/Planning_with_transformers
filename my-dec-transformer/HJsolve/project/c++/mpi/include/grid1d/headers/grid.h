#ifndef GRID_H
#define GRID_H


#include "narrowband.h"
#include "scheme.h"
#include "contouring.h"

#include "matplotlibcpp.h"

namespace plt=matplotlibcpp;

class Grid:public Scheme,public Narrowband{
    private:
        Point start,target;
        double tol,dx,dy;
        bool targetInNarrowband,reached,backtracked;
        MPI_Comm comm;
        int max_timesteps,transfer_steps;
        //MPI_Comm lcomm;
        //int prev_active_proc;
        //int lrank;
        MPI_Datatype point;
        vector<vector<vector<Point> > > zerocontourslist,temp_zcs;
        vector<vector<Point> > get_init_zc();
        vector<vector<Point> > last_contour;
        double ***Fstore,***ustore,***vstore;
        double* speeds;
        int bufferlength;
        map<int,map<int,int> > get_danger_zone_sign_list();
        void check_target_in_narrowband();
        void parallel_scheme();
        vector<vector<vector<Point> > > send_join_contours(vector<vector<vector<Point> > > zc,int proc_dim_sz,MPI_Comm llcomm);
        void send_join_contours_full(vector<vector<vector<Point> > >);
        void broadcast_last_contour();
        void output_phi(int);
        void distribute_speeds();
        void load_balance_narrowband();
    public:
        Grid (int dimensions[2],Point istart,Point itarget,double*** Fl, double*** ul,
        double*** vl,int narb_width,float dz_per,Point DX,float dt,int prank,int proc_size,int max_time,MPI_Comm &icomm,
        int order=2,int advection_term=9,int flux_limiter=0,double tol=1e-8)
            : Scheme(dimensions,DX,dt,order,advection_term,flux_limiter),
            Narrowband(dimensions,narb_width,dz_per,DX,prank,proc_size),
            Common_Params(dimensions,DX),comm(icomm){

            targetInNarrowband=reached=backtracked=false;
            dx=DX.first;
            dy=DX.second;
            start=istart;target=itarget;
            last_contour=get_init_zc();
            zerocontourslist.push_back(last_contour);
            max_timesteps=max_time;
            bufferlength=50;
            speeds=new double[int(3*bufferlength*idim*jdim)];
            //lrank=rank;
            F=alloc_3d_arr(bufferlength,idim,jdim);
            u=alloc_3d_arr(bufferlength,idim,jdim);
            v=alloc_3d_arr(bufferlength,idim,jdim);
            int blocklen[2]={1,1};
            MPI_Aint disp[2];
            disp[0]=0;
            disp[1]=sizeof(float);
            Fstore=Fl;ustore=ul;vstore=vl;
            MPI_Datatype types[2]={MPI_FLOAT,MPI_FLOAT};
            MPI_Type_create_struct(2,blocklen,disp,types,&point);
            MPI_Type_commit(&point);

        }
        void main();
        void plot_contour(vector<vector<Point> > zero_contours);
        int get_time(){
            return timestep_count;
        }
        vector<vector<vector<Point> > > get_zerocontourslist(){
            return zerocontourslist;
        }
        void plot_contours();
        void update_ghost_cells(int n);
};
#endif 
