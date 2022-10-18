#ifndef GRID_H
#define GRID_H


#include "narrowband.h"
#include "scheme.h"
#include "contouring.h"


class Grid:public Scheme,public Narrowband{
    private:
        Point start,target;
        double tol,dx,dy;
        bool targetInNarrowband,reached,backtracked;
        
        int max_timesteps,transfer_steps;
        
        
        
        vector<vector<vector<Point> > > zerocontourslist,temp_zcs;
        vector<vector<Point> > unjoined;
        vector<vector<Point> > get_init_zc();
        vector<vector<Point> > last_contour;
        vector<vector<int>> danger_signlist;
        
        double ***Fstore,***ustore,***vstore;
        int ** nb;
        
        double iotime=0;
        int bufferlength;
        void (*read_file_ptr)(int,int**,double**,double**,double**);
        void get_danger_zone_sign_list(vector<Point>);
        void check_target_in_narrowband(set<Point>);
        void serial_scheme(set<Point>);

        void send_join_contours_full(vector<vector<vector<Point> > >);
        void broadcast_last_contour();

        void distribute_speeds();

        void load_balance_narrowband();
    public:
        Grid (int dimensions[2],Point istart,Point itarget,int narb_width,float dz_per,pair<double,double> DX,float dt,int max_time,int no_threads,
            void(*read_file)(int,int**,double**,double**,double**),
        vector<pair<int,int>> noentry,int order=2,int advection_term=9,int flux_limiter=0,double tol=1e-8)
            : Scheme(dimensions,DX,dt,order,advection_term,flux_limiter,noentry),
            Narrowband(dimensions,narb_width,dz_per,DX,noentry),
            Common_Params(dimensions,DX,noentry){

            //cout<<"her"<<endl;

            targetInNarrowband=reached=backtracked=false;
            dx=DX.first;
            dy=DX.second;
            start=istart;target=itarget;
            read_file_ptr=(read_file);


            last_contour=get_init_zc();
            zerocontourslist.push_back(last_contour);
            max_timesteps=max_time;
            bufferlength=max_timesteps;
            
            nb=new int*[idim];
            for (int i=0;i<idim;i++){
                nb[i]=new int [jdim];
            }
           
           tot_threads=no_threads;
            int blocklen[2]={1,1};
            
           danger_signlist.resize(idim);
           for(int i=0;i<idim;i++){
                danger_signlist[i].resize(jdim);
           }
           
           
        }
        ~Grid(){

        }
        void main();
        void plot_contour(vector<vector<Point> > zero_contours,set<Point>);
        int get_time(){
            return timestep_count;
        }
        vector<vector<vector<Point> > > get_zerocontourslist(){
            return zerocontourslist;
        }
        double get_iotime(){
            return iotime;
        }
        
        void plot_contours();
        void update_ghost_cells(int n);
        void output_phi(set<Point> nb_local);
        void output_zerocontour(vector<vector<Point>> zcs);
};
#endif 
