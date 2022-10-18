
#ifndef NARROWBAND_H
#define NARROWBAND_H


#include <map>
#include <mpi.h>
#include "common.h"
class Narrowband:virtual public Common_Params{
    public:
        int nb_width;
        float dz_dist;
        vector<pair<int,int> > row_split_vec,col_split_vec;
        set<Point> danger_zone;
        int rank,active_proc_size,proc_size;
        int lrank,prev_active_proc;
        int procshape[2]={1,1};
        int prev_procshape[2];
        MPI_Comm lcomm;
        MPI_Comm comm_cart;
        int cart_rank[2];
        int row_wise,prev_row_wise;
        Narrowband(int dimensions[2],int narb_width,float dz_per,Point DX,int prank,int procsize):
        Common_Params(dimensions,dX){
            nb_width=narb_width;
            dz_dist=(1-dz_per)*nb_width;
            rank=prank;
            proc_size=procsize;
            prev_active_proc=active_proc_size=0;
            row_wise=1;

        }
        void narrowband_construction(vector<vector<Point> >);
    private:
        bool ray_intersect_edge( Point* ,struct Edge*,double);
        void signed_dist(map<int,map<int, int > > ray_hit_counters);
        void set_differences(set<Point>);
        void divide_contour(vector<vector<Point> > zero_contours);
        void split_communicator();

};
#endif
