
#ifndef NARROWBAND_H
#define NARROWBAND_H


#include <map>

#include "common.h"
class Narrowband:virtual public Common_Params{
    public:
        int nb_width;
        float dz_dist;
        //vector<pair<int,int> > row_split_vec,col_split_vec;
        set<Point> danger_zone;
        
        Narrowband(int dimensions[2],int narb_width,float dz_per,Point DX):
        Common_Params(dimensions,dX){
            nb_width=narb_width;
            dz_dist=(1-dz_per)*nb_width;
            
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
