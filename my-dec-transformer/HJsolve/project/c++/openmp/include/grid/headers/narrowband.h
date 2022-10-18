
#ifndef NARROWBAND_H
#define NARROWBAND_H


#include <map>

#include "common.h"
class Narrowband:virtual public Common_Params{
    public:
        int nb_width;
        float dz_dist;

        vector<Point> danger_zone;
        set<Point> nondz,set_nb;
        
        int tot_threads;
        
        
        int useful_threads;
        int** signed_prev;
        Narrowband(int dimensions[2],int narb_width,float dz_per,Point DX,vector<pair<int,int>> noentry):
        Common_Params(dimensions,dX,noentry){
            //cout<<"NARROWBAND_H"<<endl;
            nb_width=narb_width;
            dz_dist=(1-dz_per)*nb_width;
            
            useful_threads=1;
            signed_prev=new int*[idim];
            for(int i=0;i<idim;i++){
                signed_prev[i]=new int [jdim];
            }
            //cout<<"NARROWBAND_H"<<endl;
            
        }
        pair<set<Point > ,vector<Point> > narrowband_construction(vector<vector<Point> >);
    private:
        bool ray_intersect_edge( Point* ,struct Edge*,double);
        void signed_dist(map<int,map<int, int > > ray_hit_counters,set<Point>,int**);
        vector<Point> set_differences(set<Point>,set<Point>);
        void divide_contour(vector<vector<Point> > zero_contours);
        void split_communicator();
        int signs(double x);

};

#endif
