#ifndef CONTOURING_H
#define CONTOURING_H


#include "point.h"
#include <iomanip>
#include <unordered_set>
class Contouring{
    public:
        Contouring(double **cphi,set<Point> set_nband,vector<Point> vec_nband,int idimension,int jdimension,int num_threads){
            phi=cphi;
            s_narrowband=set_nband;
            v_narrowband=vec_nband;
            idim=idimension;
            jdim=jdimension;
            tot_threads=num_threads;
            //Edges_list=edges_list;
            //Edges_list.clear();
        }
        
        vector<vector< Point > > get_contours();
        list<Edge> Edges_list;
        void get_contour();


    private:
        double** phi;
        set<Point> s_narrowband;
        vector<Point> v_narrowband;
        int idim,jdim,tot_threads;
        vector<vector<Point> > zero_contours;
        Point linear_interpolation(Point p1,Point p2,double val1,double val2);
        void marching_squares(Point p1,Point p2,Point p3,Point p4,double val1,double val2,double val3,double val4);
        void contouring_grid();
        void connect_points();
};

#endif
