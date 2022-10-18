#ifndef CONTOURING_H
#define CONTOURING_H


#include "point.h"
#include <iomanip>
class Contouring{
    public:
        Contouring(double **cphi,set<Point> nband,int idimension,int jdimension){
            phi=cphi;
            narrowband=nband;
            idim=idimension;
            jdim=jdimension;
        }
        vector<vector< Point > > get_contours();
        list<Edge> Edges_list;
        void get_contour();


    private:
        double** phi;
        set<Point> narrowband;
        int idim,jdim;
        vector<vector<Point> > zero_contours;
        Point linear_interpolation(Point p1,Point p2,double val1,double val2);
        void marching_squares(Point p1,Point p2,Point p3,Point p4,double val1,double val2,double val3,double val4);
        void contouring_grid();
        void connect_points();
};

#endif
