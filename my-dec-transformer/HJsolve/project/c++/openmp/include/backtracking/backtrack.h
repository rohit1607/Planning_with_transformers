#ifndef BACKTRACK_H
#define BACKTRACK_H

#include "point.h"

// #include "matplotlibcpp.h"

// namespace plt=matplotlibcpp;

class Backtrack{
	public:
		vector<Point> path;
		vector<vector<vector<Point>>> zerocontour_list;
		vector<Point> backtracking();
		void plot_contours_with_path();

		Backtrack(Point starts,Point targets,Point dx,float DT,vector<vector<vector<Point>>>zcs,void(*read_vels_comp)(int,double*,double*,double*,int,int)){
			DX=dx;
			dt=DT;
			start=starts;
			target=targets;
			zerocontour_list=zcs;
			read_vels=read_vels_comp; 
		}
	private:
		vector<Point> projected_points;
		Point DX;
		float dt;
		Point start,target;
		double F[4],u[4],v[4];
		void(*read_vels)(int,double*,double*,double*,int,int);
		double slope(Point ,Point,double);
		pair<Point,pair<bool,bool>> get_intersection_pt(Edge e1,Edge e2,double tol);
		pair<bool,bool> check_point_inside_edge(Point,Edge);
		Point next_point_calc(Point,double,double,double,pair<double,double>);
		pair<double,double> get_direction(Edge,Point,bool);
		pair<double,double> get_direction_target(vector<vector<Point>>); 
		pair<pair<bool,Point>,pair<Edge,double>> get_min_projection(vector<vector<Point>>,Point);
		pair<pair<bool,Point>,pair<Edge,double>> get_min_projection2(vector<vector<Point>>,Point,Point);
		void approx_vels(Point,int,double*);
		void output_path();
};

#endif