#include "backtrack.h"

double Backtrack::slope(Point point1,Point point2,double tol=1e-12){
	float ia,ja,ib,jb;
	ia=point1.first;ja=point1.second;
	ib=point2.first;jb=point2.second;

	double m;

	if(abs(ja-jb)>tol){
		m=((ia-ib)/(ja-jb));
	}
	else{
		m=1/tol;
	}
	return m;
}
pair<Point,pair<bool,bool>> Backtrack::get_intersection_pt(Edge e1,Edge e2,double tol=1e-12){
	double m1=slope(e1.one,e1.two);
	double m2=slope(e2.one,e2.two);
	float y1,x1,b1,a1;
	y1=e1.one.first;x1=e1.one.second;
	b1=e2.one.first;a1=e2.one.second;
	Edge e;
	if(abs(m1-m2)>tol){
		float x=((b1-y1+m1*x1-m2*a1)/(m1-m2));
		float y=(y1+m1*(x-x1));
		pair<bool,bool> temp=check_point_inside_edge(make_pair(y,x),e1);
		bool inside_edge,is_vertex;
		inside_edge=temp.first;is_vertex=temp.second;
		if(inside_edge){
			return make_pair(make_pair(y,x),make_pair(true,is_vertex));
		}
		else{
			return make_pair(make_pair(y,x),make_pair(false,is_vertex));
		}
	}
	else{
		return make_pair(make_pair(numeric_limits<double>::infinity(),numeric_limits<double>::infinity()),make_pair(false,false));
	}
}

pair<bool,bool> Backtrack::check_point_inside_edge(Point p,Edge e){
	
	Point p1=e.one;Point p2=e.two;
	double dot_product=(p1.first-p.first)*(p.first-p2.first)+(p1.second-p.second)*(p.second-p2.second);

	if(dot_product==0){
		return make_pair(true,true);
	}
	else if(dot_product>0){
		return make_pair(true,false);
	}
	else{
		return make_pair(false,false);
	}	
}

Point Backtrack::next_point_calc(Point p,double f,double u,double v,pair<double,double> f_dir){
	Point next;
	float ib=p.first;float jb=p.second;
	next.first=ib-dt*(v+f*f_dir.second)/DX.second;
	next.second=jb-dt*(u+f*f_dir.first)/DX.first;
	return next;
}

pair<double,double> Backtrack::get_direction(Edge e,Point p,bool is_vertex){
	double normal_m;
	Point pointa=e.one;Point pointb=e.two;
	if(not is_vertex){
		
		double m=slope(pointa,pointb);
		if(m){
			normal_m=-(1/m);
		}
		else{
			normal_m=1e12;
		}
	}
	else{
		double m1=slope(p,pointa);
		double m2=slope(p,pointb);
		double normal_m1,normal_m2;
		if(m1){
			normal_m1=(-1/m1);
		}
		else{
			normal_m1=1e12;
		}
		if(m2){
			normal_m2=(-1/m2);
		}
		else{
			normal_m2=1e12;
		}
		normal_m=(normal_m1+normal_m2)/2;
	}
	double cos_theta,sin_theta,denom;
	denom=pow(1+pow(normal_m,2.0),0.5);
	cos_theta=1/denom;
	sin_theta=normal_m/denom;
	if(normal_m>=1e10){
		cos_theta=0;
		sin_theta=1;
	}
	double dot_product=(p.second-start.second)*cos_theta*DX.second+(p.first-start.first)*sin_theta*DX.first;

	if(dot_product<0){
		cos_theta*=-1;sin_theta*=-1;
	}
	return make_pair(cos_theta,sin_theta);
}
pair<double,double> Backtrack::get_direction_target(vector<vector<Point>> zcs){
	double min_dist=numeric_limits<double>::infinity();
	Point min_point;
	Edge min_edge;
	for(auto it=zcs.begin();it!=zcs.end();it++){
		vector<Point> temp_vec=(*it);
		for(int i=0;i<temp_vec.size()-1;i++){
			float ia,ja;
			ia=temp_vec[i].first;ja=temp_vec[i].second;
			double dista=pow(pow(DX.second*(target.first-ia),2.0)+pow(DX.second*(target.second-ja),2.0),0.5);
			if(dista<min_dist){
				min_dist=dista;
				int min_ind=i;
				int ind_nxt=min_ind+1;
				int ind_prv=(min_ind-1)%(temp_vec.size());
				min_edge.one=temp_vec[ind_prv];
				min_edge.two=temp_vec[ind_nxt];
				min_point=temp_vec[i];
			}
		}
	}
	return get_direction(min_edge,min_point,true);
}
double bilinear_interpolation(float ia,float ja,double q1,double q2,double q3,double q4){
	int i_prv,i_nxt,j_prv,j_nxt;
	i_prv=floor(ia);i_nxt=ceil(ia);
	j_prv=floor(ja);j_nxt=ceil(ja);

	double out=q1*(j_nxt-ja)*(i_nxt-ia)+q2*(ja-j_prv)*(i_nxt-ia)+q3*(j_nxt-ja)*(ia-i_prv)+q4*(ja-j_prv)*(ia-i_prv);
	return out;
}

void Backtrack::approx_vels(Point p,int t,double* vels){
	float ia,ja;
	ia=p.first;ja=p.second;
	int i_prv,i_nxt,j_prv,j_nxt;
	i_prv=floor(ia);i_nxt=ceil(ia);
	j_prv=floor(ja);j_nxt=ceil(ja);

	read_vels(t,F,u,v,i_prv,j_prv); 
 
	if(i_prv==i_nxt && j_prv==j_nxt){
		vels[0]=F[0];
		vels[1]=u[0];
		vels[2]=v[0];
		
	}
	else
	{
		// vels[0]=bilinear_interpolation(ia,ja,F[t][i_prv][j_prv],F[t][i_prv][j_nxt],F[t][i_nxt][j_nxt],F[t][i_nxt][j_prv]);
		// vels[1]=bilinear_interpolation(ia,ja,u[t][i_prv][j_prv],u[t][i_prv][j_nxt],u[t][i_nxt][j_nxt],u[t][i_nxt][j_prv]);
		// vels[2]=bilinear_interpolation(ia,ja,v[t][i_prv][j_prv],v[t][i_prv][j_nxt],v[t][i_nxt][j_nxt],v[t][i_nxt][j_prv]);
		vels[0]=bilinear_interpolation(ia,ja,F[0],F[1],F[2],F[3]);
		vels[1]=bilinear_interpolation(ia,ja,u[0],u[1],u[2],u[3]);
		vels[2]=bilinear_interpolation(ia,ja,v[0],v[1],v[2],v[3]);
		
	}
	//return vels;	
}	

pair<pair<bool,Point>,pair<Edge,double>> Backtrack::get_min_projection(vector<vector<Point>>zcs,Point proj_pt){
	float ip,jp;
	ip=proj_pt.first;jp=proj_pt.second;
	double min_dist=numeric_limits<double>::infinity();
	Point min_point=proj_pt;
	bool is_vertex=true;
	Edge e;
	e.one.first=e.one.second=e.two.first=e.two.second=numeric_limits<float>::infinity();

	for(auto it=zcs.begin();it!=zcs.end();it++){
		vector<Point> temp_vec=(*it);
		for(int i=0;i<temp_vec.size()-1;i++){
			int i_nxt=i+1;
			float ia,ja,ib,jb,ic,jc;

			ia=temp_vec[i].first;ja=temp_vec[i].second;
			ib=temp_vec[i_nxt].first;jb=temp_vec[i_nxt].second;
			ic=(ia+ib)/2;jc=(ja+jb)/2;
			double dista=pow(pow(DX.second*(ip-ia),2.0)+pow(DX.first*(jp-ja),2.0),0.5);
			double distc=pow(pow(DX.second*(ip-ic),2.0)+pow(DX.first*(jp-jc),2.0),0.5);

			if(min_dist>min(dista,distc)){
				min_dist=min(distc,dista);
				if(distc<dista){
					min_point.first=ic;min_point.second=jc;
					is_vertex=false;
					e.one=make_pair(ia,ja);
					e.two=make_pair(ib,jb);
				}
				else{
					min_point.first=ia;min_point.second=ja;
					is_vertex=true;
					int i_prv=(i-1)%temp_vec.size();
					e.one=temp_vec[i_prv];
					e.two=make_pair(ib,jb);
				}
			}

		}
	}
	return make_pair(make_pair(is_vertex,min_point),make_pair(e,min_dist));

}
pair<pair<bool,Point>,pair<Edge,double>> Backtrack::get_min_projection2(vector<vector<Point>> zcs,Point proj_pt,Point prev_pt){
	Edge path_edge;
	path_edge.one=prev_pt;
	path_edge.two=proj_pt;
	double min_dist=numeric_limits<float>::infinity();
	Point min_point=proj_pt;
	bool is_vertex=true;
	Edge e;
	e.one.first=e.one.second=e.two.first=e.two.second=numeric_limits<double>::infinity();
	for(auto it=zcs.begin();it!=zcs.end();it++){
		vector<Point> temp_vec=(*it);
		for(int i=0;i<temp_vec.size()-1;i++){
			int i_nxt=i+1;
			Edge cnt_edge;
			cnt_edge.one=temp_vec[i];
			cnt_edge.two=temp_vec[i_nxt];
			
			auto out=get_intersection_pt(cnt_edge,path_edge);
			bool intersects,is_avertex;
			intersects=out.second.first;is_avertex=out.second.second;
			Point intersect_pt=out.first;;
			if(intersects){
				double dist_err=pow(pow(DX.second*(proj_pt.first-intersect_pt.first),2.0)+pow(DX.first*(proj_pt.second-intersect_pt.second),2.0),0.5);
				if(min_dist>dist_err){
					min_point=intersect_pt;
					min_dist=dist_err;
					is_vertex=is_avertex;
					e=cnt_edge;
					if(is_vertex){
						int i_prv=(i-1)% (temp_vec.size());
						e.one=temp_vec[i_prv];
						e.two=temp_vec[i_nxt];
					}
				}
			}
		}
	}
	return make_pair(make_pair(is_vertex,min_point),make_pair(e,min_dist));

}

vector<Point> Backtrack::backtracking(){

	Point xt=target;
	vector<vector<Point>> zc=zerocontour_list.back();
	int timesteps=zerocontour_list.size()-3;
	path.push_back(xt);

	projected_points.push_back(xt);
	pair<double,double> fdir=get_direction_target(zc);
	double* vels=new double[3];
	Edge e;bool is_vertex;
	for(;timesteps>=0;timesteps--){
		approx_vels(xt,timesteps,vels);
		Point proj_pt=next_point_calc(xt,vels[0],vels[1],vels[2],fdir);
		
		projected_points.push_back(proj_pt);
		bool is_vertex1,is_vertex2;
		double min_dist1,min_dist2;
		Point xt1,xt2;
		Edge e1,e2;

		auto out1=get_min_projection(zerocontour_list[timesteps],proj_pt);
		auto out2=get_min_projection2(zerocontour_list[timesteps],proj_pt,xt);
		is_vertex1=out1.first.first;is_vertex2=out2.first.first;
		xt1=out1.first.second;xt2=out2.first.second;
		e1=out1.second.first;e2=out2.second.first;
		min_dist1=out1.second.second;min_dist2=out2.second.second;
		if((e2.one.first==numeric_limits<float>::infinity() && e2.one.second==numeric_limits<float>::infinity() &&e2.two.first==numeric_limits<float>::infinity() &&e2.two.second==numeric_limits<float>::infinity() ) || (min_dist2>2*min_dist1)){
			is_vertex=is_vertex1;
			xt=xt1;
			e=e1;
		}
		else{
			is_vertex=is_vertex2;
			xt=xt2;
			e=e2;
		}
		fdir=get_direction(e,xt,is_vertex);
				path.push_back(xt);
		
	}
	path.push_back(start);
	projected_points.push_back(start);
	output_path();
	return path;
}

// void Backtrack::plot_contours_with_path(){

// 	backtracking();

// 	for(auto it=zerocontour_list.begin();it!=zerocontour_list.end();++it){
//         for(auto sit=(*it).begin();sit!=(*it).end();++sit){
//             vector<float> X,Y;
//             for(auto thit=(*sit).begin();thit!=(*sit).end();++thit){
//                 X.push_back(thit->first);
//                 Y.push_back(thit->second);
//             }
//             plt::plot(Y,X);
//         }
//     }
//     vector<float> ilist,jlist,i2list,j2list;
//     for(auto it=path.begin();it!=path.end();it++){
//     	ilist.push_back(it->first);
//     	jlist.push_back(it->second);
//     }
//     for(auto it=projected_points.begin();it!=projected_points.end();it++){
//     	i2list.push_back(it->first);
//     	j2list.push_back(it->second);
//     }
//     //plt::plot(j2list,i2list);
//     plt::plot(jlist,ilist);
//     plt::show();
// }
void Backtrack::output_path(){
    ofstream outfile;
    //fill the output path folder here
    outfile.open("/home/revanth/Mtech/projct/c++/openmp/include/testlogs/zcs/path.txt",ios::out);
    for(auto it=path.begin();it!=path.end();it++){
        outfile<<it->first<<","<<it->second<<endl;
    }
    outfile.close();
}