
#include "narrowband.h"


bool Narrowband::ray_intersect_edge( Point* P,struct Edge* E,double tol=1e-12){
    double i=P->first;
    double j=P->second;
    double ia=E->one.first;
    double ja=E->one.second;
    double ib=E->two.first;
    double jb=E->two.second;
    bool intersect;
    double m_point,m_edge;

    if(ia>ib){
        swap(ia,ib);
        swap(ja,jb);
    }
    if(i==ia){
        i=ia+tol;
    }
    if(i==ib){
        i=ib+tol;
    }
    intersect=false;
    if((i>ib)||(i<ia)||(j>max(ja,jb))){
        return intersect;
    }
    if(j<min(ja,jb)){
        return true;
    }
    else{
        if(abs(ja-jb)>tol){
            m_edge=(ib-ia)/float(jb-ja);
        }
        else{
            m_edge=1/tol;
        }
        if(abs(ja-j)>tol){
            m_point=(i-ia)/float(j-ja);
        }
        else{
            m_point=1/tol;
        }
        intersect=(m_point>=m_edge);
        return intersect;
    }
}
int Narrowband::signs(double x){
    if(x==numeric_limits<double>::infinity()){
        return 5;
    }
    if(x>0){
        return 1;
    }
    if(x==0){
        return 0;
    }
    if(x<0){
        return -1;
    }
}
void Narrowband::signed_dist(map<int,map<int, int > > ray_hit_counters,set<Point> nb,int** signed_prev){
   for(auto it=nb.begin();it!=nb.end();++it){
        int i=it->first;
        int j=it->second;

        double val;
        if(signed_prev[i][j]==5){
            val=abs(phi[i][j])*(1-2*((ray_hit_counters[i][j]) %2));
        }
        else{
            val=signed_prev[i][j]*abs(phi[i][j]);
        }

        phi[i][j]=val;
    }
    for(auto it=nb.begin();it!=nb.end();it++){
        int i=it->first;
        int j=it->second;
        if(signed_prev[i][j]==5){
            if(i>1 && i<idim-1 ){
                if(signs(phi[i-1][j])==signs(phi[i+1][j]) && signs(phi[i+1][j])!=5){
                    if(signs(phi[i-1][j])!=signs(phi[i][j]))
                    {
                        phi[i][j]=signs(phi[i-1][j])*abs(phi[i][j]);
                        continue;
                    }
                    
                }
                
            }
        }
    }
    
    
}

vector<Point> Narrowband::set_differences(set<Point> set_nb_local,set<Point>non_dz){

    
    vector<Point> danger_zone_local;
    
    for(auto it=set_nb_local.begin();it!=set_nb_local.end();++it){
        Point elem=*it;
        
        if(non_dz.find(elem)==non_dz.end()){
            
            danger_zone_local.push_back(elem);
            
            
        }
    }
    return danger_zone_local;
}

void Narrowband::divide_contour(vector<vector<Point> > zerocontours){
    map<int,int> row_counter;
    row_counter.clear();
        
    row_split_vec.clear();


    for(auto i=0;i<zerocontours.size();i++){
        for(auto j=0;j<zerocontours[i].size();j++){
            int ia=floor(zerocontours[i][j].first);
            int ja=floor(zerocontours[i][j].second);
            row_counter[ia]+=1;
        }
    }


    int no_rows=row_counter.size();
    int rsum=0;
    
    for(map<int,int>::iterator it=row_counter.begin();it!=row_counter.end();++it){
            row_counter[it->first]+=rsum;
            rsum=row_counter[it->first];
    }
    int pts_per_thread=round(rsum/tot_threads);

    int ends=pts_per_thread;

    int appended=0;
    int imax,start;
    imax=row_counter.rbegin()->first;
    start=row_counter.begin()->first;


    for(auto it=row_counter.begin();it!=row_counter.end();++it){
        if((it->second>=ends)){
            if(appended==tot_threads-1){
                row_split_vec.push_back(make_pair(start,min(imax+nb_width,idim-1)));
                break;
            }
            row_split_vec.push_back(make_pair(start,(it->first)));
            start=(it->first);
            ends+=pts_per_thread;
            appended+=1;
            
        }
    }
    row_split_vec[0].first=max(row_split_vec[0].first-nb_width,0);
    int filler=min(imax+nb_width,idim-1);
    row_split_vec.back().second=filler;
    while(row_split_vec.size()!=tot_threads){
        row_split_vec.push_back(make_pair(0,-1));
    }

}


pair<set<Point > ,vector<Point> > Narrowband::narrowband_construction(vector<vector<Point> > zerocontours){
	set<Point> set_nb_local,nondz_local;
    vector<Point> narrowband_local;
    vector <set<int > >  temp_nb;
    temp_nb.resize(idim);
    narrowband_local.clear();
    map<int,map<int, int > > ray_hit_counters;
    ray_hit_counters.clear();


	#pragma omp single nowait
    {
    	divide_contour(zerocontours);

    }
    #pragma omp for schedule(dynamic,2) collapse(2)
    for(int i=0;i<idim;i++){
        for(int j=0;j<jdim;j++){
            signed_prev[i][j]=signs(phi[i][j]);
        }
    }


	#pragma omp single nowait
	{
	    for(int i=0;i<idim;i++){
	        fill(phi[i],phi[i]+jdim,numeric_limits<double>::infinity());
	    }
	}

	#pragma omp single 
	{
		narrowband.clear();
	    danger_zone.clear();
	    set_nb.clear();
	    nondz.clear();
        	
	}
    
    int timin=row_split_vec[omp_get_thread_num()].first;
    int timax=row_split_vec[omp_get_thread_num()].second;
    


	set_nb_local.clear();
    nondz_local.clear();
    

    for(auto it=zerocontours.begin();it!=zerocontours.end();++it){
        vector<Point>  zerocontour=*it;
        
        int vec_len=zerocontour.size();
        
        if(vec_len>5 ){
            for(int sit=0;sit<vec_len;sit++){
                Point pt=zerocontour[sit];
                double i=pt.first;double j=pt.second;


                int imin=max(0,int(ceil(i)-nb_width));
                int imax=min(idim,int(floor(i)+nb_width+1));
                int i1=max(imin,timin);int i2=min(imax,timax+1);
                
                for(int ia=i1;ia<i2;ia++){
                    int jdel=floor(pow(pow(nb_width,2)-pow(ia-i,2),.5));
                    int jmin=max(0,int(floor(j)-jdel));
                    int jmax=min(jdim,int(floor(j)+jdel+1));
                    for (int ja=jmin;ja<jmax;ja++){
                            
                            Point pt;
                            pt.first=ia;pt.second=ja;
                            temp_nb[ia].insert(int(ja));
                            set_nb_local.insert(pt);
                            
                            if(pow(pow(i-ia,2.0)+pow(j-ja,2.0),.5) <dz_dist){
                                nondz_local.insert(pt);
                            }

                            double dist=pow((pow(dX.second*(i-ia),2.0))+(pow(dX.first*(j-ja),2.0)),.5);
                            double val=min(abs(phi[ia][ja]),dist);
                            
                            phi[ia][ja]=val;
                    }
                }
            }
        }

    }


        
    vector<Point> danger_zone_local=set_differences(set_nb_local,nondz_local);

    
    int c=0;

    for(auto it=zerocontours.begin();it!=zerocontours.end();++it){
        vector<Point> zerocontour=*it;
        int vec_len=zerocontour.size();
        c+=1;
        if(vec_len>5 ){
            for(int sit=0;sit<vec_len-1;sit++){

                Point pt=zerocontour[sit];
                double i=pt.first;
                Point next_pt=zerocontour[sit+1];
                
                double i_nxt=next_pt.first;
                Edge edge;
                edge.one=pt;edge.two=next_pt;
                double imin=min(i,i_nxt);double imax=max(i,i_nxt);
                
                int i1=max(int(ceil(imin)),timin);int i2=min(int(ceil(imax)),timax+1);
                
                for (int ia=i1;ia<i2;ia++){
                    set<int > jlist=temp_nb[ia];

                    
                    for(auto it=jlist.begin();it!=jlist.end();++it){
                        Point P;
                        P.first=ia;P.second=*it;
                        int intersect=ray_intersect_edge(&P,&edge);
                        
                        
                        ray_hit_counters[ia][*it]+=intersect;

                        
                        
                    }
                    
                }
            }
        }
    }
    
   signed_dist(ray_hit_counters,set_nb_local,signed_prev);
    
    pair<set<Point > ,vector<Point> > temp;
    temp.first=set_nb_local;
    temp.second=danger_zone_local;
    return temp;
}
