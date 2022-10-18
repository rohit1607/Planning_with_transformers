
#include "narrowband.h"


bool Narrowband::ray_intersect_edge( Point* P,struct Edge* E,double tol=1e-8){
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

void Narrowband::signed_dist(map<int,map<int, int > > ray_hit_counters){
    //assign temp_phi to darray
    for(set<Point>::iterator it=narrowband.begin();it!=narrowband.end();++it){
        int i=it->first;
        int j=it->second;
        double val=phi[i][j]*(1-2*(ray_hit_counters[i][j]%2));
        //temp_phi.append(i,j,val);
        phi[i][j]=val;
        // if (rank==1)
        // cout<<i<<" "<<j<<"    "<<val<<" "<<rank<<endl;
    }
}
void Narrowband::set_differences(set<Point>non_dz){

    // for(auto it=non_dz.begin();it!=non_dz.end();it++){
    //     cout<<it->first<<","<<it->second<<" ";
    // }
    // cout<<endl;
    // cout<<"danger_zone"<<endl;
    for(set<Point>::iterator it=narrowband.begin();it!=narrowband.end();++it){
        Point elem=*it;
        // if(rank==0)
        //cout<<rank<<" "<<elem.first<<","<<elem.second;
        if(non_dz.find(elem)==non_dz.end()){
            //cout<<" "<<elem.first<<","<<elem.second;
            danger_zone.insert(elem);
        }
        //cout<<endl;
    }
}

void Narrowband::narrowband_construction(vector<vector<Point> > zerocontours){
    phi=new double*[idim];
    for(int i=0;i<jdim;i++){
        phi[i]=new double[jdim];
        fill(phi[i],phi[i]+jdim,numeric_limits<double>::infinity());
    }
    narrowband.clear();
    danger_zone.clear();
    
    map<int,set<int> > temp_nb;
    set<Point> nondz;
        temp_nb.clear();
    map<int,map<int, int > > ray_hit_counters;
    for(auto it=zerocontours.begin();it!=zerocontours.end();++it){
        vector<Point>  zerocontour=*it;
        int vec_len=zerocontour.size();
        
        if(vec_len>1){
            for(int sit=0;sit<vec_len;sit++){
                Point pt=zerocontour[sit];
                double i=pt.first;double j=pt.second;


                int imin=max(0,int(ceil(i)-nb_width));
                int imax=min(idim,int(floor(i)+nb_width+1));

                for(int ia=imin;ia<imax;ia++){
                    int jdel=floor(pow(pow(nb_width,2)-pow(ia-i,2),.5));
                    int jmin=max(0,int(floor(j)-jdel));
                    int jmax=min(jdim,int(floor(j)+jdel+1));
                    for (int ja=jmin;ja<jmax;ja++){
                            narrowband.insert(make_pair(ia,ja));
                            Point pt;
                            pt.first=ia;pt.second=ja;
                            //cout<<ia<<" "<<ja<<endl;
                            temp_nb[ia].insert(ja);

                            if(pow(pow(i-ia,2.0)+pow(j-ja,2.0),.5) <dz_dist){
                                nondz.insert(pt);
                            }

                            double dist=pow((pow(dX.second*(i-ia),2.0))+(pow(dX.first*(j-ja),2.0)),.5);
                            double val=min(abs(phi[ia][ja]),dist);
                            
                            phi[ia][ja]=val;
                    }
                }
            }
        }

    }
    set_differences(nondz);


    for(auto it=zerocontours.begin();it!=zerocontours.end();++it){
        vector<Point> zerocontour=*it;
        int vec_len=zerocontour.size();
        
        if(vec_len>1){

            for(int sit=0;sit<vec_len-1;sit++){
                Point pt=zerocontour[sit];
                double i=pt.first;//float j=pt.y;
                Point next_pt=zerocontour[sit+1];
                double i_nxt=next_pt.first;//float j_nxt=next_pt.y;
                Edge edge;
                edge.one=pt;edge.two=next_pt;
                double imin=min(i,i_nxt);double imax=max(i,i_nxt);
                //cout<<pt.first<<" "<<pt.second<<endl;
                //cout<<next_pt.first<<" "<<next_pt.second<<endl;

                for (int ia=ceil(imin);ia<ceil(imax);ia++){
                    set<int > jlist=temp_nb[ia];

                    //cout<<ia<<endl;
                    for(auto it=jlist.begin();it!=jlist.end();++it){
                        Point P;
                        P.first=ia;P.second=*it;
                        ray_hit_counters[ia][*it]+=ray_intersect_edge(&P,&edge);
                        //cout<<ray_hit_counters[ia][*it]<<" "<<ia<<" "<<(*it)<<" "<<endl;
                    }
                    //cout<<endl;
                }
            }
        }
    }
    signed_dist(ray_hit_counters);
}
