
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

void Narrowband::divide_contour(vector<vector<Point>> zero_contours){
    split_vec.clear();
    map<int,int> counter;

    for (auto it=zero_contours.begin();it!=zero_contours.end();++it){
        for(auto sit=(*it).begin();sit!=(*it).end();++sit){
            // if(rank==1)
            //     cout<<sit->first<<" "<<sit->second<<endl;
            counter[floor(sit->first)]+=1;
        }
    }
    int sum=0;
    for(auto it=counter.begin();it!=counter.end();++it){
        counter[it->first]+=sum;
        sum=counter[it->first];
    }

    int num_rows=counter.size();
    active_proc_size=min(int(floor((num_rows-1e-5)/4))+1,proc_size);
    //cout<<active_proc_size<<endl;
    int pts_per_proc=round(sum/active_proc_size);
    int rem=sum-pts_per_proc*active_proc_size;
    int ends=pts_per_proc;
    int appended=0;
    int imax,start;
    imax=counter.crbegin()->first;
    start=counter.begin()->first;
    for(auto it=counter.begin();it!=counter.end();++it){
        if((it->second>=ends)){
            if(appended==active_proc_size-1){
                split_vec.push_back(make_pair(start,min(imax+nb_width,idim-1)));
                break;
            }
            else{
                split_vec.push_back(make_pair(start,it->first));
                start=(it->first);
                ends+=pts_per_proc;
                appended+=1;
            }
        }
    }
    split_vec[0].first=max(split_vec[0].first-nb_width,0);
    //return procs_contour;
}


void Narrowband::signed_dist(map<int,map<int, int > > ray_hit_counters){
    //assign temp_phi to darray
    for(auto it=narrowband.begin();it!=narrowband.end();++it){
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
    for(auto it=narrowband.begin();it!=narrowband.end();++it){
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
    divide_contour(zerocontours);
    if(rank<active_proc_size){
        set<Point> nondz;
        map<int,set<int>> temp_nb;
        temp_nb.clear();
        map<int,map<int, int > > ray_hit_counters;
        int pimin,pimax;
        int nimin,nimax;
        nimin=split_vec[0].first;
        nimax=split_vec[active_proc_size-1].second;
        pimin=split_vec[rank].first;
        pimax=split_vec[rank].second;
        int simin,simax;
        simin=max(nimin,pimin-2);
        simax=min(nimax,pimax+2);
        
        for(auto it=zerocontours.begin();it!=zerocontours.end();++it){

            vector<Point>  zerocontour=*it;
            vector<Point>::iterator last_ptr;
            last_ptr=prev(zerocontour.end());
            if(zerocontour.begin()->first!=last_ptr->first && zerocontour.begin()->second!=last_ptr->second){
                last_ptr=next(last_ptr);
            }
            if(distance(zerocontour.begin(),last_ptr)>1){
                for(auto sit=zerocontour.begin();sit!=last_ptr;++sit){
                    Point pt=*sit;
                    
                    float i=pt.first;float j=pt.second;

                    int imin=max(0,int(ceil(i)-nb_width));
                    int imax=min(idim,int(floor(i)+nb_width+1));

                    for(int ia=max(imin,simin);ia<min(imax,simax+1);ia++){
                        int jdel=floor(pow(pow(nb_width,2)-pow(ia-i,2),.5));
                        int jmin=max(0,int(floor(j)-jdel));
                        int jmax=min(jdim,int(floor(j)+jdel+1));
                        for (int ja=jmin;ja<jmax;ja++){
                            temp_nb[ia].insert(ja);
                                if(ia>=pimin && ia<=pimax){
                                    narrowband.insert(make_pair(ia,ja));
                                    if(pow(pow(i-ia,2.0)+pow(j-ja,2.0),.5) <dz_dist){
                                        nondz.insert(make_pair(ia,ja));
                                    }
                                }
                                double dist=pow((pow(dX.second*(i-ia),2))+(pow(dX.first*(j-ja),2)),.5);
                                
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
            vector<Point>::iterator last_ptr;
            last_ptr=prev(zerocontour.end());
            if(zerocontour.begin()->first!=last_ptr->first && zerocontour.begin()->second!=last_ptr->second){
                last_ptr=next(last_ptr);
            }
            if(distance(zerocontour.begin(),last_ptr)>1){

                for(auto sit=zerocontour.begin();sit!=last_ptr;++sit){
                    Point pt=*sit;
                    double i=pt.first;
                    double j=pt.second;

                    Point next_pt=*next(sit,1);
                    double i_nxt=next_pt.first;
                    double j_nxt=next_pt.second;
                    Edge edge;
                    edge.one=pt;edge.two=next_pt;
                    double imin=min(i,i_nxt);double imax=max(i,i_nxt);

                    for (int ia=max(int(ceil(imin)),simin);ia<min(int(ceil(imax)),simax+1);ia++){
                        set<int > jlist=temp_nb[ia];

                        for(auto it=jlist.begin();it!=jlist.end();++it){
                            Point P;
                            P.first=ia;P.second=*it;
                            ray_hit_counters[ia][*it]+=ray_intersect_edge(&P,&edge);
                        }
                    }
                }
            }
        }
        signed_dist(ray_hit_counters);
    }
}