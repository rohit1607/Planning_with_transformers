
#include "narrowband.h"

bool isPowerOfTwo (int x)  {  
    /* First x in the below expression is for the case when x is 0 */
    return x && (!(x&(x-1)));  
}
void Narrowband::split_communicator(){
    
    if(prev_active_proc!=active_proc_size){
            if(prev_active_proc!=0){
                MPI_Comm_free(&lcomm);
                MPI_Comm_free(&comm_cart);
               // cout<<"freed"<<endl;
            }

            int color=rank/active_proc_size;
            //cout<<color<<rank<<endl;

            MPI_Comm_split(MPI_COMM_WORLD,color,rank,&lcomm);
            //cout<<"yy"<<endl;
            MPI_Comm_rank(lcomm,&lrank);
            
        prev_active_proc=active_proc_size;
    }
    int periods[2]={0,0};
    int reorder=0;
    //cout<<"ggg"<<endl;
    MPI_Cart_create(lcomm,2,procshape,periods,reorder,&comm_cart);
    MPI_Cart_coords(comm_cart,lrank,2,cart_rank);

}
void Narrowband::divide_contour(vector<vector<Point> > zero_contours){
    row_split_vec.clear();
    col_split_vec.clear();
    map<int,int> row_counter,col_counter;

    for (vector<vector<Point> >::iterator it=zero_contours.begin();it!=zero_contours.end();++it){
        for(vector<Point>::iterator sit=(*it).begin();sit!=(*it).end();++sit){
            // if(rank==1)
                //cout<<rank<<" "<<sit->first<<" "<<sit->second<<endl;
            row_counter[floor(sit->first)]+=1;
            col_counter[floor(sit->second)]+=1;
        }
    }
    //cout<<row_counter.begin()->first<<"hhh"<<endl;
    int no_rows=row_counter.size();
    int no_cols=col_counter.size();

    int prev_rows_per_proc=no_rows/procshape[0];
    int prev_cols_per_proc=no_cols/procshape[1]; 
    bool row_wise;
    //split row wise
    if(prev_rows_per_proc>=prev_cols_per_proc){
        row_wise=1;
    }
    else{
        row_wise=0;
    }

    int rsum=0;
    int csum=0;
    for(map<int,int>::iterator it=row_counter.begin();it!=row_counter.end();++it){
            row_counter[it->first]+=rsum;
            rsum=row_counter[it->first];
    }

    for(map<int,int>::iterator it=col_counter.begin();it!=col_counter.end();++it){
            col_counter[it->first]+=csum;
            csum=col_counter[it->first];
    }
    prev_active_proc=active_proc_size;
    if(row_wise){
        int new_row_size=min(int(floor((no_rows-1e-5)/4))+1,proc_size/procshape[1]);
        if(!isPowerOfTwo(new_row_size)){
            new_row_size=pow(2,floor(log2(new_row_size)));
        }
        procshape[0]=new_row_size;
    }
    else{
        int new_col_size=min(int(floor((no_cols-1e-5)/4))+1,proc_size/procshape[0]);
        if(!isPowerOfTwo(new_col_size)){
            new_col_size=pow(2,floor(log2(new_col_size)));
        }
        procshape[1]=new_col_size;
    }
    active_proc_size=procshape[0]*procshape[1];

    int rows_per_proc=round(rsum/procshape[0]);
    int ends=rows_per_proc;
    int appended=0;
    int imax,start;
    imax=row_counter.rbegin()->first;
    start=row_counter.begin()->first;

    for(map<int,int>::iterator it=row_counter.begin();it!=row_counter.end();++it){
        if((it->second>=ends)){
            if(appended==procshape[0]-1){
                row_split_vec.push_back(make_pair(start,min(imax+nb_width,idim-1)));
                break;
            }
            else{
                row_split_vec.push_back(make_pair(start,it->first));
                start=(it->first);
                ends+=rows_per_proc;
                appended+=1;
            }
        }
    }
    //cout<<"mmm090"<<row_split_vec[0].first<<endl;
    row_split_vec[0].first=max(row_split_vec[0].first-nb_width,0);
    //cout<<"mmm"<<row_split_vec[0].first<<endl;
    int cols_per_proc=round(csum/procshape[1]);
    ends=cols_per_proc;
    appended=0;
    int jmax;
    jmax=col_counter.rbegin()->first;
    start=col_counter.begin()->first;
    for(map<int,int>::iterator it=col_counter.begin();it!=col_counter.end();++it){
        if((it->second>=ends)){
            if(appended==procshape[1]-1){
                col_split_vec.push_back(make_pair(start,min(jmax+nb_width,jdim-1)));
                break;
            }
            else{
                col_split_vec.push_back(make_pair(start,it->first));
                start=(it->first);
                ends+=cols_per_proc;
                appended+=1;
            }
        }
    }
    col_split_vec[0].first=max(col_split_vec[0].first-nb_width,0);
    
    
    
    
    //return procs_contour;
}
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
    //phi=alloc_2d_int(idim,jdim);
for(int i=0;i<idim;i++){
    for(int j=0;j<jdim;j++){
        phi[i][j]=numeric_limits<double>::infinity();
}
}
    narrowband.clear();
    danger_zone.clear();
    //cout<<"jnj"<<endl;
    divide_contour(zerocontours);
    //cout<<"hhh"<<endl;
    split_communicator();
    //cout<<"dividing"<<endl;
    // for(auto it=split_vec.begin();it!=split_vec.end();++it){
    //     cout<<it->first<<","<<it->second<<" ";
    // }
    // cout<<endl;
    //if(rank<active_proc_size){
        set<Point> nondz;
        map<int,set<int> > temp_nb;
        temp_nb.clear();
        map<int,map<int, int > > ray_hit_counters;
        
        int nimin=row_split_vec[0].first;
        int njmin=col_split_vec[0].first;
        int nimax=row_split_vec[procshape[0]-1].second;
        int njmax=col_split_vec[procshape[1]-1].second;
        int pimin=row_split_vec[cart_rank[0]].first;
        int pjmin=col_split_vec[cart_rank[1]].first;
        int pimax=row_split_vec[cart_rank[0]].second;
        int pjmax=col_split_vec[cart_rank[1]].second;
        
        int simin=max(nimin,pimin-2);
        int simax=min(nimax,pimax+2);

        int sjmin=max(njmin,pjmin-2);
        int sjmax=min(njmax,pjmax+2);
        //cout<<"finished dividing"<<endl;

        for(vector<vector<Point> >::iterator it=zerocontours.begin();it!=zerocontours.end();++it){

            vector<Point>  zerocontour=*it;
            int vec_len=zerocontour.size();
            
            if(vec_len>1){
                //for(auto sit=zerocontour.begin();sit!=last_ptr;++sit){
                for(int sit=0;sit<vec_len-1;sit++){
                    //Point pt=*sit;
                    // if(rank==1){
                    //cout<<pt.first<<" ,"<<pt.second<<endl;
                    // cout<<endl;
                    // }
                    Point pt=zerocontour[sit];
                    float i=pt.first;float j=pt.second;

                    int imin=max(0,int(ceil(i)-nb_width));
                    int imax=min(idim,int(floor(i)+nb_width+1));

                    for(int ia=max(imin,simin);ia<min(imax,simax+1);ia++){
                        int jdel=floor(pow(pow(nb_width,2)-pow(ia-i,2),.5));
                        int jmin=max(max(0,int(floor(j)-jdel)),sjmin);
                        int jmax=min(min(jdim,int(floor(j)+jdel+1)),sjmax);
                        for (int ja=jmin;ja<jmax;ja++){
                            temp_nb[ia].insert(ja);
                                if(ia>=pimin && ia<=pimax && ja>=pjmin && ja<=pjmax){
                                    narrowband.insert(make_pair(ia,ja));
                                    if(pow(pow(i-ia,2.0)+pow(j-ja,2.0),.5) <dz_dist){
                                        //cout<<rank<<" "<<ia<<" "<<ja<<" "<<dz_dist<<" "<<pow(pow(i-ia,2.0)+pow(j-ja,2.0),.5)<<endl;
                                        nondz.insert(make_pair(ia,ja));
                                    }
                                }
                                double dist=pow((pow(dX.second*(i-ia),2))+(pow(dX.first*(j-ja),2)),.5);
                                //if(rank==1)
                                //cout<<ia<<" "<<ja<<" "<<phi[ia][ja]<<" "<<dist<<endl;
                                double val=min(abs(phi[ia][ja]),dist);
                                //phi.append(ia,ja,val);

                                phi[ia][ja]=val;
                                

                        }
                    }
                }

            }
        }
        //cout<<"finished jhjh"<<endl;
        set_differences(nondz);

        //cout<<"setting"<<endl;

        for(vector<vector<Point> >::iterator it=zerocontours.begin();it!=zerocontours.end();++it){
            vector<Point>  zerocontour=*it;
            int vec_len=zerocontour.size();
            
                if(vec_len>1){
                //for(auto sit=zerocontour.begin();sit!=last_ptr;++sit){
                for(int sit=0;sit<vec_len-1;sit++){
                    //Point pt=*sit;
                    Point pt=zerocontour[sit];
                    double i=pt.first;
                    double j=pt.second;

                    //Point next_pt=*next(sit,1);
                    Point next_pt=zerocontour[sit+1];
                    double i_nxt=next_pt.first;
                    double j_nxt=next_pt.second;
                    Edge edge;
                    edge.one=pt;edge.two=next_pt;
                    double imin=min(i,i_nxt);double imax=max(i,i_nxt);
                    
                    for (int ia=max(int(ceil(imin)),simin);ia<min(int(ceil(imax)),simax+1);ia++){
                        set<int > jlist=temp_nb[ia];

                        for(set<int >::iterator it=jlist.begin();it!=jlist.end();++it){
                            Point P;
                            P.first=ia;P.second=*it;
                            ray_hit_counters[ia][*it]+=ray_intersect_edge(&P,&edge);
                            // if(rank==0 )
                            // cout<<ray_hit_counters[ia][*it]<<" "<<ia<<" "<<(*it)<<" "<<i<<" "<<i_nxt<<" "<<j<<" "<<j_nxt<<endl;
                        }
                    }
                }
            }
        }
        //cout<<"before sd"<<endl;
        signed_dist(ray_hit_counters);
        //cout<<"aftersd"<<endl;
    //}
}
