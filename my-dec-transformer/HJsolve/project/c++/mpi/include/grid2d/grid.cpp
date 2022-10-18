#include "grid.h"

void Grid::update_ghost_cells(int n){
    int imin=row_split_vec[cart_rank[0]].first;
    int imax=row_split_vec[cart_rank[0]].second;
    int iMin=row_split_vec[0].first;
    int iMax=row_split_vec[procshape[0]-1].second;

    int jmin=col_split_vec[cart_rank[1]].first;
    int jmax=col_split_vec[cart_rank[1]].second;
    
    int jMin=col_split_vec[0].first;
    int jMax=col_split_vec[procshape[1]-1].second; 
    MPI_Request req;
    MPI_Status status;
    MPI_Datatype col,row;
    vector<int> row_ranklist,col_ranklist;
    for(int r=max(cart_rank[0]-2,0);r<min(cart_rank[0]+2,procshape[0]-1)+1;r++){
        if(r!=cart_rank[0]){
            row_ranklist.push_back(r);
        }
    }
    for(int r=max(cart_rank[1]-2,0);r<min(cart_rank[1]+2,procshape[1]-1)+1;r++){
        if(r!=cart_rank[1]){
            col_ranklist.push_back(r);
        }
    }

    
    
    for(vector<int>::iterator it=row_ranklist.begin();it!=row_ranklist.end();it++){
        int other_rrank=*it;
        int other_rank;
        int other_coords[2]={other_rrank,cart_rank[1]};
        MPI_Cart_rank(comm_cart,other_coords,&other_rank);
        int iomax,iomin;
        iomin=row_split_vec[other_rrank].first;
        iomax=row_split_vec[other_rrank].second;
        int jomin,jomax;
        jomin=col_split_vec[cart_rank[1]].first;
        jomax=col_split_vec[cart_rank[1]].second;
        int isendmin,count=0;
        for(int io=max(iMin,iomin-2);io!=min(iomax+2,iMax)+1;io++){
            
            if(!(io<=iomax && io>=iomin)){
                if(io>=imin && io<=imax){
                    if(count==0){
                        isendmin=io;
                    }
                    count++;
                    
                }
            }
        }
        if(count>0){
            MPI_Type_vector(count,jomax-jomin+1,jdim,MPI_DOUBLE,&row);
            MPI_Type_commit(&row);
            MPI_Isend(&phi[isendmin][jomin],1,row,other_rank,isendmin*n,lcomm,&req);
        }
    }
    vector<int> ilist;
    for(int i=max(iMin,imin-2);i!=min(imax+2,iMax)+1;i++){
        if(!(i<=imax && i>=imin)){
            ilist.push_back(i);
        }
    }
    for(vector<int>::iterator it=row_ranklist.begin();it!=row_ranklist.end();it++){
        int other_rrank=*it;
        int other_rank;
        int other_coords[2]={other_rrank,cart_rank[1]};
        MPI_Cart_rank(comm_cart,other_coords,&other_rank);
        int iomax,iomin;
        iomin=row_split_vec[other_rrank].first;
        iomax=row_split_vec[other_rrank].second;
        int jomin,jomax;
        jomin=col_split_vec[cart_rank[1]].first;
        jomax=col_split_vec[cart_rank[1]].second;
        int irecvmin,count=0;
        for (vector<int>::iterator it=ilist.begin();it!=ilist.end();it++){
            int i=*it;

            if(i>=iomin && i<=iomax ){
                if(count==0){
                    irecvmin=i;
                }
                count++;   
            }
        }
        if(count>0){
            MPI_Type_vector(count,jomax-jomin+1,jdim,MPI_DOUBLE,&row);
            MPI_Type_commit(&row);
            MPI_Irecv(&phi[irecvmin][jomin],1,row,other_rank,irecvmin*n,lcomm,&req);
            MPI_Wait(&req,&status);
        }
    }
    
    
    for(vector<int>::iterator it=col_ranklist.begin();it!=col_ranklist.end();it++){
        int other_crank=*it;
        int other_rank;
        int other_coords[2]={cart_rank[0],other_crank};
        MPI_Cart_rank(comm_cart,other_coords,&other_rank);
        int iomax,iomin;
        iomin=row_split_vec[cart_rank[0]].first;
        iomax=row_split_vec[cart_rank[0]].second;
        int jomin,jomax;
        jomin=col_split_vec[other_crank].first;
        jomax=col_split_vec[other_crank].second;
        int jsendmin,count=0;
        for(int jo=max(jMin,jomin-2);jo!=min(jomax+2,jMax)+1;jo++){
            if(!(jo<=jomax && jo>=jomin)){
                if(jo>=jmin && jo<=jmax){
                    if(count==0){
                        jsendmin=jo;

                    }
                    count++;
                    //MPI_Isend(&phi[iomin][jo],1,col,other_rank,jo*n,lcomm,&req);
                    
                    
                }
            }
        }

        if(count>0){
            MPI_Type_vector(iomax-iomin+1,count,jdim,MPI_DOUBLE,&col);
            MPI_Type_commit(&col);
            MPI_Isend(&phi[iomin][jsendmin],1,col,other_rank,jsendmin*n,lcomm,&req);
        }
    }
    vector<int> jlist;
    for(int j=max(jMin,jmin-2);j!=min(jmax+2,jMax)+1;j++){
        if(!(j<=jmax && j>=jmin)){
            jlist.push_back(j);
        }
    }
    for(vector<int>::iterator it=col_ranklist.begin();it!=col_ranklist.end();it++){
        int other_crank=*it;
        int other_rank;
        int other_coords[2]={cart_rank[0],other_crank};
        MPI_Cart_rank(comm_cart,other_coords,&other_rank);
        int iomax,iomin;
        iomin=row_split_vec[cart_rank[0]].first;
        iomax=row_split_vec[cart_rank[0]].second;
        int jomin,jomax;
        jomin=col_split_vec[other_crank].first;
        jomax=col_split_vec[other_crank].second;
        
        
        int jrecvmin,count=0;
        for (vector<int>::iterator it=jlist.begin();it!=jlist.end();it++){
            int j=*it;

            if(j>=jomin && j<=jomax ){
                //receive array
                if(count==0){
                        jrecvmin=j;

                }
                count++;
            }

        }
        if(count>0){
            MPI_Type_vector(iomax-iomin+1,count,jdim,MPI_DOUBLE,&col);
            MPI_Type_commit(&col);
            MPI_Irecv(&phi[iomin][jrecvmin],1,col,other_rank,jrecvmin*n,lcomm,&req);
            MPI_Wait(&req,&status);
        }
    }
    
    //cout<<"dddddddddd"<<endl;
}
void Grid::parallel_scheme(){
    
    update_ghost_cells(1);
    
    
    level_set_half_step();
    update_ghost_cells(2);
    
    advection_half_step();
    
    update_ghost_cells(3);
    
    level_set_half_step();
}
void Grid::distribute_speeds(){
    try{
        int r=(timestep_count/(int(max_timesteps/proc_size)));
        int timestart=r*(max_timesteps/proc_size)-timestep_count;

        transfer_steps=min(50,(r+1)*(max_timesteps/proc_size)-timestep_count);
       
        if(rank==r){
            int index;
            for(int t=0;t<transfer_steps;t++){
                for(int i=0;i<idim;i++){
                    for(int j=0;j<jdim;j++){
                        index=t*idim*jdim+i*jdim+j;
                        speeds[index]=Fstore[t][i][j];
                        speeds[bufferlength*idim*jdim+index]=ustore[t][i][j];
                        speeds[2*bufferlength*idim*jdim+index]=vstore[t][i][j];
                    }
                }
            }
        }
        MPI_Bcast(speeds,3*bufferlength*idim*jdim,MPI_DOUBLE,r,MPI_COMM_WORLD);
        for(int t=0;t<transfer_steps;t++){
            for(int i=0;i<idim;i++){
                for(int j=0;j<jdim;j++){
                    int index=t*idim*jdim+i*jdim+j;
                    F[t][i][j]=speeds[index];
                    u[t][i][j]=speeds[bufferlength*idim*jdim+index];
                    v[t][i][j]=speeds[2*bufferlength*idim*jdim+index];
                }
            }
        }
    }
    catch(...){
        cout<<"problem with distribute_speeds"<<endl;
        exit(3);
    }
}
void Grid::check_target_in_narrowband(){
    for(set<Point>::iterator it=narrowband.begin();it!=narrowband.end();++it){
        int i=it->first;
        int j=it->second;
        if(i==target.first && j==target.second){
            targetInNarrowband=true;
            return;
        }
    }
}
int sign(double x){
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
map<int,map<int,int> > Grid::get_danger_zone_sign_list(){
    map<int ,map<int,int> > danger_signlist;
    for(set<Point>::iterator it=danger_zone.begin();it!=danger_zone.end();++it){
        int i=it->first;
        int j=it->second;
        double val=phi[i][j];
        danger_signlist[i][j]=sign(val);
    }
    return danger_signlist;
}
void join_contours(vector<vector<Point> > unjoined ,vector<vector<Point> >& joined){
    double tole=1e-6;

    
    list<list<Point> > lunjoined;
    for(vector<vector<Point> >::iterator it=unjoined.begin();it!=unjoined.end();++it){
        list<Point> temp;
		if(!(*it).empty()){
        	for(vector<Point>::iterator sit=(*it).begin();sit!=(*it).end();++sit){
            	temp.push_back(*sit);
        	}
        	
        	lunjoined.push_back(temp);
		}
    }
   
    while(! lunjoined.empty()){
        list<Point> edge;
        edge=lunjoined.front();
        
        lunjoined.pop_front();
        if(!edge.empty()){
            
            bool connection=true;
            while(connection){
                connection=false;
                Point fpoint=edge.front();
                Point bpoint=edge.back();
                for(list<list<Point> >::iterator it=lunjoined.begin();it!=lunjoined.end();){
                    list<Point> li=(*it);
                    
                    Point fop=li.front();
                    Point bop=li.back();
                    if(abs(fop.first-fpoint.first)<tole && abs(fop.second-fpoint.second)<tole){
                        li.pop_front();
                        li.reverse();
                        edge.splice(edge.begin(),li);
                        it=lunjoined.erase(it);
                        connection=true;
                        break;
                    }
                    else if(abs(bop.first-fpoint.first)<tole && abs(bop.second-fpoint.second)<tole){
                        
                        li.pop_back();
                        edge.splice(edge.begin(),li);
                        connection=true;
                        it=lunjoined.erase(it);
                        break;
                    }
                    else if(abs(bop.first-bpoint.first)<tole && abs(bop.second-bpoint.second)<tole){
                        li.pop_back();
                        li.reverse();
                        edge.splice(edge.end(),li);
                        it=lunjoined.erase(it);
                        connection=true;
                        break;
                    }
                    else if(abs(fop.first-bpoint.first)<tole && abs(fop.second-bpoint.second)<tole){
                        li.pop_front();
                        edge.splice(edge.end(),li);
                        it=lunjoined.erase(it);
                        connection=true;
                        break;
                    }
                    else{
                        ++it;
                    }
                }
            }
            vector<Point> tempv;
            for(list<Point>::iterator it=edge.begin();it!=edge.end();it++){
                tempv.push_back(*it);
            }
            joined.push_back(tempv);
        }
    }

}
vector<vector<vector<Point> > > Grid::send_join_contours(vector<vector<vector<Point> > > zcs,int proc_dim_sz,MPI_Comm llcomm){
    float infinity=numeric_limits<float>::infinity();
    Point* temparr;
    Point* sizesarr;
    int m,n,p;
    m=zcs.size();
    n=0;
    p=0;
    for (int i=0;i<m;i++){
        int sz=zcs[i].size();
        if(sz>n){
            n=sz;
        }
        for(int j=0;j<sz;j++){
            int szi=zcs[i][j].size();
            if(szi>p){
                p=szi;
            }
        }
    }
    vector<vector<vector<Point> > > temp_zc;
    Point* arr=new Point[m*n*p];
    fill(arr,arr+m*n*p,make_pair(infinity,infinity));
    //cout<<"ff"<<endl;
    int i=0;
    for(vector<vector<vector<Point> > >::iterator it=zcs.begin();it!=zcs.end();++it){
        int j=0;
        for(vector<vector<Point> >::iterator sit=(*it).begin();sit!=(*it).end();++sit){
            int k=0;
            for(vector<Point>::iterator tit=(*sit).begin();tit!=(*sit).end();tit++){
                
                arr[n*p*i+p*j+k]=(*tit);
                k++;
            }
            j++; 
        }
        
        i++;
    }
    
    Point sizes;
    sizes.first=float(n);sizes.second=float(p);
    
    sizesarr=new Point[proc_dim_sz];
    
    MPI_Allgather(&sizes,1,point,sizesarr,1,point,llcomm);
    for(int i=0;i<proc_dim_sz;i++){
    }
    int *counts=new int[proc_dim_sz];
    
    for(int i=0;i<proc_dim_sz;i++){
        int val=m*(sizesarr[i].first)*(sizesarr[i].second);
        counts[i]=val;
    }
   
    int *disps=new int[proc_dim_sz];
    disps[0]=0;
    for (int i=1;i<proc_dim_sz;i++){
        disps[i]=disps[i-1]+counts[i-1];
        
    }
    
    temparr=new Point[disps[proc_dim_sz-1]+counts[proc_dim_sz-1]];

    MPI_Allgatherv(arr,m*n*p,point,temparr,counts,disps,point,llcomm);
    delete [] arr ;

    
    {
        
        double infinity=numeric_limits<double>::infinity();

        for (int i=0;i<m;i++){
            vector<vector<Point> > curves;
            for(int r=0;r<proc_dim_sz;r++){

                int nr,pr;
                nr=sizesarr[r].first;pr=sizesarr[r].second;
                int disp=disps[r];
                for(int j=0;j<nr;j++){
                    vector<Point> edges;
                    for (int k=0;k<pr;k++){
                        Point a=temparr[disp+i*nr*pr+j*pr+k];
                        
                        if(a.first==infinity && a.second==infinity){
                            break;
                        }
                        edges.push_back(a);
                    }
                    curves.push_back(edges);
                }
            }
            
            vector<vector<Point> > joined;
            join_contours(curves,joined);
            temp_zc.push_back(joined);
        }
    }
    
    delete [] temparr;
    delete [] sizesarr;
    delete [] counts;
    delete [] disps;
     return temp_zc;
    
}
void Grid::send_join_contours_full(vector<vector<vector<Point> > > zcs){
    MPI_Comm llcomm;
    int color;
    
    color=cart_rank[1];
    MPI_Comm_split(lcomm,color,lrank,&llcomm);
    vector<vector<vector<Point> > > temp_zc=send_join_contours(zcs,procshape[0],llcomm);
    MPI_Comm_free(&llcomm);
    
    color=cart_rank[0];
    MPI_Comm_split(lcomm,color,lrank,&llcomm);
    temp_zc=send_join_contours(temp_zc,procshape[1],llcomm);
    MPI_Comm_free(&llcomm);
    
    for(auto it=temp_zc.begin();it!=temp_zc.end();it++){
        zerocontourslist.push_back(*it);
    }
    last_contour=zerocontourslist.back();
}
vector<vector<Point > > Grid::get_init_zc(){
    
    set<Point> nband;
    for(int i=0;i<idim;i++){
        for(int j=0;j<jdim;j++){
            double val=pow(pow(dx*(i-start.first),2)+pow(dy*(j-start.second),2),.5)-pow(dx*dx+dy*dy,.5);
            phi[i][j]=val;
            nband.insert(make_pair(i,j));
        }
    }
    
    return Contouring(phi,nband,idim,jdim).get_contours();
}
void Grid::broadcast_last_contour(){
    last_contour.clear();
    //cout<<"last_contour"<<endl;
    Point* arr;
    float infinity=numeric_limits<float>::infinity();
    int sizes[2];
    if(rank%active_proc_size==0){
        vector<vector<Point> > lcontour=zerocontourslist.back();
        int m,n;
        m=lcontour.size();
        n=0;
        for(int i=0;i<m;i++){
            int sz=lcontour[i].size();
            if(sz>n) n=sz;
        }
        sizes[0]=m;sizes[1]=n;

        arr=new Point[m*n];
        fill(arr,arr+m*n,make_pair(infinity,infinity));
        for(int i=0;i<lcontour.size();i++){
            for(int j=0;j<lcontour[i].size();j++){
                arr[i*n+j]=lcontour[i][j];
            }
            
        }
    }
    MPI_Bcast(sizes,2,MPI_INT,0,lcomm);
    if(rank!=0){
        arr=new Point[sizes[0]*sizes[1]];
    }
    MPI_Bcast(arr,sizes[0]*sizes[1],point,0,lcomm);

    for(int i=0;i<sizes[0];i++){
        vector<Point> temp_vec;
        for(int j=0;j<sizes[1];j++){
            Point p=arr[i*sizes[1]+j];
            if(p.first==infinity && p.second==infinity){
                break;
            }
            temp_vec.push_back(p);
            
        }
        
        last_contour.push_back(temp_vec);
    }
    
}
void Grid::main(){
    map<int,map<int,int> > danger_signlist;
    bool reinitialise;
    distribute_speeds();
    while(!reached){
        if(timestep_count>=max_timesteps){
                    break;
            }
        
        narrowband_construction(last_contour);
        
        {
            
            danger_signlist=get_danger_zone_sign_list();
            
            check_target_in_narrowband();
            reinitialise=false;
            temp_zcs.clear();
            while(!reinitialise){
                
              
                if(timestep_count>=max_timesteps){
                    break;
                }
                if(iter_no>=bufferlength){
                    distribute_speeds();
                    iter_no=0;
                }
                parallel_scheme();

                timestep_count++;
                iter_no++;
                vector<vector<Point> > tempvec=Contouring(phi,narrowband,idim,jdim).get_contours();
                temp_zcs.push_back(tempvec);
               

                if(targetInNarrowband){
                    double targetphi=phi[int(target.first)][int(target.second)];
                    if(targetphi<=0){
                        reached=true;
                        reinitialise=true;
                    }
                }

                for(set<Point>::iterator it=danger_zone.begin();it!=danger_zone.end();it++){
                    int i=it->first;int j=it->second;
                    double val=phi[i][j];

                    if(danger_signlist[i][j]!=sign(val)){
                        reinitialise=true;
                        break;
                    }
                }

                int reduced[2]={reached,reinitialise};
                MPI_Allreduce(MPI_IN_PLACE,reduced,2,MPI_INT,MPI_SUM,lcomm);
                reached=reduced[0];reinitialise=reduced[1];
                
                if(reached || reinitialise){
                    send_join_contours_full(temp_zcs);
                        
                }
            } 
        }
        //cout<<rank<<"allreduced"<<endl;
        
    }
}
// void Grid::plot_contour(vector<vector<Point> > zero_contours){

//     for(auto it=zero_contours.begin();it!=zero_contours.end();++it){
//         vector<float> X,Y;
//         for(auto sit=(*it).begin();sit!=(*it).end();++sit){
//             X.push_back(sit->first);
//             Y.push_back(sit->second);
//         }
//         plt::plot(Y,X);
//     }
//     vector<float> nX,nY;
//     vector<float> pX,pY;
//     for(auto it=narrowband.begin();it!=narrowband.end();++it){

//         double val=phi[int(it->first)][int(it->second)];
//         if(val>0){
//             nX.push_back(it->first);
//             nY.push_back(it->second);
//         }
//         else if(val<0){
//             pX.push_back(it->first);
//             pY.push_back(it->second);
//         }


//     }
//     plt::scatter(nY,nX);
//     plt::scatter(pY,pX);

//    vector<float> dzX,dzY;
//    for(auto it=danger_zone.begin();it!=danger_zone.end();++it){
       
//            dzX.push_back(it->first);
//            dzY.push_back(it->second);
       
//    }
   

//     plt::title(to_string(timestep_count)+" "+to_string(rank));
//     plt::show();
// }
// void Grid::plot_contours(){
//     //if(rank==0){
//     for(auto it=zerocontourslist.begin();it!=zerocontourslist.end();++it){
//         for(auto sit=(*it).begin();sit!=(*it).end();++sit){
//             vector<float> X,Y;
//             for(auto thit=(*sit).begin();thit!=(*sit).end();++thit){
//                 X.push_back(thit->first);
//                 Y.push_back(thit->second);
//             }
//             plt::plot(Y,X);
//         }
//     }
//     plt::show();
// }
// void Grid::output_phi(int x){
//     ofstream outfile;
//     outfile.open("testlogs/"+to_string(timestep_count)+"-"+to_string(rank)+":"+to_string(x)+".txt",ios::out);
//     for(auto it=narrowband.begin();it!=narrowband.end();it++){
//         int i,j;
//         i=it->first;j=it->second;
//         outfile<<i<<","<<j<<"-"<<phi[i][j]<<endl;
//     }
//     outfile.close();
// }