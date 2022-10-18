#include "grid.h"
void Grid::update_ghost_cells(){
    int imin,imax;
    imin=split_vec[rank].first;
    imax=split_vec[rank].second;
    int iMin,iMax;
    iMin=split_vec[0].first;
    iMax=split_vec[active_proc_size-1].second;
    MPI_Request req;
    MPI_Status status;

    vector<int> ranklist;
    for(int r=max(rank-2,0);r<min(rank+2,active_proc_size-1)+1;r++){
        if(r!=rank){
            ranklist.push_back(r);
        }
    }

    for(auto it=ranklist.begin();it!=ranklist.end();it++){
        int other_rank=*it;
        int iomax,iomin;
        iomin=split_vec[other_rank].first;
        iomax=split_vec[other_rank].second;
        //vector <int> iolist;
        for(int io=max(iMin,iomin-2);io!=min(iomax+2,iMax)+1;io++){
            if(!(io<=iomax && io>=iomin)){
                if(io>=imin && io<=imax){
                    //send array
                    MPI_Isend(&phi[io][0],jdim,MPI_DOUBLE,other_rank,io,lcomm,&req);
                    //cout<<endl;
                    //cout<<timestep_count<<" "<<rank<<" sent to "<<other_rank<<" - "<<io<<endl;
                    // for(int k=0;k<40;k++){
                    //     if(phi[io][k]!=numeric_limits<double>::infinity() && rank==1)
                    //     cout<<phi[io][k]<<" "<<io<<" "<<k<<endl;
                    // }

                }
            }
        }
    }
    vector<int> ilist;
    for(int i=max(iMin,imin-2);i!=min(imax+2,iMax)+1;i++){
        if(!(i<=imax && i>=imin)){
            ilist.push_back(i);
        }
    }
    for(auto it=ranklist.begin();it!=ranklist.end();it++){
        int other_rank=*it;

        int iomin,iomax;
        iomin=split_vec[other_rank].first;
        iomax=split_vec[other_rank].second;

        for (auto it=ilist.begin();it!=ilist.end();it++){
            int i=*it;

            if(i>=iomin && i<=iomax ){
                //receive array
                MPI_Irecv(&phi[i][0],jdim,MPI_DOUBLE,other_rank,i,lcomm,&req);
                MPI_Wait(&req,&status);
                //cout<<timestep_count<<' '<< rank<<" received from "<<other_rank<<" - "<<i<<endl;
                //cout<<i<<" "<<phi[i][27]<<" "<<rank<<endl;
                // for(int k=0;k<40;k++){
                //         if(phi[i][k]!=numeric_limits<double>::infinity() && rank==0)
                //         cout<<phi[i][k]<<" "<<i<<" "<<k<<endl;
                // }

            }
        }
    }
}

void Grid::parallel_scheme(){
    
    update_ghost_cells();
    
    level_set_half_step();
    update_ghost_cells();
    
    advection_half_step();
    
    update_ghost_cells();
    
    level_set_half_step();
}

void Grid::check_target_in_narrowband(){
    for(auto it=narrowband.begin();it!=narrowband.end();++it){
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
    for(auto it=danger_zone.begin();it!=danger_zone.end();++it){
        int i=it->first;
        int j=it->second;
        double val=phi[i][j];
        //danger_signlist.push_back(sign(val));
        danger_signlist[i][j]=sign(val);
    }
    return danger_signlist;
}
vector<vector<Point>> join_contours(vector<vector<Point>> unjoined ){
    double tole=1e-6;

    vector<vector<Point>> joined;
    list<list<Point>> lunjoined;
    //cout<<"inside joined contours"<<endl;
    for(auto it=unjoined.begin();it!=unjoined.end();++it){
        list<Point> temp;
        if(!(*it).empty()){
            for(auto sit=(*it).begin();sit!=(*it).end();++sit){
                temp.push_back(*sit);
            }
            
            lunjoined.push_back(temp);
        }
    }
    //cout<<"copied"<<endl;
    while(! lunjoined.empty()){
        list<Point> edge;
        edge=lunjoined.front();

        lunjoined.pop_front();

        bool connection=true;
        //cout<<"looping"<<endl;
        while(connection){
            connection=false;
            Point fpoint=edge.front();
            Point bpoint=edge.back();
            // cout<<fpoint.first<<" "<<fpoint.second<<endl;
            // cout<<bpoint.first<<" "<<bpoint.second<<endl;
            for(auto it=lunjoined.begin();it!=lunjoined.end();){
                list<Point> li=(*it);

                //cout<<"ffront"<<endl;
                Point fop=li.front();
                //cout<<"fb"<<endl;
                Point bop=li.back();
                //cout<<"back"<<endl;
                // if(fop.first==22 || bop.first==22 || fpoint.first==22 || bpoint.first==22){
                //     cout<<fop.first<<" "<<fop.second<<endl;
                //     cout<<bop.first<<" "<<bop.second<<endl;

                // }
                //cout<<endl;
                if(abs(fop.first-fpoint.first)<tole && abs(fop.second-fpoint.second)<tole){
                    li.pop_front();
                    //cout<<11<<endl;
                    li.reverse();
                    edge.splice(edge.begin(),li);
                    //cout<<"a0"<<endl;
                    it=lunjoined.erase(it);
                    //cout<<"b0"<<endl;
                    connection=true;
                    break;
                }
                else if(abs(bop.first-fpoint.first)<tole && abs(bop.second-fpoint.second)<tole){
                    //cout<<20<<endl;
                    // cout<<bop.first<<" "<<bop.second<<endl;
                    // cout<<fpoint.first<<" "<<fpoint.second<<endl;
                    li.pop_back();
                    edge.splice(edge.begin(),li);
                    connection=true;
                    //cout<<"a1"<<endl;
                    it=lunjoined.erase(it);
                    //cout<<"b1"<<endl;
                    break;
                }
                else if(abs(bop.first-bpoint.first)<tole && abs(bop.second-bpoint.second)<tole){
                    //cout<<30<<endl;
                    // cout<<bop.first<<" "<<bop.second<<endl;
                    // cout<<bpoint.first<<" "<<bpoint.second<<endl;
                    li.pop_back();
                    li.reverse();
                    edge.splice(edge.end(),li);
                    //cout<<"a2"<<endl;
                    it=lunjoined.erase(it);
                    connection=true;
                    //cout<<"b2"<<endl;
                    break;
                }
                else if(abs(fop.first-bpoint.first)<tole && abs(fop.second-bpoint.second)<tole){
                    //cout<<40<<endl;
                    // cout<<fop.first<<" "<<fop.second<<endl;
                    // cout<<bpoint.first<<" "<<bpoint.second<<endl;
                    li.pop_front();
                    edge.splice(edge.end(),li);
                    //cout<<"a4"<<endl;
                    it=lunjoined.erase(it);
                    connection=true;
                    //cout<<"b4"<<endl;
                    break;
                }
                else{
                    //cout<<"bbb"<<endl;
                    ++it;
                    //cout<<"fff"<<endl;
                }
            }
        }
        //cout<<"doneee"<<endl;
        vector<Point> tempv;
        for(auto it=edge.begin();it!=edge.end();it++){
            tempv.push_back(*it);
            //cout<<(*it).first<<" "<<(*it).second<<endl;
        }
        //copy(edge.begin(),edge.end(),tempv.begin());
        joined.push_back(tempv);
        //cout<<"doneee2"<<endl;
    }
    //cout<<"done"<<endl;
    return joined;

}
void Grid::send_join_contours(vector<vector<vector<Point>>> zcs){
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

    Point* arr=new Point[m*n*p];
    fill(arr,arr+m*n*p,make_pair(infinity,infinity));
    //cout<<"ff"<<endl;
    int i=0;
    for(auto it=zcs.begin();it!=zcs.end();++it){
        int j=0;
        for(auto sit=(*it).begin();sit!=(*it).end();++sit){
            int k=0;
            for(auto tit=(*sit).begin();tit!=(*sit).end();tit++){
                //arr[i][j][k++]=(*tit);
                arr[n*p*i+p*j+k]=(*tit);
                // if(i==0)
                // cout<<rank<<" "<<arr[n*p*i+p*j+k].first<<" "<<arr[n*p*i+p*j+k].second<<" "<<n*p*i+p*j+k<<endl;
                k++;
            }
            //cout<<endl;
            //copy((*sit).begin(),(*sit).end(),arr[i][j++][0]);
            j++; 
        }
        //cout<<endl;
        i++;
    }
    Point sizes={n,p};
    //cout<<n<<" "<<p<<endl;
    //cout<<"copied cnts"<<endl;
    sizesarr=new Point[active_proc_size];
    //cout<<m<<endl;
    MPI_Allgather(&sizes,1,point,sizesarr,1,point,lcomm);
    //cout<<"MPI_Allgather"<<endl;
    int *counts=new int[active_proc_size];
    //cout<<sizesarr[0].first<<" "<<sizesarr[0].second<<endl;
    for(int i=0;i<active_proc_size;i++){
        int val=m*(sizesarr[i].first)*(sizesarr[i].second);
        counts[i]=val;
        //cout<<counts[i]<<endl;
    }

    int *disps=new int[active_proc_size];
    disps[0]=0;
    for (int i=1;i<active_proc_size;i++){
        disps[i]=disps[i-1]+counts[i-1]; 
    }

    if(rank==0){
        temparr=new Point[disps[active_proc_size-1]+counts[active_proc_size-1]];
    }

    MPI_Gatherv(arr,m*n*p,point,temparr,counts,disps,point,0,lcomm);

    //cout<<"gathered"<<endl;
    if(rank==0){

        double infinity=numeric_limits<double>::infinity();

        for (int i=0;i<m;i++){
            vector<vector<Point>> curves;
            for(int r=0;r<active_proc_size;r++){

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
                        //cout<<a.first<<" "<<a.second<<" "<<i*nr*pr+j*pr+k<<endl;
                        edges.push_back(a);
                    }
                    curves.push_back(edges);
                }
            }
            //cout<<i<<endl;
            zerocontourslist.push_back(join_contours(curves));
        }
    }

    //cout<<"made it"<<endl;
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
    // phi=cphi.phi;
    // if(!slices.empty()){
    //     for(auto it=slices.begin();it!=slices.end();it++){
    //         int x=it->first;
    //         int y=it->second;
    //         phi.append(x,y,INFINITY);
    //     }
    // }
    
    return Contouring(phi,nband,idim,jdim).get_contours();
}
void Grid::broadcast_last_contour(){
    last_contour.clear();
    //cout<<"last_contour"<<endl;
    Point* arr;
    float infinity=numeric_limits<float>::infinity();
    int sizes[2];
    if(rank==0){
        vector<vector<Point>> lcontour=zerocontourslist.back();
        //cout<<timestep_count<<endl;
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
        //cout<<"start contour"<<endl;
        for(int i=0;i<lcontour.size();i++){
            for(int j=0;j<lcontour[i].size();j++){
                arr[i*n+j]=lcontour[i][j];
                //cout<<arr[i*n+j].first<<" "<<arr[i*n+j].second<<endl;
            }
            //cout<<endl;
        }
    }
    MPI_Bcast(sizes,2,MPI_INT,0,comm);
    if(rank!=0){
        arr=new Point[sizes[0]*sizes[1]];
    }
    MPI_Bcast(arr,sizes[0]*sizes[1],point,0,comm);

    for(int i=0;i<sizes[0];i++){
        vector<Point> temp_vec;
        for(int j=0;j<sizes[1];j++){
            Point p=arr[i*sizes[1]+j];
            if(p.first==infinity && p.second==infinity){
                break;
            }
            temp_vec.push_back(p);
            // if(rank==0)
            // cout<<p.first<<","<<p.second<<" ";
        }
        // if(rank==0)
        // cout<<endl;
        last_contour.push_back(temp_vec);
    }
    // cout<<"plot_contour"<<rank<<endl;
    // if(active_proc_size==3 && rank==0)
    // plot_contour(last_contour);
}

void Grid::main(){
    map<int,map<int,int>> danger_signlist;
    bool reinitialise;

    while(!reached){
        prev_active_proc=active_proc_size;
        
        narrowband_construction(last_contour);
        MPI_Barrier(comm);
        
        if(prev_active_proc!=active_proc_size){
            
            MPI_Comm_split(comm,(rank<active_proc_size),rank,&lcomm);
        }
        if(rank<active_proc_size){
            danger_signlist=get_danger_zone_sign_list();
            check_target_in_narrowband();
            MPI_Allreduce(MPI_IN_PLACE,&targetInNarrowband,1,MPI_INT,MPI_SUM,lcomm);
            reinitialise=false;
            temp_zcs.clear();
            while(!reinitialise){



                parallel_scheme();

                timestep_count++;
                vector<vector<Point>> tempvec=Contouring(phi,narrowband,idim,jdim).get_contours();
                
                temp_zcs.push_back(tempvec);
                
                if(targetInNarrowband){
                    double targetphi=phi[int(target.first)][int(target.second)];
                    if(targetphi<=0){
                        reached=true;
                        reinitialise=true;
                    }
                }

                for(auto it=danger_zone.begin();it!=danger_zone.end();++it){
                    int i=it->first;int j=it->second;
                    double val=phi[i][j];

                    if(danger_signlist[i][j]!=sign(val)){
                        reinitialise=true;
                        break;
                    }
                }

                MPI_Barrier(lcomm);
                int reduced[2]={reached,reinitialise};
                MPI_Allreduce(MPI_IN_PLACE,reduced,2,MPI_INT,MPI_SUM,lcomm);
                reached=reduced[0];reinitialise=reduced[1];
                if(reached || reinitialise){
                    
                    send_join_contours(temp_zcs);
                                    }
            } 
        }
        MPI_Barrier(comm);
        MPI_Allreduce(MPI_IN_PLACE,&reached,1,MPI_INT,MPI_SUM,comm);

        broadcast_last_contour();
        MPI_Bcast(&timestep_count,1,MPI_INT,0,comm);
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
   

//     plt::title(to_string(timestep_count)+" ");
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