#include "grid.h"


void Grid::serial_scheme(set<Point> nb_local){
    
    
    level_set_half_step(nb_local);
    #pragma omp barrier
    advection_half_step(nb_local);
    #pragma omp barrier
    level_set_half_step(nb_local);
    #pragma omp barrier
}

void Grid::check_target_in_narrowband(set<Point> narrowband_local){
    
    for(auto it=narrowband_local.begin();it!=narrowband_local.end();++it){
        int i=it->first;
        int j=it->second;
        if(i==target.first && j==target.second){
            targetInNarrowband=true;
            break;
        }
    }
}

int sign(double x){
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

void Grid::get_danger_zone_sign_list(vector<Point> danger_zone_local){
    
    
    for(auto it=danger_zone_local.begin();it<danger_zone_local.end();++it){
        int i=it->first;
        int j=it->second;
        double val=phi[i][j];
        
        danger_signlist[i][j]=sign(val);
    }
    
    
}

vector<vector<Point > > Grid::get_init_zc(){
    
    set<Point> nband;
    vector<Point> vband;
    
    for(int i=0;i<idim;i++){
        for(int j=0;j<jdim;j++){

            double val2=pow(pow(dy*(i-start.first),2)+pow(dx*(j-start.second),2),.5);
            double val=val2-3*pow(dx*dx+dy*dy,.5);
            phi[i][j]=val;
            
            nband.insert(make_pair(i,j));
            vband.push_back(make_pair(i,j));
        }
    }
    #pragma omp barrier
    if(noentry_points.size()){
        #pragma omp for schedule(dynamic,2)
        for(auto it=noentry_points.begin();it<noentry_points.end();it++){
            int i,j;
            i=it->first;j=it->second;
            phi[i][j]=pow(10.0,10.0);
        }
    }
   

    auto last_=Contouring(phi,nband,vband,idim,jdim,tot_threads).get_contours();
    
    double t1=omp_get_wtime();
    output_zerocontour(last_);
    iotime+=omp_get_wtime()-t1;
    return last_;}

void join_contours(vector<vector<Point> > unjoined ,vector<vector<Point> >& joined){
    double tole=1e-6;

    
    list<list<Point> > lunjoined;
    for(vector<vector<Point> >::iterator it=unjoined.begin();it!=unjoined.end();++it){
        list<Point> temp;
        for(vector<Point>::iterator sit=(*it).begin();sit!=(*it).end();++sit){
            temp.push_back(*sit);
        }
        lunjoined.push_back(temp);
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
            if(tempv.size()!=0)
            joined.push_back(tempv);
        }
    }
    //cout<<"done"<<endl;
    //return joined;

}
void Grid::main(){
    bool reinitialise=true;
    omp_set_num_threads(tot_threads); 
    int iter_count;
    
    #pragma omp parallel private(iter_count) 
    
    {
    int tid=omp_get_thread_num();
    
    
    vector<vector<vector<Point> > > temp_zcs;
    vector<Point> dz_local;set<Point> nb_local;vector<Point> vnb;
    for(iter_count=0;iter_count<max_timesteps;iter_count++){
        
        #pragma omp barrier
        if(reinitialise){  
            
            #pragma omp for schedule(dynamic,2)
            for (int i=0;i<idim;i++){
                for(int j=0;j<jdim;j++){
                    nb[i][j]=0;
                }
            }

            
            pair<set<Point > ,vector<Point> > temp=narrowband_construction(zerocontourslist.back());
            nb_local=temp.first;
            dz_local=temp.second;
            vnb.assign(nb_local.begin(),nb_local.end());
            
            get_danger_zone_sign_list(dz_local);
                
            check_target_in_narrowband(nb_local);
                
            reinitialise=false;
            temp_zcs.clear();
            for(auto it=nb_local.begin();it!=nb_local.end();it++){
                nb[int(it->first)][int(it->second)]=1;
            }
            

            #pragma omp barrier
        } 
        #pragma omp master
        {
            
            double t1=omp_get_wtime();
            read_file_ptr(iter_count,nb,F,u,v);
            iotime+=omp_get_wtime()-t1;
        }
        #pragma omp barrier
        
        serial_scheme(nb_local);
        

        auto contr=Contouring(phi,nb_local,vnb,idim,jdim,tot_threads).get_contours();
        
              
        
        #pragma omp critical
        {
            
            
            unjoined.insert(unjoined.end(),contr.begin(),contr.end());

            
        }

        #pragma omp barrier

        #pragma omp master
        {
            
            vector<vector<Point> > joined;
            joined.clear();
            join_contours(unjoined,joined);
            zerocontourslist.push_back(joined);
            
            timestep_count=iter_count;
            
            double t1=omp_get_wtime();
            output_zerocontour(joined);
            iotime+=omp_get_wtime()-t1;
            
            unjoined.clear();
            
            

        }

        if(targetInNarrowband){
            double targetphi=phi[int(target.first)][int(target.second)];
            if(targetphi<=0){
                reached=true;
                break;
            }
        }
            
        
        for(auto it=dz_local.begin();it<dz_local.end();++it){
            int i=it->first;int j=it->second;
            double val=phi[i][j];

            if(danger_signlist[i][j]!=sign(val)){

                reinitialise=true;
                
                break;
                
            }
        }
        #pragma omp barrier
        

    }
    }
   
}

// void Grid::plot_contour(vector<vector<Point> > zero_contours,set<Point> nb){

//     //cout<<"fff"<<endl;
//     plt::clf();
//     cout<<timestep_count<<" "<<zero_contours.size()<<endl;
//     int i=0;
//     for(auto it=zero_contours.begin();it!=zero_contours.end();++it){
//         vector<float> X,Y;
//         //cout<<i++<<" "<<(*it).size()<<endl;
//         for(auto sit=(*it).begin();sit!=(*it).end();++sit){

//             X.push_back(sit->first);
//             Y.push_back(sit->second);
//             cout<<(sit->first)<<" "<<(sit->second)<<endl;
//         }
//         plt::plot(Y,X);
//     }
//     cout<<endl;
//     //cout<<"plott"<<endl;
//     // vector<int> ilist,jlist;
//     // for(auto it=noentry_points.begin();it!=noentry_points.end();it++){
//     //     //cout<<it->first<<" "<<it->second<<endl;
//     //     ilist.push_back(it->first);
//     //     jlist.push_back(it->second);
//     // }
//     // plt::scatter(jlist,ilist);
//     vector<float> nX,nY;
//     vector<float> pX,pY;
//     for(auto it=nb.begin();it!=nb.end();++it){

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

//    // vector<float> dzX,dzY;
//    // for(auto it=dz.begin();it!=dz.end();++it){
       
//    //         dzX.push_back(it->first);
//    //         dzY.push_back(it->second);
       
//    // }
   

//     plt::title(to_string(timestep_count)+" ");
//     plt::save("/home/revanth/Mtech/bp/cpp2/openmp4/include/figures/"+to_string(timestep_count));
//     // if(timestep_count>100)
//     //plt::show();
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
void Grid::output_phi(set<Point> nb_local){
    ofstream outfile;
    outfile.open("/home/revanth/Mtech/projct/c++/openmp/include/testlogs/phis/"+to_string(timestep_count)+".txt",ios::out);
    for(auto it=nb_local.begin();it!=nb_local.end();it++){
        int i,j;
        i=it->first;j=it->second;
        outfile<<i<<","<<j<<"-   "<<phi[i][j]<<endl;
    }
    outfile.close();
}
void Grid::output_zerocontour(vector<vector<Point>> zcs){
    ofstream outfile;
    outfile.open("/home/revanth/Mtech/projct/c++/openmp/include/testlogs/zcs/"+to_string(timestep_count)+".txt",ios::out);
    int n=0;
    for(auto it=zcs.begin();it!=zcs.end();it++){
        n+=1; 
        for(auto sit=(*it).begin();sit!=(*it).end();sit++){
            outfile<<n<<")"<<sit->first<<","<<sit->second<<endl;
        }
        outfile<<endl;
    }
    outfile.close();
}