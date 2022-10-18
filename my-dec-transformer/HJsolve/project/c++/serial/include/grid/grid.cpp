#include "grid.h"


void Grid::serial_scheme(){
    
    
    level_set_half_step();
    
    
    advection_half_step();
    
    
    
    level_set_half_step();
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
        //danger_signlist.push_back(sign(val));
        danger_signlist[i][j]=sign(val);
    }
    return danger_signlist;
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

void Grid::main(){
    map<int,map<int,int>> danger_signlist;
    bool reinitialise=true;
    for(timestep_count=0;timestep_count<max_timesteps;timestep_count++){
            
            if(reinitialise){  
                    
                narrowband_construction(zerocontourslist.back());
                danger_signlist=get_danger_zone_sign_list();
                check_target_in_narrowband();
                reinitialise=false;
                
            }
            
            serial_scheme();
            
            
            zerocontourslist.push_back(Contouring(phi,narrowband,idim,jdim).get_contours());
            
            if(targetInNarrowband){
                double targetphi=phi[int(target.first)][int(target.second)];
                if(targetphi<=0){
                    reached=true;
                    
                }
            }
            
            
            if(reached)
                break;
            
            for(auto it=danger_zone.begin();it!=danger_zone.end();++it){
                int i=it->first;int j=it->second;
                double val=phi[i][j];

                if(danger_signlist[i][j]!=sign(val)){

                    reinitialise=true;
                    
                }
            }

        
    }
}
void Grid::plot_contour(vector<vector<Point> > zero_contours){

    for(auto it=zero_contours.begin();it!=zero_contours.end();++it){
        vector<float> X,Y;
        for(auto sit=(*it).begin();sit!=(*it).end();++sit){
            X.push_back(sit->first);
            Y.push_back(sit->second);
        }
        plt::plot(Y,X);
    }
    vector<float> nX,nY;
    vector<float> pX,pY;
    for(auto it=narrowband.begin();it!=narrowband.end();++it){

        double val=phi[int(it->first)][int(it->second)];
        if(val>0){
            nX.push_back(it->first);
            nY.push_back(it->second);
        }
        else if(val<0){
            pX.push_back(it->first);
            pY.push_back(it->second);
        }


    }
    plt::scatter(nY,nX);
    plt::scatter(pY,pX);

   vector<float> dzX,dzY;
   for(auto it=danger_zone.begin();it!=danger_zone.end();++it){
       
           dzX.push_back(it->first);
           dzY.push_back(it->second);
       
   }
   

    plt::title(to_string(timestep_count)+" ");
    plt::show();
}
void Grid::plot_contours(){
    //if(rank==0){
    for(auto it=zerocontourslist.begin();it!=zerocontourslist.end();++it){
        for(auto sit=(*it).begin();sit!=(*it).end();++sit){
            vector<float> X,Y;
            for(auto thit=(*sit).begin();thit!=(*sit).end();++thit){
                X.push_back(thit->first);
                Y.push_back(thit->second);
            }
            plt::plot(Y,X);
        }
    }
    plt::show();
}
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