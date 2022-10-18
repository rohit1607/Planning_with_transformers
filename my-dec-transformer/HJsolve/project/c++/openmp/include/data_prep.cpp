
#include "data_prep.h"

void data_preparation(int idim,int jdim,void(*readfile)(int,int**,double**,double**,double**),void(*readvels)(int,double*,double*,double*,int,int),void(*plotfn)(int),int starti,int startj,int targeti,int targetj,int maxtimesteps,int nb_width,float dz_per,float dt,float dx,float dy,int num_threads,int** land=NULL,int land_length=0){

    Point start,target;
    pair<double,double> DX;
    start.first=starti;start.second=startj;
    target.first=targeti;target.second=targetj;
    DX.first=dx;DX.second=dy;

    cout<<idim<<" "<<jdim<<" "<<starti<<" "<<startj<<" "<<targeti<<targetj<<endl;
    int dimensions[2]={idim,jdim};
    
    vector<pair<int,int>> noentry={};
    if(land!=NULL){
        for(int i=0;i<land_length;i++){
            pair<int,int> temp;
            temp.first=land[i][0];
            temp.second=land[i][1];
            noentry.push_back(temp);
        }
    }
    cout<<"started it as "<<endl;
    double t1=omp_get_wtime();
    cout<<"jjj"<<endl;

        Grid g=Grid(dimensions,start,target,nb_width,dz_per,DX,dt,maxtimesteps,num_threads,readfile,noentry);
        cout<<"started"<<endl;
        g.main();
        t1=omp_get_wtime()-t1;
        cout<<"\nomp time in secs : "<<(float)t1<<endl;
        
        cout<<"io time is: "<<g.get_iotime()<<endl;
        cout<<"proc time is: "<<t1-g.get_iotime()<<endl;
        
        cout<<g.timestep_count<<endl;
        auto b=Backtrack(start,target,DX,dt,g.get_zerocontourslist(),readvels);
        vector<Point> li=b.backtracking();
        plotfn(g.timestep_count+1);
}



// int main(int argc,char** argv){
//     int idim,jdim;
//     idim=jdim=100;
    
//     int prank,psize;
//     prank=0;
//     psize=1;
//     Point start,target;
//     start=make_pair(20,20);
//     int nb_width=5;
//     float dz_per=0.5;
//     int num_threads=4;
//     if(argc==6){
//         //cout<<argv[1]<<argv[2]<<argv[3]<<endl;
//         idim=atoi(argv[1]);
//         jdim=atoi(argv[2]);
//         nb_width=atoi(argv[3]);
//         dz_per=stof(argv[4]);
//         num_threads=atoi(argv[5]);

//     }
    
//     //cout<<"args done"<<endl;
//     // int dims[14]={100,150,200,250,300,350,400,450,500,600,700,800,900,1000};
//     // for(auto mt=0;mt<14;mt++){
        
//        //  idim=jdim=dims[mt];
//         int max_timesteps=10*idim;
//         int dimensions[2]={idim,jdim};
//         double*** Fl,***ul,***vl;
        
//         target=make_pair(int(0.9*idim),int(0.9*jdim));

        
        
//         //cout<<"alloc_2d_int"<<endl;
//         Point DX=make_pair(1,1);
//         //double t1,t2;
        

//         int store_steps;
//         store_steps=max_timesteps;
        
        
//         //cout<<prank<<" "<<store_steps<<endl;
//         Fl=alloc_3d_arr(store_steps,idim,jdim);
//         ul=alloc_3d_arr(store_steps,idim,jdim);
//         vl=alloc_3d_arr(store_steps,idim,jdim);

//         for(int i=0;i<store_steps;i++){
//             for(int j=0;j<dimensions[0];j++){
//                 for(int k=0;k<dimensions[1];k++){   
//                     Fl[i][j][k]=1;
//                     ul[i][j][k]=0.0;
//                     vl[i][j][k]=0.0;
//                     if( j>=30 && j<.8*idim){
//                         ul[i][j][k]=0;
                  
//                     }
//                     if(k>=30 && k<.8*jdim){
//                         vl[i][j][k]=0;
//                     }
//                 }
//             }
//         }
//         vector<pair<int,int>> noentry;
//         for(int i=35;i<40;i++){
//             for (int j=40;j<45;j++){
//                 noentry.push_back(make_pair(i,j));
//             }
//             // for(int j=51;j<100;j++){
//             //     noentry.push_back(make_pair(i,j));
//             // }

//         }
//         // for(int k=0;k<store_steps;k++){
//         //     for(int i=35;i<40;i++){
//         //         for(int j=0;j<jdim;j++){
//         //                 ul[k][i][j]=0;
//         //                 vl[k][i][j]=0;
//         //             }
//         //     }
//         // }
//         //cout<<"start"<<endl;
//         data_preparation( idim, jdim,20,20,target.first,target.second, max_timesteps, nb_width, dz_per,.1, 1,1, num_threads, noentry);
        
//         // double t1=omp_get_wtime();
//         // Grid g=Grid(dimensions,start,target,Fl,ul,vl,nb_width,dz_per,DX,.1,max_timesteps,num_threads,noentry=noentry);
//         // //cout<<idim<<endl;
        
//         // g.main();
//         // t1=omp_get_wtime()-t1;
//         // cout<<"\nomp time in secs : "<<(float)t1<<endl;
        
        
//         // cout<<g.timestep_count<<endl;
//         // g.plot_contours();
//         // auto b=Backtrack(start,target,DX,.1,Fl,ul,vl,g.get_zerocontourslist());
//         // //vector<Point> li=b.backtracking();
//         // b.plot_contours_with_path();
//         //     // cout<<"active_proc_size is "<<g.active_proc_size<<endl;
//         //     //g.plot_contours();
//         //     // cout<<"grid dims are "<<idim<<","<<jdim<<endl;
//         //     // cout<<"target is "<<target.first<<","<<target.second<<endl;

//         // //}
        
                
//     return 0;
// }

