#include <ctime>
#include "grid.h"

void data_preparation2(int idim,int jdim,int starti,int startj,int targeti,int targetj,int maxtimesteps,double* F,double* u,double* v,int nb_width,float dz_per,float dt,float dx,float dy){
    Point start,target,DX;
    start.first=starti;start.second=startj;
    target.first=targeti;target.second=targetj;
    DX.first=dx;DX.second=dy;
    int dimensions[2]={idim,jdim};
    //double t1=omp_get_wtime();

    double*** Fl,***ul,***vl;
    Fl=new double**[maxtimesteps];
    ul=new double**[maxtimesteps];
    vl=new double**[maxtimesteps];
    for(int t=0;t<maxtimesteps;t++){
        Fl[t]=new double*[idim];
        ul[t]=new double*[idim];
        vl[t]=new double*[idim];
        for(int i=0;i<idim;i++){
            Fl[t][i]=&(F[(t*idim+i)*jdim]);
            ul[t][i]=&(u[(t*idim+i)*jdim]);
            vl[t][i]=&(v[(t*idim+i)*jdim]);
        }
    }
        Grid g=Grid(dimensions,start,target,Fl,ul,vl,nb_width,dz_per,DX,dt,maxtimesteps);
        
        
        g.main();
        // t1=omp_get_wtime()-t1;
        // cout<<"\nomp time in secs : "<<(float)t1<<endl;
        
        
        //cout<<g.timestep_count<<endl;
        g.plot_contours();
        //auto b=Backtrack(start,target,DX,dt,Fl,ul,vl,g.get_zerocontourslist());
        //vector<Point> li=b.backtracking();
        //b.plot_contours_with_path();
}

int main(int argc,char** argv){
    int idim,jdim;
    idim=jdim=100;
    
    int prank,psize;
    prank=0;
    psize=1;
    Point start,target;
    start=make_pair(20,20);
    int nb_width=5;
    float dz_per=0.5;
    if(argc==5){
        //cout<<argv[1]<<argv[2]<<argv[3]<<endl;
        idim=atoi(argv[1]);
        jdim=atoi(argv[2]);
        nb_width=atoi(argv[3]);
        dz_per=stof(argv[4]);

    }
    //cout<<"args done"<<endl;
    // int dims[14]={100,150,200,250,300,350,400,450,500,600,700,800,900,1000};
    // for(auto mt=0;mt<14;mt++){
        
       //  idim=jdim=dims[mt];
        int max_timesteps=10*idim;
        int dimensions[2]={idim,jdim};
        double*** Fl,***ul,***vl;
        
        target=make_pair(int(0.9*idim),int(0.9*jdim));

        
        
        //cout<<"alloc_2d_int"<<endl;
        Point DX=make_pair(1,1);
        //double t1,t2;
        

        int store_steps;
        store_steps=max_timesteps;
        
        
        //cout<<prank<<" "<<store_steps<<endl;
        Fl=alloc_3d_arr(store_steps,idim,jdim);
        ul=alloc_3d_arr(store_steps,idim,jdim);
        vl=alloc_3d_arr(store_steps,idim,jdim);

        for(int i=0;i<store_steps;i++){
            for(int j=0;j<dimensions[0];j++){
                for(int k=0;k<dimensions[1];k++){   
                    Fl[i][j][k]=1;
                    ul[i][j][k]=0.0;
                    vl[i][j][k]=0.0;
                    if( j>=30 && j<=.8*idim){
                        ul[i][j][k]=0.5;
                  
                    }
                    if(k>=30 && k<=.8*jdim){
                        vl[i][j][k]=0.5;
                    }
                }
            }
        }

        time_t t2=clock();
        Grid g=Grid(dimensions,start,target,Fl,ul,vl,nb_width,dz_per,DX,.1,max_timesteps);
        //cout<<idim<<endl;
        
        g.main();
        t2=clock()-t2;
        cout<<"\nserial time in secs : "<<(float)t2/CLOCKS_PER_SEC<<endl;
        
        
        cout<<g.timestep_count<<endl;
        g.plot_contours();
            // cout<<"active_proc_size is "<<g.active_proc_size<<endl;
            //g.plot_contours();
            // cout<<"grid dims are "<<idim<<","<<jdim<<endl;
            // cout<<"target is "<<target.first<<","<<target.second<<endl;

        //}
        
                
    return 0;
}