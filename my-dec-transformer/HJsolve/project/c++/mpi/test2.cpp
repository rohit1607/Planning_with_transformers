#include "grid.h"

int main(int argc,char** argv){
    int idim,jdim;
    idim=jdim=100;
    
    int prank,psize;
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
        double t1,t2;
        MPI_Comm comm= MPI_COMM_WORLD;
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(comm,&prank);
        MPI_Comm_size(comm,&psize);

        int store_steps;
        store_steps=max_timesteps/psize;
        if(prank==psize-1){
            store_steps=max_timesteps-(psize-1)*store_steps;
        }
        
        cout<<prank<<" "<<store_steps<<endl;
        Fl=new double**[store_steps];
        ul=new double**[store_steps];
        vl=new double**[store_steps];
        for(int i=0;i<store_steps;i++){
            Fl[i]=new double *[idim];
            ul[i]=new double *[idim];
            vl[i]=new double *[idim];
            for(int j=0;j<idim;j++){
                Fl[i][j]=new double[jdim];
                ul[i][j]=new double[jdim];
                vl[i][j]=new double[jdim];
            }
        }

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

        t1=MPI_Wtime();
        Grid g=Grid(dimensions,start,target,Fl,ul,vl,nb_width,dz_per,DX,.1,prank,psize,max_timesteps,comm);
        cout<<idim<<endl;
        g.main();
        t2=MPI_Wtime();
        //cout<<"jjj"<<endl;
        if(prank ==0){
            //cout<<dims[mt]<<endl;
            cout<</*"time taken in s "<<*/(t2-t1)<<endl;
            cout<<g.timestep_count<<endl;
            // cout<<"active_proc_size is "<<g.active_proc_size<<endl;
            //g.plot_contours();
            // cout<<"grid dims are "<<idim<<","<<jdim<<endl;
            // cout<<"target is "<<target.first<<","<<target.second<<endl;

        }
        for(int i=0;i<store_steps;i++){
            for(int j=0;j<idim;j++){
                delete[] Fl[i][j];
                delete[] ul[i][j];
                delete[] vl[i][j];
            }
        }
        MPI_Finalize();
        
    return 0;
}
