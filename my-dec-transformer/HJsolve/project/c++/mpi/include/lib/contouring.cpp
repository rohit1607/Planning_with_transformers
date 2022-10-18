#include "contouring.h"

Point Contouring::linear_interpolation(Point p1,Point p2,double vall1,double vall2){

    double i1=p1.first;double j1=p1.second;
    double i2=p2.first;double j2=p2.second;
    float i,j;
    double val1=vall1;double val2=vall2;

    if(val1 >val2){
        swap(j1,j2);
        swap(i1,i2);
        swap(val1,val2);
    }
    if(i1==i2){
        i=i1;
        j=j2-val2*(j1-j2)/(val1-val2);
    }
    if(j1==j2){
        j=j1;
        i=i2-val2*(i1-i2)/(val1-val2);
    }
    Point p;
    p.first=i;p.second=j;
    
    return p;
}

void Contouring::marching_squares(Point p1,Point p2,Point p3,Point p4,double val1,double val2,double val3,double val4){



    if(val1==0 || val2==0 || val3==0 || val4==0){
        bool z1,z2,z3,z4;
        z1=(val1==0);
        z2=(val2==0);
        z3=(val3==0);
        z4=(val4==0);
        int action=z1+z2+z3+z4;
        int decision=8*(!z4)+4*(!z3)+2*(!z2)+(!z1);

        if(action==4){
            Edge e1={p1,p2};
            Edge e2={p2,p3};
            Edge e3={p3,p4};
            Edge e4={p4,p1};
            Edges_list.push_back(e1);
            Edges_list.push_back(e2);
            Edges_list.push_back(e3);
            Edges_list.push_back(e4);
        }
        else if(action==3){
            if(decision==1){
                Edge e1={p2,p3};
                Edge e2={p3,p4};
                Edges_list.push_back(e1);
                Edges_list.push_back(e2);

            }
            else if(decision==2){
                Edge e1={p3,p4};
                Edge e2={p4,p1};
                Edges_list.push_back(e1);
                Edges_list.push_back(e2);

            }
            else if(decision==4){
                Edge e1={p1,p2};
                Edge e2={p1,p4};
                Edges_list.push_back(e1);
                Edges_list.push_back(e2);

            }
            else{
                Edge e1={p1,p2};
                Edge e2={p2,p3};
                Edges_list.push_back(e1);
                Edges_list.push_back(e2);

            }
        }

        else if(action==2){
            if(decision==6){
                if(val2*val3>0){
                    Edge e={p1,p4};
                    Edges_list.push_back(e);

                }
                else{
                    Point m=linear_interpolation(p3,p2,val3,val2);
                    Edge e1={p1,m};
                    Edge e2={m,p4};
                    Edges_list.push_back(e1);
                    Edges_list.push_back(e2);

                }
            }
            else if(decision==12){
                if(val4*val3>0){
                    Edge e={p1,p2};
                    Edges_list.push_back(e);

                }
                else{
                    Point m=linear_interpolation(p3,p4,val3,val4);
                    Edge e1={p1,m};
                    Edge e2={m,p2};
                    Edges_list.push_back(e1);
                    Edges_list.push_back(e2);

                }
            }
            else if(decision==9){
                if(val1*val4>0){
                    Edge e={p2,p3};
                    Edges_list.push_back(e);

                }
                else{
                    Point m=linear_interpolation(p1,p4,val1,val4);
                    Edge e1={p2,m};
                    Edge e2={m,p3};
                    Edges_list.push_back(e1);
                    Edges_list.push_back(e2);
                }
            }
            else if(decision==3){
                if(val2*val1>0){
                    Edge e={p3,p4};
                    Edges_list.push_back(e);

                }
                else{
                    Point m=linear_interpolation(p1,p2,val1,val2);
                    Edge e1={p3,m};
                    Edge e2={m,p4};

                    Edges_list.push_back(e1);
                    Edges_list.push_back(e2);

                }
            }
            else if(decision==5){
                Edge e={p2,p4};
                Edges_list.push_back(e);

            }
            else if(decision==10){

                Edge e={p1,p3};
                Edges_list.push_back(e);

            }
        }
        else if(action==1){
            if(val1==0){
                if(val2*val3<0){
                    Point m=linear_interpolation(p2,p3,val2,val3);
                    Edge e={m,p1};
                    Edges_list.push_back(e);
                }
                else if(val3*val4<0){
                    Point m=linear_interpolation(p3,p4,val3,val4);
                    Edge e={p1,m};
                    Edges_list.push_back(e);
                }

            }
            else if(val2==0){
                if(val1*val4<0){
                    Point m=linear_interpolation(p1,p4,val1,val4);
                    Edge e={m,p2};
                    Edges_list.push_back(e);
                }
                else if(val3*val4<0){
                    Point m=linear_interpolation(p3,p4,val3,val4);
                    Edge e={p2,m};
                    Edges_list.push_back(e);
                }

            }
            else if(val3==0){
                if(val2*val1<0){
                    Point m=linear_interpolation(p2,p1,val2,val1);
                    Edge e={m,p3};
                    Edges_list.push_back(e);
                }
                else if(val1*val4<0){
                    Point m=linear_interpolation(p1,p4,val1,val4);
                    Edge e={p3,m};
                    Edges_list.push_back(e);
                }

            }
            else if(val4==0){
                if(val2*val3<0){
                    Point m=linear_interpolation(p2,p3,val2,val3);
                    Edge e={m,p4};
                    Edges_list.push_back(e);
                }
                else if(val1*val2<0){
                    Point m=linear_interpolation(p1,p2,val1,val2);
                    Edge e={p4,m};
                    Edges_list.push_back(e);
                }

            }
        }
    }

    else{
        bool b1,b2,b3,b4;
        b1=(val1>0);
        b2=(val2>0);
        b3=(val3>0);
        b4=(val4>0);
        int action=b1*8+b2*4+b3*2+b4;

        if(action==15 || action==0){

        }
        else if(action==14 || action==1){
            Point z1=linear_interpolation(p1,p4,val1,val4);
            Point z2=linear_interpolation(p3,p4,val3,val4);
            Edge e;
            e.one=z1;
            e.two=z2;
            Edges_list.push_back(e);

        }
        else if(action==13 || action==2){
            Point z1=linear_interpolation(p3,p4,val3,val4);
            Point z2=linear_interpolation(p3,p2,val3,val2);
            Edge e;
            e.one=z1;
            e.two=z2;
            Edges_list.push_back(e);

        }
        else if(action==12 || action==3){
            Point z1=linear_interpolation(p1,p4,val1,val4);
            Point z2=linear_interpolation(p3,p2,val3,val2);
            Edge e;
            e.one=z1;
            e.two=z2;
            Edges_list.push_back(e);

        }

        else if(action==11 || action==4){
            Point z1=linear_interpolation(p1,p2,val1,val2);
            Point z2=linear_interpolation(p3,p2,val3,val2);
            Edge e;
            e.one=z1;
            e.two=z2;
            Edges_list.push_back(e);
        }
        else if(action==9 || action==6){
            
            Point z1=linear_interpolation(p1,p2,val1,val2);
            Point z2=linear_interpolation(p3,p4,val3,val4);
            Edge e;
            e.one=z1;
            e.two=z2;

            Edges_list.push_back(e);
        }
        else if(action==8 || action==7){
            Point z1=linear_interpolation(p1,p4,val1,val4);
            Point z2=linear_interpolation(p1,p2,val1,val2);
            Edge e;
            e.one=z1;
            e.two=z2;
            Edges_list.push_back(e);

        }
        else if(action==10){
            double center_val=(val1+val2+val3+val4);
            if(center_val>0){
                Point z1=linear_interpolation(p1,p4,val1,val4);
                Point z2=linear_interpolation(p3,p4,val3,val4);
                Edge e1;
                e1.one=z1;
                e1.two=z2;

                Point z3=linear_interpolation(p1,p2,val1,val2);
                Point z4=linear_interpolation(p3,p2,val3,val2);
                Edge e2;
                e2.one=z3;
                e2.two=z4;
                Edges_list.push_back(e1);
                Edges_list.push_back(e2);

            }
            else{
                Point z1=linear_interpolation(p1,p2,val1,val2);
                Point z2=linear_interpolation(p1,p4,val1,val4);
                Edge e1;
                e1.one=z1;
                e1.two=z2;

                Point z3=linear_interpolation(p3,p2,val3,val2);
                Point z4=linear_interpolation(p3,p4,val3,val4);
                Edge e2;
                e2.one=z3;
                e2.two=z4;
                Edges_list.push_back(e1);
                Edges_list.push_back(e2);

            }
        }

        else if(action==5){
            double center_val=(val1+val2+val3+val4);
            if(center_val<0){
                Point z1=linear_interpolation(p1,p4,val1,val4);
                Point z2=linear_interpolation(p3,p4,val3,val4);
                Edge e1;
                e1.one=z1;
                e1.two=z2;

                Point z3=linear_interpolation(p1,p2,val1,val2);
                Point z4=linear_interpolation(p3,p2,val3,val2);
                Edge e2;
                e2.one=z3;
                e2.two=z4;
                Edges_list.push_back(e1);
                Edges_list.push_back(e2);

            }
            else{
                Point z1=linear_interpolation(p1,p2,val1,val2);
                Point z2=linear_interpolation(p1,p4,val1,val4);
                Edge e1;
                e1.one=z1;
                e1.two=z2;

                Point z3=linear_interpolation(p3,p2,val3,val2);
                Point z4=linear_interpolation(p3,p4,val3,val4);
                Edge e2;
                e2.one=z3;
                e2.two=z4;
                Edges_list.push_back(e1);
                Edges_list.push_back(e2);

            }
        }
    }
}
void Contouring::contouring_grid(){
    for(set<Point>::iterator it=narrowband.begin();it!=narrowband.end();++it){
        int i=it->first;
        int j=it->second;
        if(i<idim && j<jdim){
            bool i_nxt=(narrowband.find(make_pair(i+1,j))!=narrowband.end());
            bool j_nxt=(narrowband.find(make_pair(i,j+1))!=narrowband.end());
            bool ij_nxt=(narrowband.find(make_pair(i+1,j+1))!=narrowband.end());
            if (i_nxt && j_nxt && ij_nxt){
                Point p1,p2,p3,p4;
                p1.first=i+1;p1.second=j;
                p2.first=i+1;p2.second=j+1;
                p3.first=i;p3.second=j+1;
                p4.first=i;p4.second=j;

                marching_squares(p1,p2,p3,p4,phi[i+1][j],phi[i+1][j+1],phi[i][j+1],phi[i][j]);
            }
        }
    }
}
bool Equal(Point p1,Point p2){
    if(pow((pow((p1.first-p2.first),2))+(pow((p1.second-p2.second),2)),.5)<1e-8){
        return true;
    }
    return false;
}
bool eqal(Edge e1,Edge e2){
    if(Equal(e1.one,e2.one) && Equal(e1.two,e2.two)){
        return true;
    }
    if(Equal(e1.one,e2.two) && Equal(e1.two,e2.one)){
        return true;
    }
    return false;
}
void Contouring::connect_points(){
    while(!Edges_list.empty()){
        list<Point> temp_zc;
        Edge e=Edges_list.front();
        Edges_list.pop_front();
        setprecision(6);
        Point p1=e.one;
        Point p2=e.two;
        temp_zc.push_back(p1);
        temp_zc.push_back(p2);
        bool connection=true;

        while(connection){
            connection=false;
            for (list<Edge>::iterator it=Edges_list.begin();it!=Edges_list.end();){
                Edge e1=*it;
                if(Equal(e1.one,temp_zc.back())){
                    //temp_zc.push_back(e1.one);
                    temp_zc.push_back(e1.two);
                    it=Edges_list.erase(it);
                    connection=true;
                    break;
                }
                else if(Equal(e1.two,temp_zc.back())){
                    //temp_zc.push_back(e1.two);
                    temp_zc.push_back(e1.one);
                    it=Edges_list.erase(it);
                    connection=true;
                    break;
                }
                else{
                    it++;
                }
            }
            for (list<Edge>::iterator it=Edges_list.begin();it!=Edges_list.end();){
                Edge e1=*it;
                if(Equal(e1.one,temp_zc.front())){
                    //temp_zc.push_front(e1.one);
                    temp_zc.push_front(e1.two);
                    it=Edges_list.erase(it);
                    connection=true;
                    break;
                }
                else if(Equal(e1.two,temp_zc.front())){
                    //temp_zc.push_front(e1.two);
                    temp_zc.push_front(e1.one);
                    it=Edges_list.erase(it);
                    connection=true;
                    break;
                }
                else{
                    it++;
                }
            }

        }
        //cout<<"connected"<<endl;
        vector<Point> zc;
        for(list<Point>::iterator it=temp_zc.begin();it!=temp_zc.end();++it){
            zc.push_back(*it);
        }
        zero_contours.push_back(zc);
    }
}

vector<vector< Point > > Contouring::get_contours(){
    contouring_grid();
    //cout<<"contoured"<<endl;
    Edges_list.unique(eqal);
    connect_points();

    
    return zero_contours;
}