from math import sin,cos,pi,floor,ceil

def lat_length(lat_deg):
    lat=lat_deg*pi/180
    return 111132.92-559.82*cos(2*lat)+1.175*cos(4*lat)-0.0023*cos(6*lat)
def lon_length(lat_deg):
    lat=lat_deg*pi/180
    return 111412.84*cos(lat)-93.5*cos(3*lat)+0.118*cos(5*lat)


def Grid_generation(lonsrange,latsrange,res=0.08):
    Grid=[]

    n_steps=int((lonsrange[1]-lonsrange[0])/res)
    print(n_steps)
    ref_lat=max(0,min(latsrange[0],latsrange[1]))

    lon_stepsize=res*lon_length(ref_lat)
    lat_stepsize=res*lat_length(ref_lat)
    #print(lat_stepsize,lon_stepsize)
    next_lat=latsrange[0]
    while(next_lat<=latsrange[1]): 
        diff=(lon_length(ref_lat)-lon_length(next_lat))*(lonsrange[1]-lonsrange[0])
        start=lonsrange[0]-((.5*diff)/lon_length(next_lat))
        lonstep_lat=lon_stepsize/lon_length(next_lat)
        cur_lon=start

        lonlist=[(next_lat,cur_lon)]
        for i in range(n_steps):
            cur_lon+=lonstep_lat
            lonlist.append((next_lat,cur_lon))
        Grid.append(lonlist)
        next_lat=round(next_lat+res,2)
    return lat_stepsize,lon_stepsize,Grid
