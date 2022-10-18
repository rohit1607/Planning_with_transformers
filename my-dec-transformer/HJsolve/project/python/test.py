from Grid import *

start=(20,20)
target=[90,90]
u=1.2
f=1
dims=(100,100)
U=np.zeros((1000,dims[0],dims[1]))
V=np.zeros((1000,dims[0],dims[1]))
U[:,30:80,:]=.5
V[:,:,:]=0
# for i in range(30,60):
#     u[:,i]=i*.01
#v[:,30:59,]=.5
F=np.ones((1000,dims[0],dims[1]))*f
#F[:,50,0:20]=0
# F[:,30,0:70]=0
# F[:,50:,50]=0
# F[:,50,:20]=0
# F[:,50,30:50]=0
# xslice=[slice(50,100),slice(50,51),slice(30,100),slice(30,31)]
# yslice=[slice(50,51),slice(0,50),slice(70,71),slice(0,70)]

xslice=[slice(35,40),slice(35,40)]
yslice=[slice(0,50),slice(53,100)]

noentry_slices=[xslice,yslice]
cnts=Grid(dims,start,target,F,U,V,5,.5,(1,1),.1,order=2,advection_term=9,noentry_slices=noentry_slices)

cnts.main()
print(cnts._timestep_count)
cnts._plot_contours_with_path2()
