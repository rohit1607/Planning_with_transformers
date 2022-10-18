import matplotlib.pyplot as plt

def plot_contour(timestep):
	plt.clf()
	root="/home/revanth/Mtech/projct/c++/openmp/include"
	folder=root+"/testlogs/zcs/"
	file="{}.txt".format(timestep)

	with open(folder+file,"r") as f:
		text=f.readlines()
	lines=[i.strip("\n").split(")") for i in text]
	contours=[[]]
	for line in lines:
		if(len(line)==2):
			index=int(line[0])
			pts=[float(t) for t in line[1].split(",")]
			if(len(contours)==index-1):
				contours.append([])
			contours[index-1].append(pts)
	for contour in contours:
		ilist=[i[0] for i in contour]
		jlist=[i[1] for i in contour]
		plt.plot(jlist,ilist)
	#plt.show()
	plt.savefig(root+"/figures/{}.png".format(timestep))
def plot_allcontours(total_timesteps):
	plt.clf()
	root="/home/revanth/Mtech/projct/c++/openmp/include"
	folder=root+"/testlogs/zcs/"
	for t in range(total_timesteps):
		file="{}.txt".format(t)

		with open(folder+file,"r") as f:
			text=f.readlines()
		lines=[i.strip("\n").split(")") for i in text]
		contours=[[]]
		for line in lines:
			if(len(line)==2):
				index=int(line[0])
				pts=[float(t) for t in line[1].split(",")]
				if(len(contours)==index-1):
					contours.append([])
				contours[index-1].append(pts)
		for contour in contours:
			ilist=[i[0] for i in contour]
			jlist=[i[1] for i in contour]
			plt.plot(jlist,ilist)
	plt.savefig(root+"/figures/all.png")
def plot_contours_with_path(total_timesteps):
	plt.clf()
	root="/home/revanth/Mtech/projct/c++/openmp/include"
	folder=root+"/testlogs/zcs/"
	for t in range(0,total_timesteps,1):
		file="{}.txt".format(t)

		with open(folder+file,"r") as f:
			text=f.readlines()
		lines=[i.strip("\n").split(")") for i in text]
		contours=[[]]
		for line in lines:
			if(len(line)==2):
				index=int(line[0])
				pts=[float(t) for t in line[1].split(",")]
				if(len(contours)==index-1):
					contours.append([])
				contours[index-1].append(pts)
		for contour in contours:
			ilist=[i[0] for i in contour]
			jlist=[i[1] for i in contour]
			plt.plot(jlist,ilist)
	#print("all co")
	file="/testlogs/zcs/path.txt"
	with open(root+file,"r") as f:
		text=f.readlines()
	lines=[i.strip("\n").split(",") for i in text]
	#print(lines)
	path=[]

	for line in lines:
		if(len(line)==2):
			path.append([float(line[0]),float(line[1])])
	#print(path)
	ilist=[i[0] for i in path]
	jlist=[i[1] for i in path]
	plt.plot(jlist,ilist,c="r",label="optimal path")
	
	start=(227,261)
	destination=(165,72)
	plt.scatter(start[1],start[0],marker=".",label="chennai",s=200,zorder=100)
	plt.scatter(destination[1],destination[0],marker="*",label="kavaratti",s=200,zorder=101)
	#plt.scatter(jlist,ilist)
	plt.legend()
	plt.savefig(root+"/figures/path.png")

# for i in range(0,3104):
# 	plot_contour(i)
#for i in range(4572,4674):
#	plot_contour(i)
t=733
#for i in range(t):
	# plot_contour(i)

plot_contours_with_path(t)
