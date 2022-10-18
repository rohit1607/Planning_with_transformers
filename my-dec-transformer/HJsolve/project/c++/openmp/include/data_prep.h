#ifndef DATA_PREP_H
#define DATA_PREP_H
#include "grid.h"
#include "backtrack.h"

extern "C"
{
	void data_preparation(int idim,int jdim,void(*readfile)(int,int**,double**,double**,double**),void(*readvels)(int,double*,double*,double*,int,int),void(*plotfn)(int),int starti,int startj,int targeti,int targetj,int maxtimesteps,int nb_width,float dz_per,float dt,float dx,float dy,int num_threads,int** land,int land_length);

	//void data_preparation2(int idim,int jdim,int starti,int startj,int targeti,int targetj,int maxtimesteps,int nb_width,float dz_per,float dt,float dx,float dy,int num_threads,void(*readfile)(int,double**,double**,double**));

	//void data_preparation2(int idim,int jdim,void(*readfile)(int,int**,double**,double**,double**),int starti,int startj,int targeti,int targetj,int maxtimesteps,int nb_width,float dz_per,float dt,float dx,float dy,int num_threads);

}





#endif