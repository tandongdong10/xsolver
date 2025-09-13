#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
//#include "mpi.h"
#ifndef _MPICPU_H_
#define _MPICPU_H_
extern struct topology_c topo_c;
struct topology_c{
    int total_nbs_thisproc;
    int myid;
    int size;
    int *nbs_thisproc;
    int *exchange_ptr;
    //int *exchange_cnts_proc;
    int *exchange_displs_proc;
    int *exchange_displs_proc_receive;
    int nIntCells;
    int nSendCells;
    int nGhstCells;
    int *scount;
    int *displs;
};
#endif

//extern struct topology_c topo_c;
//MPI_Request *ireqnbs_1d_s;//=new MPI_Request[topo_c.total_nbs_thisproc];
//MPI_Request *ireqnbs_1d_r;//=new MPI_Request[topo_c.total_nbs_thisproc];
//double *sendarrayhx;// = new double[topo_c.nGhstCells];

extern "C" void xsolver_communicator_setup(int total_nbs_thisproc,int nIntCells, int nGhstCells, int *nbs_thisproc,int *exchange_ptr,int *exchange_displs_proc, int nSendCells,int *exchange_displs_proc_receive);
extern "C" void xsolver_communicator_distroy();

//extern "C" void  communicator_p2p(double *dataArray){
void  communicator_p2p(int *dataArray);
void  communicator_p2p(double *dataArray);

//extern "C" void  communicator_p2p_waitall(){
void  communicator_p2p_waitall();
void  communicator_barrier();
void  communicator_sum(double &value);
void communicator_sum(double *value,int n=1);
void  communicator_sum(float &value);
void communicator_sum(float *value,int n=1);
void  communicator_sum(int &value);
void communicator_sum(int *value,int n=1);
