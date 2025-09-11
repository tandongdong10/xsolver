#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string.h>
#include <assert.h>
#ifdef HAVE_MPI
#include <mpi.h>
#include "mpicpu.h"
struct topology_c topo_c;
MPI_Request *ireqnbs_1d_s;//=new MPI_Request[topo_c.total_nbs_thisproc];
MPI_Request *ireqnbs_1d_r;//=new MPI_Request[topo_c.total_nbs_thisproc];
double *sendarrayhx;// = new double[topo_c.nGhstCells];
#endif
#ifdef HAVE_MPI
//extern "C" void mpi_topoinit_(int &total_nbs_thisproc,int *nbs_thisproc,int *exchange_ptr,int *exchange_cnts_proc,int *exchange_displs_proc,int &nIntCells, int &nGhstCells)
extern "C" void communicator_init(int &total_nbs_thisproc,int &nIntCells, int &nGhstCells, int *nbs_thisproc,int *exchange_ptr,int *exchange_displs_proc)
{
    topo_c.total_nbs_thisproc=total_nbs_thisproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&topo_c.myid);
    MPI_Comm_size(MPI_COMM_WORLD,&topo_c.size);
    topo_c.nbs_thisproc=nbs_thisproc;
    topo_c.exchange_ptr=exchange_ptr;
    //topo_c.exchange_cnts_proc=exchange_cnts_proc;
    topo_c.exchange_displs_proc=exchange_displs_proc;
    topo_c.nIntCells=nIntCells;
    topo_c.nGhstCells=nGhstCells;
    ireqnbs_1d_s=new MPI_Request[topo_c.total_nbs_thisproc];
    ireqnbs_1d_r=new MPI_Request[topo_c.total_nbs_thisproc];
    sendarrayhx = new double[topo_c.nGhstCells];
    //printf("!!!!!!!!topo_c.nIntCells=%d, myrank=%d,ranksize=%d!!!\n",topo_c.nIntCells,topo_c.myid,topo_c.size);
    //topo_c.nbs_thisproc=new int[];
    //memcpy(topo_c.nbs_thisproc,nbs_thisproc,  *sizeof(int));
    //topo_c.exchange_ptr=new int[];
    //memcpy(topo_c.exchange_ptr,exchange_ptr,  *sizeof(int));
    //topo_c.exchange_cnts_proc=new int[];
    //memcpy(topo_c.exchange_cnts_proc,exchange_cnts_proc,  *sizeof(int));
    //topo_c.exchange_displs_proc=new int[];
    //memcpy(topo_c.exchange_displs_proc,exchange_displs_proc,  *sizeof(int));

}

void  communicator_p2p(double *dataArray){
    if(topo_c.size==1)
        return;
    int iproc;
    int sendnum;
    int startindex,endindex;
    double *receivepointerhx;
   for(int ii = 0; ii<topo_c.total_nbs_thisproc; ii++){
       iproc = topo_c.nbs_thisproc[ii];
       startindex = topo_c.exchange_displs_proc[iproc];
       endindex  = topo_c.exchange_displs_proc[iproc+1];
       sendnum = endindex - startindex;//topo_c.exchange_cnts_proc[iproc];
       for(int i=startindex;i<endindex;i++)
           sendarrayhx[i] = dataArray[topo_c.exchange_ptr[i]];///exchange_ptr 0-based
       receivepointerhx = &dataArray[topo_c.nIntCells+startindex];
       MPI_Irecv(receivepointerhx,sendnum,MPI_DOUBLE_PRECISION,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_r[ii]);
       //MPI_Isend(sendarrayhx,sendnum,MPI_DOUBLE_PRECISION,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_s[ii]);
       MPI_Isend(&sendarrayhx[startindex],sendnum,MPI_DOUBLE_PRECISION,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_s[ii]);
   }
}


//extern "C" void  communicator_p2p_waitall(){
void  communicator_p2p_waitall(){
    if(topo_c.size==1)
        return;
    MPI_Status *status=new MPI_Status[topo_c.total_nbs_thisproc];
    MPI_Waitall(topo_c.total_nbs_thisproc,ireqnbs_1d_r,status);
    MPI_Waitall(topo_c.total_nbs_thisproc,ireqnbs_1d_s,status);


}

#endif
//extern "C" void  communicator_sum(double &value){
void  communicator_sum(double &value){
#ifdef HAVE_MPI
    double tmp;
    MPI_Allreduce(&value,&tmp,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    value = tmp;
#endif
}
void communicator_sum(double *value,int n){
#ifdef HAVE_MPI
    double *tmp=new double[n];
    MPI_Allreduce(value,tmp,n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    memcpy(value,tmp,n*sizeof(double));
#endif
}
void  communicator_sum(float &value){
#ifdef HAVE_MPI
    float tmp;
    MPI_Allreduce(&value,&tmp,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    value = tmp;
#endif
}
void communicator_sum(float *value,int n){
#ifdef HAVE_MPI
    float *tmp=new float[n];
    MPI_Allreduce(value,tmp,n,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    memcpy(value,tmp,n*sizeof(float));
#endif
}
void  communicator_sum(int &value){
#ifdef HAVE_MPI
    int tmp;
    MPI_Allreduce(&value,&tmp,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    value = tmp;
#endif
}
void communicator_sum(int *value,int n){
#ifdef HAVE_MPI
    int *tmp=new int[n];
    MPI_Allreduce(value,tmp,n,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    memcpy(value,tmp,n*sizeof(int));
#endif
}

