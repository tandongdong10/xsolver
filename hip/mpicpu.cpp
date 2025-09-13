#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#ifdef HAVE_MPI
#include <mpi.h>
#include "mpicpu.h"
struct topology_c topo_c;
MPI_Request *ireqnbs_1d_s;//=new MPI_Request[topo_c.total_nbs_thisproc];
MPI_Request *ireqnbs_1d_r;//=new MPI_Request[topo_c.total_nbs_thisproc];
double *sendarrayhx;// = new double[topo_c.nGhstCells];
int *sendarrayhx_int;// = new double[topo_c.nGhstCells];
#endif
#ifdef HAVE_MPI
//extern "C" void mpi_topoinit_(int &total_nbs_thisproc,int *nbs_thisproc,int *exchange_ptr,int *exchange_cnts_proc,int *exchange_displs_proc,int &nIntCells, int &nGhstCells)
extern "C" void xsolver_communicator_setup(int total_nbs_thisproc,int nIntCells, int nGhstCells, int *nbs_thisproc,int *exchange_ptr,int *exchange_displs_proc, int nSendCells=0, int *exchange_displs_proc_receive=NULL)
{
    topo_c.total_nbs_thisproc=total_nbs_thisproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&topo_c.myid);
    MPI_Comm_size(MPI_COMM_WORLD,&topo_c.size);
    topo_c.nbs_thisproc=nbs_thisproc;
    topo_c.exchange_ptr=exchange_ptr;
    //topo_c.exchange_cnts_proc=exchange_cnts_proc;
    topo_c.exchange_displs_proc=exchange_displs_proc;
    if(exchange_displs_proc_receive==NULL)
	topo_c.exchange_displs_proc_receive=exchange_displs_proc;
    topo_c.nIntCells=nIntCells;
    topo_c.nGhstCells=nGhstCells;
    topo_c.nSendCells=nSendCells>0?nSendCells:nGhstCells; 
    /*if(topo_c.myid==0){
	printf("nHalo=%d\n",topo_c.nGhstCells);
        printf("topo_c.total_nbs_thisproc=%d\n",topo_c.total_nbs_thisproc);
    	for(int i=0;i<total_nbs_thisproc;i++)
            printf("topo_c.nbs_thisproc[%d]=%d\n",i,topo_c.nbs_thisproc[i]);
        for(int i=0;i<topo_c.size+1;i++)
            printf("%dexchange_displs[%d]=%d\n",topo_c.myid,i,topo_c.exchange_displs_proc[i]);
        for(int i=0;i<total_nbs_thisproc+1;i++)
            printf("%dexchange_displs_receive[%d]=%d\n",topo_c.myid,i,topo_c.exchange_displs_proc_receive[i]);
        for(int ii = 0; ii<topo_c.total_nbs_thisproc; ii++){
       	    int iproc = topo_c.nbs_thisproc[ii];
       	    int startindex = topo_c.exchange_displs_proc[iproc];
       	    int endindex  = topo_c.exchange_displs_proc[iproc+1];
       	    int sendnum = endindex - startindex;//topo_c.exchange_cnts_proc[iproc];
       	    for(int i=startindex;i<endindex;i++)
            	printf("%dexchange_ptr[%d]=%d\n",topo_c.myid,i,topo_c.exchange_ptr[i]);
        }
        //exit(0);
    }*/
    ireqnbs_1d_s=new MPI_Request[topo_c.total_nbs_thisproc];
    ireqnbs_1d_r=new MPI_Request[topo_c.total_nbs_thisproc];
    sendarrayhx = new double[topo_c.nSendCells];
    sendarrayhx_int = new int[topo_c.nSendCells];

/*    printf("total_nbs_this_proc%d = %d\n",topo_c.myid, total_nbs_thisproc);
    printf("nHalo%d = %d\n",topo_c.myid, nGhstCells);
    printf("nInterior%d = %d\n",topo_c.myid, nIntCells);
    printf("nbs_thisproc%d = {",topo_c.myid); for(int i=0; i<total_nbs_thisproc; i++) {printf("%d, ",topo_c.nbs_thisproc[i]);} printf("}\n");
    //int k = exchange_displs_proc[total_nbs_thisproc]-exchange_displs_proc[0];
    //printf("%d\n",k);
    printf("exchange_ptr%d = {",topo_c.myid); for(int i=0; i<4; i++) {printf("%d, ",topo_c.exchange_ptr[i]);} printf("}\n");
    printf("exchange_displs_proc%d = {",topo_c.myid); for(int i=0; i<=4; i++) {printf("%d, ",topo_c.exchange_displs_proc[i]);} printf("}\n");
*/

}
extern "C" void xsolver_communicator_distroy(){
    if(topo_c.nbs_thisproc!=NULL){delete []topo_c.nbs_thisproc; topo_c.nbs_thisproc=NULL;}
    if(topo_c.exchange_displs_proc!=NULL){delete []topo_c.exchange_displs_proc;topo_c.exchange_displs_proc=NULL;} 
    if(topo_c.exchange_displs_proc_receive!=NULL){delete []topo_c.exchange_displs_proc_receive;topo_c.exchange_displs_proc_receive=NULL;} 
    if(topo_c.exchange_ptr!=NULL){delete []topo_c.exchange_ptr; topo_c.exchange_ptr=NULL;}
    if(ireqnbs_1d_s!=NULL){delete []ireqnbs_1d_s;ireqnbs_1d_s=NULL;} 
    if(ireqnbs_1d_r!=NULL){delete []ireqnbs_1d_r;ireqnbs_1d_r=NULL;} 
    if(sendarrayhx!=NULL){delete []sendarrayhx;sendarrayhx=NULL;} 
    if(sendarrayhx_int!=NULL){delete []sendarrayhx_int; sendarrayhx_int=NULL;}
    if(topo_c.scount!=NULL){delete []topo_c.scount; topo_c.scount=NULL;}
    if(topo_c.displs!=NULL){delete []topo_c.displs; topo_c.displs=NULL;}
}
void communicator_allgather(int num_send,int *receivebuf){
    if(topo_c.size==1)
        return;
    MPI_Allgather(&num_send,1,MPI_INT,receivebuf,1,MPI_INT,MPI_COMM_WORLD);
}
void  communicator_p2p(int *dataArray){
    if(topo_c.size==1)
        return;
    int iproc;
    int sendnum;
    //int startindex,endindex;
    int *receivepointerhx;
    int startindex,endindex,startindex_receive;
   for(int ii = 0; ii<topo_c.total_nbs_thisproc; ii++){
       iproc = topo_c.nbs_thisproc[ii];
       startindex = topo_c.exchange_displs_proc[ii];
       startindex_receive = topo_c.exchange_displs_proc_receive[ii];
       endindex  = topo_c.exchange_displs_proc[ii+1];
       sendnum = endindex - startindex;//topo_c.exchange_cnts_proc[iproc];
       for(int i=startindex;i<endindex;i++)
           sendarrayhx_int[i] = dataArray[topo_c.exchange_ptr[i]];
       //receivepointerhx = &dataArray[topo_c.nIntCells+startindex];
       receivepointerhx = &dataArray[topo_c.nIntCells+startindex_receive];
       MPI_Irecv(receivepointerhx,topo_c.nGhstCells,MPI_INT,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_r[ii]);
       //MPI_Isend(sendarrayhx,sendnum,MPI_DOUBLE_PRECISION,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_s[ii]);
       MPI_Isend(&sendarrayhx_int[startindex],sendnum,MPI_INT,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_s[ii]);
   }
}
void  communicator_p2p(double *dataArray){
    if(topo_c.size==1)
        return;
    int iproc;
    int sendnum;
    int startindex,endindex,startindex_receive;
    double *receivepointerhx;
   for(int ii = 0; ii<topo_c.total_nbs_thisproc; ii++){
       iproc = topo_c.nbs_thisproc[ii];
       startindex = topo_c.exchange_displs_proc[ii];
       startindex_receive = topo_c.exchange_displs_proc_receive[ii];
       endindex  = topo_c.exchange_displs_proc[ii+1];
       sendnum = endindex - startindex;//topo_c.exchange_cnts_proc[iproc];
       for(int i=startindex;i<endindex;i++)
           sendarrayhx[i] = dataArray[topo_c.exchange_ptr[i]];
       receivepointerhx = &dataArray[topo_c.nIntCells+startindex_receive];
       MPI_Irecv(receivepointerhx,topo_c.nGhstCells,MPI_DOUBLE,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_r[ii]);
       //MPI_Isend(sendarrayhx,sendnum,MPI_DOUBLE_PRECISION,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_s[ii]);
       MPI_Isend(&sendarrayhx[startindex],sendnum,MPI_DOUBLE,iproc,99,MPI_COMM_WORLD,&ireqnbs_1d_s[ii]);
   }
}


//extern "C" void  communicator_p2p_waitall(){
void  communicator_p2p_waitall(){
    if(topo_c.size==1)
        return;
    //MPI_Status *status=new MPI_Status[topo_c.total_nbs_thisproc];
    MPI_Waitall(topo_c.total_nbs_thisproc,ireqnbs_1d_r,MPI_STATUSES_IGNORE);
    MPI_Waitall(topo_c.total_nbs_thisproc,ireqnbs_1d_s,MPI_STATUSES_IGNORE);

}

void  communicator_barrier(){
    MPI_Barrier(MPI_COMM_WORLD);
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
    delete []tmp;
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
    delete []tmp;
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
    delete []tmp;
#endif
}

