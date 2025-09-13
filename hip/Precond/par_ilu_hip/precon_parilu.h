#ifndef _PRECOND_parilu_H
#define _PRECOND_parilu_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "preconditioner.h"
//#include"parilu.h"
#include <sys/time.h>
#include<iostream>
#include<fstream>
#include"../par_ilu_hip/par_ilu_solve.h"


#define T double
class Precond_parilu:public Precond{
public:
    HostMatrix *hostmtx;
	
	int sweep;

	T*lvald;
	int*lrowsd;
	int*lcolsd;
	int nnzl;

    int*setd;

    T*uvald;
	int*urowsd;
	int*ucolsd;
	int nnzu;

    T*d_left_sum;
    int*d_indegree;
	
	int*row_referenced;

    T*xd_inter;
  
    Precond_parilu(){
    
	n_p=0;
 	hostmtx=NULL;
    sweep=5;
    
    }
    void create_precond(int n_in, HostMatrix *hostmtx_in){
	n_p=n_in;
	hostmtx=hostmtx_in;
    hipMalloc(&setd,sizeof(int)*n_p);
    hipMalloc(&d_left_sum,sizeof(T)*n_p);
    hipMalloc(&d_indegree,sizeof(int)*n_p);
    hipMalloc(&xd_inter,sizeof(T)*n_p);

    }
    void set_parilu(int sweep=5){
        this->sweep=sweep;
    
    }
    void preconditioner_init(){
   		printf("precon_init\n");
        //edit here!!!!!!!!!!!!!!!!!!!!!
		int n=n_p;
        double*val=hostmtx->getval();
        int*cols=hostmtx->getidx();
        int*rows=hostmtx->getptr();
	    int nnz=rows[n];
	    sweep=5;

        T*vald;
        int*rowsd;
        int*colsd;
	    hipMalloc(&vald,sizeof(T)*nnz);
        hipMalloc(&rowsd,sizeof(int)*(n+1));
        hipMalloc(&colsd,sizeof(int)*nnz);
    
        hipMemcpyHtoD(vald,val,sizeof(T)*nnz);
        hipMemcpyHtoD(rowsd,rows,sizeof(int)*(n+1));
        hipMemcpyHtoD(colsd,cols,sizeof(int)*nnz);
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //分配l,u的val，cols，rows，并用原矩阵赋值
    parilu_pre_set(vald,rowsd,colsd,n,nnz,\
    lvald,lrowsd,lcolsd,nnzl,\
    uvald,ucolsd,urowsd,nnzu,\
    row_referenced);

    //可以的调用Ucsc_trsv_sync_free_pre了

    //求解parilu
	parilu_fact(vald,rowsd,colsd,row_referenced,\
    			lvald,lrowsd,lcolsd,\
                uvald,ucolsd,urowsd,n,nnz,sweep);
    
    //U求解的准备，可以使用另一个流
    {
            int blocksize=warpSize*16;// warpSize=64
            int gridsize=nnzu/blocksize+1;
            hipLaunchKernelGGL(Ucsc_trsv_sync_free_pre,dim3(gridsize),dim3(blocksize),0,0,\
    ucolsd,urowsd,n,d_indegree);
    }
    
    /*
    hipFree(vald);
    hipFree(rowsd);
    hipFree(colsd); 
    */
        
    }
    void preconditioner(HostVector *x,HostVector *y){
    	
		struct timeval t1,t2;
		gettimeofday(&t1,NULL);
        int n=n_p;
        //edit here!!!!!!!!!
        T*xd;
        T*yd;
        //T*xd_inter;
        hipMalloc(&xd,sizeof(T)*n);
        hipMalloc(&yd,sizeof(T)*n);
        //hipMalloc(&xd_inter,sizeof(T)*n);

        hipMemcpyHtoD(yd,y,sizeof(T)*n);
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        lcsr_trsv_setd(lvald,lrowsd,lcolsd,xd_inter,yd,n,setd);


        ucsc_trsv_leftsum_indegree(uvald,ucolsd,urowsd,xd,xd_inter,n,nnzu,d_left_sum,d_indegree);


    gettimeofday(&t2,NULL);
    double elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000. +(t2.tv_usec - t1.tv_usec) / 1000.; 
        
    printf("  lusolve time=%lf\n",elapsed_time);


    }
    void preconditioner_free(){
        hipFree(lvald);
        hipFree(lrowsd);
        hipFree(lcolsd);
        hipFree(setd);
        hipFree(uvald);
        hipFree(urowsd);
        hipFree(ucolsd);
        hipFree(d_left_sum);
        hipFree(d_indegree);
        hipFree(row_referenced);
    }
    ~Precond_parilu(){}
};
Precond* precond_set_parilu(){
    return new Precond_parilu();
}
#undef T
#endif
