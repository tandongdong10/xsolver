#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "Tool/ReadMatrix/csr_mmio_highlevel.h"
#include "solver-C/libhead.h"
#include "mpi_solve_base.h"
//#include "solve_base.h"
#include <bits/stdc++.h>
#include "mpi.h"
int main(int argc, char **argv)
{
    MPI_Init(&argc,&argv);
    int argi = 1;
    char *file_nameA=NULL;

    //read bench.txt
    if(argc>argi)
    {
        file_nameA = argv[argi];
        argi++;
    }
    int mA, nA, nnzA, isSymmetricA;
    // load mtx data to the csr format
    mmio_info(&mA, &nA, &nnzA, &isSymmetricA, file_nameA);
    int *colPtrA = (int *)malloc((nA+1)*sizeof(int));
    int *rowIdxA = (int *)malloc(nnzA*sizeof(int));
    double *valA = (double *)malloc(nnzA*sizeof(double));
    double *a_p = (double *)malloc(mA*sizeof(double));
    double small = 0;
    mmio_data(colPtrA, rowIdxA, valA, a_p, file_nameA,small);
    nnzA=colPtrA[nA];
    //printf("matrixA: %s, mA=%d, nA=%d, nnzA=%d, nnzA(ptr)=%d, isSymmetricA=%d\n", file_nameA, mA, nA, nnzA, colPtrA[mA], isSymmetricA);
    int maxIteration=10000;
    int restart=maxIteration;
    double tol=1e-4;
    int ghst=0;
    double res0;
    int usediter;
    struct timeval time1, time2;
    double *q = (double *)malloc(mA*sizeof(double));
    double *res = (double *)malloc(mA*sizeof(double));
    double *phi = (double *)malloc(mA*sizeof(double));
    memset(phi,0,mA*sizeof(double));
    double one=1.0, zero=0.0,minone=-1;
    for(int i=0;i<mA;i++){
	q[i]=1;//i%10+0.5;
    }
    //omp_set_num_threads(1);
    for(int i=0;i<1;i++){
    	memset(phi,0,mA*sizeof(double));
   	 bmax ( nA, nA, colPtrA, valA, rowIdxA, q, phi, res);
     	double res2 = dot(nA,res,res);
     	res2 = sqrt(res2);
    	//printf("start !!! res0=%f\n",res2);
    	MPI_Barrier(MPI_COMM_WORLD);
    	gettimeofday(&time1, NULL); 
    	solve_base(mA, colPtrA, rowIdxA, valA, q, phi, maxIteration, tol, res0, usediter);
    	MPI_Barrier(MPI_COMM_WORLD);
    	gettimeofday(&time2, NULL); 
    	double elapsed_time = (time2.tv_sec - time1.tv_sec) * 1000. +(time2.tv_usec - time1.tv_usec) / 1000.; 
    if(topo_c.myid<3)
    	printf("Elapsed time: %lf(ms)\n", elapsed_time); 
    	if(topo_c.myid==0){
    	    bmax ( nA, nA, colPtrA, valA, rowIdxA, q, phi, res);
     	    double res1 = dot(nA,res,res);
     	    res1 = sqrt(res1);
    	    printf("Final !!! res0=%f, res=%f, usediter=%d\n",res0, res1,usediter);
    	}
    }
    if(a_p != NULL){free(a_p); a_p=NULL;}
    if(q != NULL){free(q); q=NULL;}
    if(res != NULL){free(res); res=NULL;}
    if(phi != NULL){free(phi); phi=NULL;}
    if(colPtrA != NULL) {free(colPtrA); colPtrA=NULL;}
    if(rowIdxA != NULL) {free(rowIdxA); rowIdxA=NULL;}
    if(valA != NULL){free(valA); valA=NULL;}
    MPI_Finalize();

    return 0;
}
