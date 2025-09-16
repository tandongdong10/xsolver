#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "hip/hip_runtime.h"
#include <iostream>
#include "Tool/ReadMatrix/csr_mmio_highlevel.h"
//#include "omp.h"
#include "mkl_spblas.h"
#include "mkl.h"
#include "mkl_types.h"
#include "solve_base.h"
#include <bits/stdc++.h>
/*__global__ void matrixTranspose(const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

//    out[y * width + x] = in[x * width + y];
}
__global__ void test(const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    
}
*/
int main(int argc, char **argv)
{
    int argi = 1;
    char *file_nameA;

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
    int *rowIdxA = (int *)malloc(2*nnzA*sizeof(int));
    double *valA = (double *)malloc(2*nnzA*sizeof(double));
    double *a_p = (double *)malloc(mA*sizeof(double));
    double small = 0;
    mmio_data(colPtrA, rowIdxA, valA, a_p, file_nameA,small);
    nnzA=colPtrA[nA];
    printf("matrixA: %s, mA=%d, nA=%d, nnzA=%d, nnzA(ptr)=%d, isSymmetricA=%d\n", file_nameA, mA, nA, nnzA, colPtrA[mA], isSymmetricA);
    printf("matrixA: %s,colPtrA=%d,%d,rowIdxA=%d, valA=%lg,rowIdxA=%d, valA=%lg,rowIdxA=%d, valA=%lg,rowIdxA=%d, valA=%lg\n", file_nameA, colPtrA[3],colPtrA[4],rowIdxA[8],valA[8],rowIdxA[9],valA[9],rowIdxA[10],valA[10],rowIdxA[11],valA[11]);
    int maxIteration=300;
    int restart=maxIteration;
    double tol=1e-4;
    int ghst=0;
    double res0;
    int usediter;
    struct timeval time1, time2;
    double *q = (double *)malloc(mA*sizeof(double));
    double *res = (double *)malloc(mA*sizeof(double));
    //double *expected_solution = (double *)malloc(mA*sizeof(double));
    double *phi = (double *)malloc(mA*sizeof(double));
    memset(phi,0,mA*sizeof(double));
    double one=1.0, zero=0.0,minone=-1;
    for(int i=0;i<mA;i++){
	q[i]=1;//i%10+0.5;
    }

    //test whether matrix diag has 0
    /*for(int i=0;i<mA;i++){
	double diagval=0;
	for(int j=colPtrA[i];j<colPtrA[i+1];j++){
	    if(i==rowIdxA[j])
		diagval=valA[j];
	}
	//if(diagval==0)
		printf("row %d diag val=%lg\n",i,diagval);
    }*/
	//expected_solution[i]=1;
    //mkl_dcscmv ( "N", &nA, &nA, &one, "G**C" , valA , rowIdxA, colPtrA, colPtrA+1,expected_solution, &zero, q);
    int num_threads=1;
    //mkl_set_num_threads(num_threads);
/*#pragma omp parallel
{
    num_threads=omp_get_num_threads();
}*/
    //printf("num_threads= %d\n",num_threads);
    //printf("=========solver start!==========\n");
    //int threadnum=mkl_get_max_threads();
    //printf("num_threads= %d\n",threadnum);
    /*hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;
    hipLaunchKernelGGL(matrixTranspose, dim3(1,1),
                    dim3(4, 4), 0, 0, mA);
    	hipLaunchKernelGGL(test,dim3(1),dim3(512),0,0,mA);*/
    for(int i=0;i<1;i++){
    memset(phi,0,mA*sizeof(double));
    mkl_dcsrmv ( "N", &nA, &nA, &minone, "G**C" , valA , rowIdxA, colPtrA, colPtrA+1,phi, &zero, res);
     cblas_daxpy(nA,one,q,1,res,1);
     double res2 = cblas_ddot(nA,res,1,res,1);
     res2 = sqrt(res2);
    printf("start !!! res0=%f\n",res2);
    gettimeofday(&time1, NULL); 
    solve_base(mA, colPtrA, rowIdxA, valA, q, phi, maxIteration, tol, res0, usediter);
    gettimeofday(&time2, NULL); 
    double elapsed_time = (time2.tv_sec - time1.tv_sec) * 1000. +(time2.tv_usec - time1.tv_usec) / 1000.; 
    printf("Elapsed time: %lf(ms)\n", elapsed_time); 
    mkl_dcsrmv ( "N", &nA, &nA, &minone, "G**C" , valA , rowIdxA, colPtrA, colPtrA+1,phi, &zero, res);
     cblas_daxpy(nA,one,q,1,res,1);
     double res1 = cblas_ddot(nA,res,1,res,1);
     res1 = sqrt(res1);
    printf("Final !!! res0=%f, res=%f, usediter=%d\n",res0, res1,usediter);}
    if(a_p != NULL){free(a_p); a_p=NULL;}
    if(q != NULL){free(q); q=NULL;}
    if(res != NULL){free(res); res=NULL;}
    if(phi != NULL){free(phi); phi=NULL;}
    if(colPtrA != NULL) {free(colPtrA); colPtrA=NULL;}
    if(rowIdxA != NULL) {free(rowIdxA); rowIdxA=NULL;}
    if(valA != NULL){free(valA); valA=NULL;}

    return 0;
}
