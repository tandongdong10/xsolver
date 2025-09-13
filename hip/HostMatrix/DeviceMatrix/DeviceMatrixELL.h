#ifndef __DEVICEMATRIXELL_H_
#define __DEVICEMATRIXELL_H_
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string.h>
#include "hip/hip_runtime.h"
#include <rocblas.h>
#include <sys/time.h>
#include "../HostMatrix.h"
#include "DeviceMatrix.h"

__global__ void matMultELL( const int nrow, const int nz, const int onebase, const double *val, const int *idx, double *d_x, double *d_y);
__global__ void bmAxELL( const int nrow, const int nz, const int onebase, const double *val,  const int *idx, double *d_q, double *d_x, double *d_y);
class DeviceMatrixELL:public DeviceMatrix{
public:
    int onebase=0;
#ifdef HAVE_MPI    
    double *x_nHalo;
    int *exchange_ptr;
#endif
    int valSizes;// = nInterior * num_nz;
    int num_nz;
    int *ell_idx;
    double *ell_val;
    DeviceMatrixELL(){
        ell_idx=NULL;
        ell_val=NULL;
#ifdef HAVE_MPI
	exchange_ptr=NULL;
	x_nHalo=NULL;
#endif
    }
    MATRIXFORMAT getmatrixformat(){
	return GPUELL;
    }
    double* getval(){
	return ell_val;}
    int *getidx(){
	return ell_idx;}
    int getonebase(){
	return onebase;}
    void ToDeviceMatrix(HostMatrix *hstmtx){
	CSRToDeviceELL(hstmtx);
    }
    void CSRToDeviceELL(HostMatrix *mtxcsr){
	m=mtxcsr->m;
	n=mtxcsr->n;
	onebase=mtxcsr->getptr()[0];
	if(mtxcsr->getmatrixformat()!=CSR){
	    printf("Matrix Format is not CPU CSR!!!Cannot trans to DeviceMatrixELL!!!\n");
	    exit(0);
	}
#ifdef HAVE_MPI
        nHalo=mtxcsr->nHalo;
    	x_nHalo=(double *)malloc((n+nHalo)*sizeof(double));
#endif
        nnz=mtxcsr->nnz;
	num_nz=16;
        valSizes = n * num_nz;
    	hipMalloc((void **)&ell_idx, sizeof(int) * valSizes);
    	hipMalloc((void **)&ell_val, sizeof(double) * valSizes);
#ifdef HAVE_MPI
    	hipMalloc((void **)&exchange_ptr, sizeof(int) * nHalo);
    	hipMemcpy(exchange_ptr, mtxcsr->exchange_ptr, sizeof(int) * nHalo, hipMemcpyHostToDevice);
#endif
	InsertZero(mtxcsr);
    }
    void operator=(const DeviceMatrixELL & rhs){
	m=rhs.m;
	n=rhs.n;
#ifdef HAVE_MPI
        nHalo=rhs.nHalo;
	exchange_ptr=rhs.exchange_ptr;
#endif
        nnz=rhs.nnz;
	num_nz=rhs.num_nz;
        ell_idx=rhs.ell_idx;
        ell_val=rhs.ell_val;
    }
    void InsertZero(HostMatrix *mtxcsr){
	InsertZero0(mtxcsr);
	InsertZero1(mtxcsr);
    }
    void InsertZero0(HostMatrix *mtxcsr);
    void InsertZero1(HostMatrix *mtxcsr);
    void Update(HostMatrix *hstmtx);
    void SpMV(HostVector *x,HostVector *y);
    void bmAx(HostVector *q,HostVector *x,HostVector *y);
    void FreeMatrix(){
	DeviceMatrixELLFree();
    }
    void DeviceMatrixELLFree(){
    	hipFree(ell_idx);
    	hipFree(ell_val);
#ifdef HAVE_MPI
    	if(exchange_ptr!=NULL)
	    hipFree(exchange_ptr);
	if(x_nHalo!=NULL)
	   free(x_nHalo); 
#endif
    }
    ~DeviceMatrixELL(){}
};

HostMatrix* set_matrix_gpu_ell();

template<class T>
__global__ void TransposeMatrix_kernel(T * matCSR, T * matELL, int nrow);
template<class T>
__global__ void TransposeMatrix_kernel_32(T * matCSR, T * matELL, int nrow);
#endif
