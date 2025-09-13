#ifndef _DEVICEMATRIXCSR_H_
#define _DEVICEMATRIXCSR_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hip/hip_runtime.h"
#include <rocblas.h>
#include <assert.h>
#include <sys/time.h>
#include "../HostMatrixCSR.h"
#include "DeviceMatrix.h"
#include "../../Precond/gpuparilu.h"
#include "../../Precond/parilut.h"
#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

__global__ void getCSRdiag (const int d_nIntCells, const int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double* diag);
__global__ void LMultCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *d_y);
__global__ void lsolveiterCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *tmp, double *d_y);
__global__ void UMultCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *d_y);
__global__ void usolveiterCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *tmp, double *d_y, double *diagu);
__global__ void matMultCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *d_y);
__global__ void bmAxCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double *d_q,double*d_x, double *d_y);
__global__ void GetDiagCSRptr (const int d_nIntCells, const int onebase, const int* d_ptr,const int* d_idx, const double* val, int *d_ptr_new, int *d_idx_new, double *d_val_new);
__global__ void GetDiagCSR (const int d_nIntCells, const int onebase, const int* d_ptr,const int* d_idx, const double* d_val, int *d_ptr_new, int *d_idx_new, double *d_val_new);
__global__ void SetCSR0basedPtr (const int d_nIntCells, int* d_ptr);
__global__ void SetCSR0based (const int d_nIntCells, int* d_ptr,int* d_idx);
class DeviceMatrixCSR:public DeviceMatrix{
public:
    int onebase=0;
#ifdef HAVE_MPI
    int *exchange_ptr;
    double *x_nHalo;
#endif
    double *val;
    int *rowptr;
    int *colidx;
    DeviceMatrixCSR(){
        val=NULL;
	rowptr=NULL;
	colidx=NULL;
#ifdef HAVE_MPI
	exchange_ptr=NULL;
	x_nHalo=NULL;
#endif
    }
    MATRIXFORMAT getmatrixformat(){
	return GPUCSR;
    }
    double* getval(){
	return val;}
    int* getptr(){
	return rowptr;}
    int* getidx(){
	return colidx;}
    int getonebase(){
	return onebase;}
    void ToDeviceMatrix(HostMatrix *hstmtx){
	ToDeviceMatrixCSR(hstmtx);
    }
    void ToDeviceMatrixCSR(HostMatrix *hstmtx){
	m=hstmtx->m;
	n=hstmtx->n;
	onebase=hstmtx->getptr()[0];
	if(hstmtx->getmatrixformat()!=CSR){
	    printf("HostMatrix Format is not CSR!!!Cannot copy to DeviceMatrixCSR!!!\n");
	    exit(0);
	}
#ifdef HAVE_MPI
        nHalo=hstmtx->nHalo;
    	x_nHalo=(double *)malloc((n+nHalo)*sizeof(double));
#endif
        nnz=hstmtx->nnz;
    	//if(hipSuccess != hipMalloc((void **)&val, sizeof(double) * nInterior))
	//	printf("WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    	CHECK(hipMalloc((void **)&val, sizeof(double) * nnz));
    	CHECK(hipMalloc((void **)&rowptr, sizeof(int) * (m + 1)));
    	CHECK(hipMalloc((void **)&colidx, sizeof(int) * nnz));
#ifdef HAVE_MPI
    	hipMalloc((void **)&exchange_ptr, sizeof(int) * nHalo);
#endif
    	hipMemcpy(val, hstmtx->getval(), sizeof(double) * nnz, hipMemcpyHostToDevice);
    	hipMemcpy(rowptr, hstmtx->getptr(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);//??????????????????????
    	hipMemcpy(colidx, hstmtx->getidx(), sizeof(int) * nnz, hipMemcpyHostToDevice);
#ifdef HAVE_MPI
    	hipMemcpy(exchange_ptr, hstmtx->exchange_ptr, sizeof(int) * nHalo, hipMemcpyHostToDevice);
#endif
    }
    void Update(HostMatrix hstmtx){
    	hipMemcpy(val, hstmtx.getval(), sizeof(double) * nnz, hipMemcpyHostToDevice);
    }
    void create_matrix(int n_in,double *val_in,int *rowptr_in, int *colidx_in){
	HostMatrix *hstmtx=new HostMatrixCSR();
	hstmtx->create_matrix(n_in,val_in,rowptr_in,colidx_in);
	onebase=rowptr_in[0];
	ToDeviceMatrixCSR(hstmtx);
	delete hstmtx;
    }
    void MallocMatrix(int n_in,int nnz_in){
	m=n=n_in;
	nnz=nnz_in;
    	hipMalloc((void **)&val, sizeof(double) * nnz);
    	hipMalloc((void **)&rowptr, sizeof(int) * (m + 1));
    	hipMalloc((void **)&colidx, sizeof(int) * nnz);
    }
#ifdef HAVE_MPI
    void create_matrix(int m_in, int nHalo_in,double *val_in,int *rowptr_in, int *colidx_in){
	onebase=rowptr_in[0];
	HostMatrix *hstmtx=new HostMatrixCSR();
	hstmtx->create_matrix(m_in,nHalo_in,val_in,rowptr_in,colidx_in);
	ToDeviceMatrixCSR(hstmtx);
	delete hstmtx;
    }
    void MallocMatrix(int m_in,int nHalo_in,int nnz_in){
	m=n=m_in;
	nHalo=nHalo_in;
	nnz=nnz_in;
    	hipMalloc((void **)&val, sizeof(double) * nnz);
    	hipMalloc((void **)&rowptr, sizeof(int) * (m + 1));
    	hipMalloc((void **)&colidx, sizeof(int) * nnz);
    }
#else
    void create_matrix(int m_in, int n_in,double *val_in,int *rowptr_in, int *colidx_in){
	m=m_in;
	n=n_in;
	onebase=rowptr_in[0];
	nnz=rowptr_in[m]-onebase;
	HostMatrix *hstmtx=new HostMatrixCSR();
	hstmtx->create_matrix(m_in,n_in,val_in,rowptr_in,colidx_in);
	ToDeviceMatrixCSR(hstmtx);
	delete hstmtx;
    }
    void MallocMatrix(int m_in,int n_in,int nnz_in){
	m=m_in;
	n=n_in;
	nnz=nnz_in;
    	hipMalloc((void **)&val, sizeof(double) * nnz);
    	hipMalloc((void **)&rowptr, sizeof(int) * (m + 1));
    	hipMalloc((void **)&colidx, sizeof(int) * nnz);
    }
#endif 
    void CopyMatrix(HostMatrix *hostmtx){
	if(hostmtx->getmatrixformat()!=GPUCSR){
	    printf("Wrong!!! copy matrix is not gpu csr!!!\n");
	    exit(0);
	}
	m=hostmtx->m;
	n=hostmtx->n;
	nnz=hostmtx->nnz;
	onebase=hostmtx->getonebase();
    	hipMemcpy(val, hostmtx->getval(), sizeof(double) * nnz, hipMemcpyDeviceToDevice);
    	hipMemcpy(rowptr, hostmtx->getptr(), sizeof(int) * (m + 1), hipMemcpyDeviceToDevice);//??????????????????????
    	hipMemcpy(colidx, hostmtx->getidx(), sizeof(int) * nnz, hipMemcpyDeviceToDevice);
    }
    void SetMatrix(int m_in,int n_in,int nnz_in ,int *&ptr_in,int *&idx_in,double *&val_in){
	m=m_in;
	n=n_in;
	nnz=nnz_in;
	rowptr=ptr_in;
	colidx=idx_in;
	val=val_in;
    	//hipMalloc((void **)&val, sizeof(double) * nnz);
    	//hipMalloc((void **)&rowptr, sizeof(int) * (m + 1));
    	//hipMalloc((void **)&colidx, sizeof(int) * nnz);
    	//hipMemcpy(val, val_in, sizeof(double) * nnz, hipMemcpyDeviceToDevice);
    	//hipMemcpy(rowptr, ptr_in, sizeof(int) * (m + 1), hipMemcpyDeviceToDevice);//??????????????????????
    	//hipMemcpy(colidx, idx_in, sizeof(int) * nnz, hipMemcpyDeviceToDevice);
    }
    void based1To0Matrix(){
	if(onebase==0)
	    return;
	onebase=0;
        int d_nblockptr = (m+d_nthread)/d_nthread;
        int d_nblock = (m+d_nthread-1)/d_nthread;
    	hipLaunchKernelGGL(SetCSR0basedPtr,dim3(d_nblockptr),dim3(d_nthread),0,0,m+1,rowptr);
    	hipLaunchKernelGGL(SetCSR0based,dim3(d_nblock),dim3(d_nthread),0,0,m,rowptr, colidx);
	
    }
    void operator=(DeviceMatrixCSR & rhs){
	m=rhs.m;
	n=rhs.n;
        nnz=rhs.nnz;
        val = rhs.getval();//new double[nSizes],
        rowptr=rhs.getptr();
        colidx=rhs.getidx();
	onebase=rhs.onebase;
#ifdef HAVE_MPI
        nHalo=rhs.nHalo;
	exchange_ptr=rhs.exchange_ptr;
#endif
    }
    void getdiag(double *a_p){
        int d_nblock = (m+d_nthread-1)/d_nthread;
    	hipLaunchKernelGGL(getCSRdiag,dim3(d_nblock),dim3(d_nthread),0,0,n,onebase, rowptr, colidx, val, a_p);
    }
    void getdiag(HostVector *a_p_vec){
	double *a_p=a_p_vec->val;
        int d_nblock = (m+d_nthread-1)/d_nthread;
    	hipLaunchKernelGGL(getCSRdiag,dim3(d_nblock),dim3(d_nthread),0,0,n,onebase, rowptr, colidx, val, a_p);
    }
    void SpMV(HostVector *d_x,HostVector *d_y);
    void bmAx(HostVector *d_rhs, HostVector *d_x, HostVector *d_y);
    void ToDiagMatrix(HostMatrix *hostmtxold);
    void parilu(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU,int **row_referenced, int sweep);
    void parilut(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU, int sweep);
    void parilu_csr(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU,int **row_referenced, HostVector *diag_U,int sweep);
    void parilut_csr(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU, HostVector *diag_U,int sweep);
    void Lsolve(HostVector *x,HostVector *y);
    void Usolve(HostVector *x,HostVector *y);
    void Lsolve_iter(HostVector *x,HostVector *y,HostVector *tmp,int maxiter);
    void Usolve_iter(HostVector *x,HostVector *y,HostVector *tmp,HostVector *diag_U,int maxiter);
    void bmAx(DeviceVector *q, DeviceVector *x, DeviceVector *y);
    void FreeMatrix(){
	DeviceMatrixCSRFree();
    }
    void DeviceMatrixCSRFree(){
    	hipFree(val);
    	hipFree(rowptr);
    	hipFree(colidx);
#ifdef HAVE_MPI
    	if(exchange_ptr!=NULL)
	    hipFree(exchange_ptr);
	if(x_nHalo!=NULL)
	   free(x_nHalo); 
#endif
    }
    ~DeviceMatrixCSR(){}
};
HostMatrix* set_matrix_gpu_csr();

#endif
