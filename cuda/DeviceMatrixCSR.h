#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <sys/time.h>
#include "HostMatrix.h"
#include "DeviceMatrix.h"
#include "device_launch_parameters.h"
#ifndef _DEVICEMATRIXCSR_H_
#define _DEVICEMATRIXCSR_H_

__global__ static void matMultCSR (const int d_nIntCells, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const int d_nthread,const double*d_a_p,const double*d_a_l, double*d_x, double *d_y)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    double r = 0.0;
    for (int iside = d_NbCell_ptr_c[icell]; iside < d_NbCell_ptr_c[icell+1]; iside++)
    {
        r += d_a_l[iside] * d_x[d_NbCell_s[iside]];//0-based
    }
	
    d_y[icell] =d_a_p[icell] * d_x[icell] + r;//diag?
}
}
__global__ static void bmAxCSR (const int d_nIntCells, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const int d_nthread,const double*d_a_p,const double*d_a_l, double *d_q,double*d_x, double *d_y)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    double r = d_q[icell];
    for (int iside = d_NbCell_ptr_c[icell]; iside < d_NbCell_ptr_c[icell+1]; iside++)
    {
        r -= d_a_l[iside] * d_x[d_NbCell_s[iside]];//0-based
    }
	
    d_y[icell] = r - d_a_p[icell] * d_x[icell];//diag?
  }
}
class DeviceMatrixCSR:public DeviceMatrix{
public:
    DeviceMatrixCSR(){
	nInterior=0;
        nHalo=0;
        nSizes=0;
        diag_val=NULL;
        offdiag_val=NULL;
        offdiag_row_offset=NULL;
        offdiag_col_index=NULL;
        exchange_ptr=NULL;
    }
    void ToDeviceMatrix(HostMatrix hstmtx){
	ToDeviceMatrixCSR(hstmtx);
    }
    void ToDeviceMatrixCSR(HostMatrix hstmtx){
	nInterior=hstmtx.nInterior;
        nHalo=hstmtx.nHalo;
        nSizes=hstmtx.nSizes;
    	cudaMalloc((void **)&diag_val, sizeof(double) * nInterior);
    	cudaMalloc((void **)&offdiag_val, sizeof(double) * nSizes);
    	cudaMalloc((void **)&offdiag_row_offset, sizeof(int) * (nInterior + 1));
    	cudaMalloc((void **)&offdiag_col_index, sizeof(int) * nSizes);
    	cudaMalloc((void **)&exchange_ptr, sizeof(int) * nHalo);
    	cudaMemcpy(diag_val, hstmtx.diag_val, sizeof(double) * nInterior, cudaMemcpyHostToDevice);
    	cudaMemcpy(offdiag_val, hstmtx.offdiag_val, sizeof(double) * nSizes, cudaMemcpyHostToDevice);
    	cudaMemcpy(offdiag_row_offset, hstmtx.offdiag_row_offset, sizeof(int) * (nInterior + 1), cudaMemcpyHostToDevice);//??????????????????????
    	cudaMemcpy(offdiag_col_index, hstmtx.offdiag_col_index, sizeof(int) * nSizes, cudaMemcpyHostToDevice);
    	cudaMemcpy(exchange_ptr, hstmtx.exchange_ptr, sizeof(int) * nHalo, cudaMemcpyHostToDevice);
    }
    void Update(HostMatrix hstmtx){
    	cudaMemcpy(diag_val, hstmtx.diag_val, sizeof(double) * nInterior, cudaMemcpyHostToDevice);
    	cudaMemcpy(offdiag_val, hstmtx.offdiag_val, sizeof(double) * nSizes, cudaMemcpyHostToDevice);
    	//cudaMemcpy(offdiag_row_offset, hstmtx.offdiag_row_offset, sizeof(int) * (nInterior + 1), cudaMemcpyHostToDevice);//??????????????????????
    }
    void operator=(const DeviceMatrixCSR & rhs){
	nInterior=rhs.nInterior;
        nHalo=rhs.nHalo;
        nSizes=rhs.nSizes;
        diag_val = rhs.diag_val;//new double[nInterior];
        offdiag_val = rhs.offdiag_val;//new double[nSizes],
        offdiag_row_offset=rhs.offdiag_row_offset;
        offdiag_col_index=rhs.offdiag_col_index;
	exchange_ptr=rhs.exchange_ptr;
    }
    void SpMV(double *x, double *d_x,double *d_y);
    void bmAx(double *d_q, double *x, double *d_x,double *d_y);
    void DeviceMatrixFree(){
	DeviceMatrixCSRFree();
    }
    void DeviceMatrixCSRFree(){
    	cudaFree(diag_val);
    	cudaFree(offdiag_val);
    	cudaFree(offdiag_row_offset);
    	cudaFree(offdiag_col_index);
    	cudaFree(exchange_ptr);
    }
    ~DeviceMatrixCSR(){}
};
DeviceMatrixCSR devicemtxcsr;
void setCSRMatrix(){
    devicemtx=&devicemtxcsr;
}
void DeviceMatrixCSR::SpMV(double *x, double *d_x,double *d_y)
{
    int d_nblock = (nInterior+d_nthread-1)/d_nthread;
#ifdef HAVE_MPI
    getsendarrayCUDA<<<dim3(d_nblock),dim3(d_nthread)>>>( nHalo, d_x, exchange_ptr, d_x+nInterior);
    cudaMemcpy(x+nInterior, d_x+nInterior, sizeof(double)*(nHalo), cudaMemcpyDeviceToHost);
    for(int i=0;i<nHalo;i++)
	x[topo_c.exchange_ptr[i]]=x[nInterior+i];//0-based
    communicator_p2p(x);
    communicator_p2p_waitall();
    cudaMemcpy(d_x+nInterior, x+nInterior, sizeof(double)*nHalo, cudaMemcpyHostToDevice);
    //matMultELL<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( d_nIntCells, num_nz, ell_val, d_a_p, ell_idx, d_x, d_y);
#endif
    matMultCSR<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nInterior, offdiag_row_offset, offdiag_col_index, d_nthread, diag_val, offdiag_val, d_x, d_y);
}
void DeviceMatrixCSR::bmAx(double *d_q, double *x, double *d_x,double *d_y)
{
    int d_nblock = (nInterior+d_nthread-1)/d_nthread;
#ifdef HAVE_MPI
    getsendarrayCUDA<<<dim3(d_nblock),dim3(d_nthread)>>>( nHalo, d_x, exchange_ptr, d_x+nInterior);
    cudaMemcpy(x+nInterior, d_x+nInterior, sizeof(double)*(nHalo), cudaMemcpyDeviceToHost);
    for(int i=0;i<nHalo;i++)
	x[topo_c.exchange_ptr[i]]=x[nInterior+i];//0-based
    communicator_p2p(x);
    communicator_p2p_waitall();
    cudaMemcpy(d_x+nInterior, x+nInterior, sizeof(double)*nHalo, cudaMemcpyHostToDevice);
#endif
    bmAxCSR<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nInterior, offdiag_row_offset, offdiag_col_index, d_nthread, diag_val, offdiag_val, d_q, d_x, d_y);
}
#endif
