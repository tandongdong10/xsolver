#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string>
#include <assert.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <sys/time.h>
#include "HostMatrix.h"
//#include "DeviceMatrixCSR.h"
//#include "DeviceMatrixELL.h"
#include "device_launch_parameters.h"
#ifndef _DEVICEMATRIX_H_
#define _DEVICEMATRIX_H_
cublasHandle_t handle;
int d_nthread = 256;
int d_nblock;// = (nIntCells+d_nthread-1)/d_nthread;
//enum PRECON{ JACOBI = 1,ILU0} precon;
__global__ static void getsendarrayCUDA (const int d_nGhstCells, const double *d_a, const int* d_ptr, double *d_b)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nGhstCells) {
    d_b[icell] = d_a[d_ptr[icell]];//0-based
}
}
__global__ static void jacobiInitCUDA (const int d_nIntCells, const double *d_a_p, double *d_diag, const double small)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    double tmp;
    if (icell < d_nIntCells) {
	tmp=d_a_p[icell]+small;
    	d_diag[icell] = 1.0/tmp;
    }
}

__global__ static void jacobiCUDA (const int d_nIntCells, const double *d_a, const double *d_b, double *d_c)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    	d_c[icell] = d_a[icell] * d_b[icell];
    }
}
class DeviceMatrix{
public:
    int nInterior;
    int nHalo;
    int nSizes;
    int num_nz;
    double *diag_val;
    double *offdiag_val;
    int *offdiag_row_offset;
    int *offdiag_col_index;
    int *ell_idx;
    double *ell_val;
    int *exchange_ptr;
    DeviceMatrix(){
	nInterior=0;
        nHalo=0;
        nSizes=0;
	num_nz=0;
        diag_val=NULL;
        exchange_ptr=NULL;
    }
    virtual void Update(HostMatrix hstmtx){
    	//cudaMemcpy(diag_val, hstmtx.diag_val, sizeof(double) * nInterior, cudaMemcpyHostToDevice);
    }
    virtual void ToDeviceMatrix(HostMatrix hstmtx){}
    void operator=(const DeviceMatrix & rhs){
	//nInterior=rhs.nInterior;
        //nHalo=rhs.nHalo;
        //nSizes=rhs.nSizes;
        //diag_val = rhs.diag_val;//new double[nInterior];
	//exchange_ptr=rhs.exchange_ptr;
    }
    virtual void SpMV(double *x, double *d_x,double *d_y){}
    virtual void bmAx(double *d_q, double *x, double *d_x,double *d_y){}
    virtual void DeviceMatrixFree(){
    	//cudaFree(diag_val);
    	//cudaFree(exchange_ptr);
    }
    virtual ~DeviceMatrix(){}
};

extern void setCSRMatrix();
extern void setELLMatrix();
DeviceMatrix *devicemtx;

double *d_diag;
void preconditionerInit ()
{
    int nIntCells=hostmtx.nInterior;
    if(precon==1){
        cudaMalloc((void **)&d_diag, sizeof(double) * nIntCells);
	double small = 1e-20;
        jacobiInitCUDA<<<dim3(d_nblock), dim3(d_nthread),0,0>>>(nIntCells, devicemtx->diag_val, d_diag, small);
    }
}

void preconditionerUpdate ()
{
    int nIntCells=hostmtx.nInterior;
    if(precon==1){
	double small = 1e-20;
        jacobiInitCUDA<<<dim3(d_nblock), dim3(d_nthread),0,0>>>(nIntCells, devicemtx->diag_val, d_diag, small);
    }
}

void preconditioner (const double *d_a, double *d_b)
{
    if(precon==1){
        jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( hostmtx.nInterior, d_diag, d_a, d_b);
    }
}
void preconditionerFree()
{
    if(precon==1){
        cudaFree(d_diag);
    }
}
extern "C" void gpu_mat_setup(){
    cublasCreate(&handle);
    setCSRMatrix();
    devicemtx->ToDeviceMatrix(hostmtx);
    d_nblock = (hostmtx.nInterior+d_nthread-1)/d_nthread;
}
extern "C" void gpu_mat_format(char* fmt,int &num_nz){
    //char *str_ELL="ELL";
    if(strcmp(fmt,"ELL")==0){
	devicemtx->DeviceMatrixFree();
	setELLMatrix();
	if(num_nz<=16&&num_nz>0)
	    num_nz=16;
	else if(num_nz>16&&num_nz<=32)
	    num_nz=32;
	else{
	    printf("Error : The number of non-zero elements in each row is wrong!!!\n");
	    printf("Error : The number must be between 1 and 32 when converting to ELL format!\n");
	    exit(0);
	}
	devicemtx->num_nz=num_nz;
        devicemtx->ToDeviceMatrix(hostmtx);
    }	
}
extern "C" void gpu_mat_update(){
    devicemtx->Update(hostmtx);
    //preconditionerUpdate();
}
extern "C" void gpu_mat_free(){
    devicemtx->DeviceMatrixFree();
    cublasDestroy(handle);
}
extern "C" void gpu_preconditioner_setup(char *fmt){
    if(strcmp(fmt,"Jacobi")==0){
	precon=JACOBIGPU;
    }
    preconditionerInit();
}
extern "C" void gpu_preconditioner_update(char *fmt){
    if(strcmp(fmt,"Jacobi")==0){
	if(precon!=JACOBIGPU)
	    printf("Preconditioner is not matched!\n");
    }
    preconditionerUpdate();
}
extern "C" void gpu_preconditioner_free( char *fmt){
    preconditionerFree();
}
#endif
