#ifndef _DEVICEMATRIX_H_
#define _DEVICEMATRIX_H_
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string>
#include "hip/hip_runtime.h"
#include"hipsparse.h"
#include <rocblas.h>
#include <sys/time.h>
#include "../HostMatrix.h"
#include "../../Precond/parilut.h"
#include "../../mpicpu.h"
extern rocblas_handle handle;
extern hipsparseHandle_t handle1;
extern hipStream_t stream[13];
extern int d_nthread;
extern int d_nblock;// = (nIntCells+d_nthread-1)/d_nthread;
__global__ void getsendarrayHIP (const int d_nGhstCells, const int onebase, const double *d_a, const int* d_ptr, double *d_b);
class DeviceMatrix:public HostMatrix{
public:
    int nSizes;
    int num_nz;
    DeviceMatrix(){
	m=0;
        n=0;
        nnz=0;
	num_nz=0;
    }
    virtual double* getval(){return nullptr;};
    virtual int* getptr(){return nullptr;};
    virtual int* getidx(){return nullptr;};
    virtual void Update(HostMatrix hstmtx){};
    virtual void ToDeviceMatrix(HostMatrix *hstmtx){};
    virtual void based1To0Matrix(){};
    virtual void SetMatrix(int m_in,int n_in,int nnz_in ,int *ptr_in,int *idx_in,double *val_in){};
    virtual void SpMV(HostVector *d_x,HostVector *d_y){};
    virtual void bmAx(HostVector *d_rhs, HostVector *d_x, HostVector *d_y){};
    virtual void FreeMatrix(){
    };
    virtual ~DeviceMatrix(){};
};

void setupgpu(HostMatrix *hostmtx,int precon);
void freegpu(int precon);
#endif
