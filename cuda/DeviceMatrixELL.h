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
#include "device_launch_parameters.h"
#ifndef _DEVICEMATRIXELL_H_
#define _DEVICEMATRIXELL_H_

__global__ void matMultELL( const int nrow, const int nz, const double *val, const double *diag_val, const int *idx, double *d_x, double *d_y)
{
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(rowIdx < nrow)
  {
    double dot = 0;
    for(int i=0; i<nz; i++)
    {
      int pos = nrow * i + rowIdx;
      double tmp = val[pos];
      int col = idx[pos];//0-based
      dot += tmp * d_x[col];
    }
    dot += diag_val[rowIdx] * d_x[rowIdx];
    d_y[rowIdx] = dot;
  }
}
__global__ void bmAxELL( const int nrow, const int nz, const double *val, const double *diag_val, const int *idx, double *d_q, double *d_x, double *d_y)
{
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(rowIdx < nrow)
  {
    double dot = d_q[rowIdx];
    for(int i=0; i<nz; i++)
    {
      int pos = nrow * i + rowIdx;
      double tmp = val[pos];
      int col = idx[pos];//0-based
      dot -= tmp * d_x[col];
    }
    dot -= diag_val[rowIdx] * d_x[rowIdx];
    d_y[rowIdx] = dot;
  }
}
class DeviceMatrixELL:public DeviceMatrix{
public:
    //int nInterior;
    //int nHalo;
    //int nSizes;
    int valSizes;// = nInterior * num_nz;
    //int num_nz;
    //double *diag_val;
    //int *ell_idx;
    //double *ell_val;
    //int *exchange_ptr;
    DeviceMatrixELL(){
	nInterior=0;
        nHalo=0;
        nSizes=0;
        valSizes=0;
 	num_nz=0;
        diag_val=NULL;
        ell_idx=NULL;
        ell_val=NULL;
        exchange_ptr=NULL;
    }
    void ToDeviceMatrix(HostMatrix hstmtx){
	CSRToDeviceELL(hstmtx);
    }
    void CSRToDeviceELL(HostMatrix mtxcsr){
	nInterior=mtxcsr.nInterior;
        nHalo=mtxcsr.nHalo;
        nSizes=mtxcsr.nSizes;
	//num_nz=16;
        valSizes = nInterior * num_nz;
    	cudaMalloc((void **)&diag_val, sizeof(double) * nInterior);
    	cudaMalloc((void **)&ell_idx, sizeof(int) * valSizes);
    	cudaMalloc((void **)&ell_val, sizeof(double) * valSizes);
    	cudaMalloc((void **)&exchange_ptr, sizeof(int) * nHalo);
    	cudaMemcpy(diag_val, mtxcsr.diag_val, sizeof(double) * nInterior, cudaMemcpyHostToDevice);
	InsertZero(mtxcsr);
    	cudaMemcpy(exchange_ptr, mtxcsr.exchange_ptr, sizeof(int) * nHalo, cudaMemcpyHostToDevice);
    }
    void operator=(const DeviceMatrixELL & rhs){
	nInterior=rhs.nInterior;
        nHalo=rhs.nHalo;
        nSizes=rhs.nSizes;
	num_nz=rhs.num_nz;
        diag_val = rhs.diag_val;//new double[nInterior];
        ell_idx=rhs.ell_idx;
        ell_val=rhs.ell_val;
	exchange_ptr=rhs.exchange_ptr;
    }
    void InsertZero(const HostMatrix mtxcsr){
	InsertZero0(mtxcsr);
	InsertZero1(mtxcsr);
    }
    void InsertZero0(const HostMatrix mtxcsr);
    void InsertZero1(const HostMatrix mtxcsr);
    void Update(HostMatrix hstmtx);
    void SpMV(double *x, double *d_x,double *d_y);
    void bmAx(double *d_q, double *x, double *d_x,double *d_y);
    void DeviceMatrixFree(){
	DeviceMatrixELLFree();
    }
    void DeviceMatrixELLFree(){
    	cudaFree(diag_val);
    	cudaFree(ell_idx);
    	cudaFree(ell_val);
    	cudaFree(exchange_ptr);
    }
    ~DeviceMatrixELL(){}
};
DeviceMatrixELL devicemtxell;

template<class T>
__global__ void TransposeMatrix_kernel(T * matCSR, T * matELL, int nrow)
{
  int rowIndex = blockIdx.x * 16 + threadIdx.y;
  int colIndex = threadIdx.x;
  int x = threadIdx.x, y = threadIdx.y;
  __shared__ T temp[16][16];
  if(rowIndex < nrow){ 
    temp[y][x] = (matCSR + rowIndex * 16)[colIndex];
    __syncthreads();
    matELL[x * nrow + y + blockIdx.x * 16] = temp[y][x];
  }
}

template<class T>
__global__ void TransposeMatrix_kernel_32(T * matCSR, T * matELL, int nrow)
{
  int rowIndex = blockIdx.x * 32 + threadIdx.y;
  int colIndex = threadIdx.x;
  int x = threadIdx.x, y = threadIdx.y;
  __shared__ T temp[32][32];
  if(rowIndex < nrow){ 
    temp[y][x] = (matCSR + rowIndex * 32)[colIndex];
    __syncthreads();
    matELL[x * nrow + y + blockIdx.x * 32] = temp[y][x];
  }
}
void DeviceMatrixELL::Update(HostMatrix hstmtx){
    cudaMemcpy(diag_val, hstmtx.diag_val, sizeof(double) * nInterior, cudaMemcpyHostToDevice);
    InsertZero1(hstmtx);
}

void DeviceMatrixELL::InsertZero0(const HostMatrix mtxcsr)
{	
  int nInterior=mtxcsr.nInterior;
  //int nHalo=mtxcsr.nHalo;
  //int nSizes=mtxcsr.nSizes;
  int valSizes = nInterior * num_nz;
  int *idx = new int[valSizes];//(int *)malloc(valsize*sizeof(int));//new int[valsize];
  for(int i=0;i<valSizes;i++){
    idx[i]=1;
  }
  int *ptr=new int[nInterior+1];//(int *)malloc((nIntCells+1)*sizeof(int));//new int[nIntCells+1];
  ptr[0]=0;//0-based
  int nz_cnt = 0;
  int *NbCell_ptr_c=mtxcsr.offdiag_row_offset;
  int *NbCell_s=mtxcsr.offdiag_col_index;
  int *d_NbCell_ptr_c,*d_NbCell_s;
  //printf("NbCell_ptr_c[0]=%d\n",NbCell_ptr_c[0]);
  //printf("NbCell_s[0]=%d\n",NbCell_s[0]);
  for(int i=0; i<nInterior; i++)
  {
    nz_cnt = NbCell_ptr_c[i+1] - NbCell_ptr_c[i];
    if(nz_cnt>num_nz){
	printf("Error : The number of non-zero elements in each row is wrong!!!\n");
	printf("Error : The number is larger than what you inputed!\n");
	exit(0);
    }
    ptr[i+1]=num_nz+ptr[i];
    memcpy(&idx[i*num_nz], &NbCell_s[NbCell_ptr_c[i]], nz_cnt * sizeof(int));//0-based
  }
    cudaMalloc((void **)&d_NbCell_ptr_c, sizeof(int) * (nInterior + 1));
    cudaMalloc((void **)&d_NbCell_s, sizeof(int) * valSizes);
  //for(int i=0;i<2*32;i++){
  //  printf("idx[%d]=%d,val[%d]=%f\n",i,idx[i],i,val[i]);
  //}
    cudaMemcpy(d_NbCell_ptr_c, ptr, sizeof(int) * (nInterior + 1), cudaMemcpyHostToDevice);//??????????????????????
    cudaMemcpy(d_NbCell_s, idx, sizeof(int) * valSizes, cudaMemcpyHostToDevice);
  dim3 threads2(num_nz, num_nz);
  dim3 block2( (nInterior+num_nz-1)/num_nz );
  if(num_nz==16) 
  	TransposeMatrix_kernel <<<block2, threads2, 0>>>( d_NbCell_s, ell_idx, nInterior);
  else
  	TransposeMatrix_kernel_32 <<<block2, threads2, 0>>>( d_NbCell_s, ell_idx, nInterior);
  cudaDeviceSynchronize();
  delete []idx;
  delete []ptr;
  cudaFree(d_NbCell_ptr_c);
  cudaFree(d_NbCell_s);
}
void DeviceMatrixELL::InsertZero1(const HostMatrix mtxcsr)
{
  int nInterior=mtxcsr.nInterior;
  //int nHalo=mtxcsr.nHalo;
  //int nSizes=mtxcsr.nSizes;
  int valSizes = nInterior * num_nz;
  double *val = new double[valSizes];//(double *)malloc(valsize*sizeof(double));//new double[valsize];
  double *d_a_l;
  for(int i=0;i<valSizes;i++){
    val[i]=0;
  }
  int nz_cnt = 0;
  int *NbCell_ptr_c=mtxcsr.offdiag_row_offset;
  double *a_l=mtxcsr.offdiag_val;
  cudaMalloc((void **)&d_a_l, sizeof(double) * valSizes);
  for(int i=0; i<nInterior; i++)
  {
    nz_cnt = NbCell_ptr_c[i+1] - NbCell_ptr_c[i];
    memcpy(&val[i*num_nz], &a_l[NbCell_ptr_c[i]], nz_cnt * sizeof(double));//0-based
  }
    cudaMemcpy(d_a_l, val, sizeof(double) * valSizes, cudaMemcpyHostToDevice);

  cudaMemset(ell_val,0,sizeof(double) * valSizes);
  dim3 threads2(num_nz, num_nz);
  dim3 block2( (nInterior+num_nz-1)/num_nz ); 
  if(num_nz==16) 
  	TransposeMatrix_kernel <<<block2, threads2, 0>>>( d_a_l, ell_val, nInterior);
  else
  	TransposeMatrix_kernel_32 <<<block2, threads2, 0>>>( d_a_l, ell_val, nInterior);
  cudaDeviceSynchronize();
  delete []val;
  cudaFree(d_a_l);
}

void DeviceMatrixELL::SpMV(double *x, double *d_x,double *d_y)
{
    int nblock = (nInterior+d_nthread-1)/d_nthread;
#ifdef HAVE_MPI
    getsendarrayCUDA<<<dim3(nblock),dim3(d_nthread)>>>( nHalo, d_x, exchange_ptr, d_x+nInterior);
    cudaMemcpy(x+nInterior, d_x+nInterior, sizeof(double)*(nHalo), cudaMemcpyDeviceToHost);
    for(int i=0;i<nHalo;i++)
	x[topo_c.exchange_ptr[i]]=x[nInterior+i];//0-based
    communicator_p2p(x);
    communicator_p2p_waitall();
    cudaMemcpy(d_x+nInterior, x+nInterior, sizeof(double)*nHalo, cudaMemcpyHostToDevice);
#endif
    matMultELL<<<dim3(nblock), dim3(d_nthread), 0, 0>>>( nInterior, num_nz, ell_val, diag_val, ell_idx, d_x, d_y);
}
void DeviceMatrixELL::bmAx(double *d_q, double *x, double *d_x,double *d_y)
{
    int nblock = (nInterior+d_nthread-1)/d_nthread;
#ifdef HAVE_MPI
    getsendarrayCUDA<<<dim3(nblock),dim3(d_nthread)>>>( nHalo, d_x, exchange_ptr, d_x+nInterior);
    cudaMemcpy(x+nInterior, d_x+nInterior, sizeof(double)*(nHalo), cudaMemcpyDeviceToHost);
    for(int i=0;i<nHalo;i++)
	x[topo_c.exchange_ptr[i]]=x[nInterior+i];//0-based
    communicator_p2p(x);
    communicator_p2p_waitall();
    cudaMemcpy(d_x+nInterior, x+nInterior, sizeof(double)*nHalo, cudaMemcpyHostToDevice);
#endif
    bmAxELL<<<dim3(nblock),dim3(d_nthread), 0, 0>>>( nInterior, num_nz, ell_val, diag_val, ell_idx, d_q, d_x, d_y);
}
void setELLMatrix(){
    devicemtx=&devicemtxell;
}
#endif
