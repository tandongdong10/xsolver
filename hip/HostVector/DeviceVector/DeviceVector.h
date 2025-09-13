#ifndef _DEVICEVECTOR_H_
#define _DEVICEVECTOR_H_
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string.h>
#include <assert.h>
#include "hip/hip_runtime.h"
#include <rocblas.h>
#include <sys/time.h>
#include "../HostVector.h"
#include "../../mpicpu.h"
extern rocblas_handle handle;
extern int d_nthread;
__global__ void jacobiInitHIP (const int d_nIntCells, const double *d_a_p, double *d_diag, const double small);
__global__ void jacobiHIP (const int d_nIntCells, const double *d_a, const double *d_b, double *d_c);
__global__ void vecnorm1HIP (const int d_nIntCells, double* __restrict__ d_a,double* __restrict__ result,int num_task, int task_more, int stepSize);
__global__ void kernel1HIP(const int len,const double omega,const double alpha,const double *res,const double *uk,double *pk);
__global__ void kernel2HIP(const int len,const double *res,const double gama,const double *uk,double *sk);
__global__ void kernel3HIP(const int len,const double gama,const double *pk,const double alpha,const double *sk, double *xk);
class DeviceVector:public HostVector{
public:
    int d_nblock_reduce;
    int num_task;
    int allthread;
    int taskmore;
    double *d_result;
    double *temp_result;
    void norm1Init(){
	d_nblock_reduce = n/d_nthread;
    	num_task=1;
    	allthread=d_nthread*d_nblock_reduce;
    	taskmore=n-allthread*num_task;
    	if(d_nblock_reduce>512){
      	    d_nblock_reduce=512;
      	    allthread=d_nthread*d_nblock_reduce;
      	    num_task=n/allthread;
      	    taskmore=n-allthread*num_task;
    	}
	hipMalloc((void **)&d_result, sizeof(double) * d_nblock_reduce);
	temp_result=new double[d_nblock_reduce];	
    }
    void norm1Free(){
	hipFree(d_result);
	delete []temp_result;
    }
    DeviceVector(){
	n=0;
	val=NULL;
    }
    DeviceVector(int Num){
	n=Num;
	hipMalloc((void **)&val, sizeof(double) * n);
	hipMemset(val, 0, sizeof(double) * n);
        norm1Init();
    }
#ifdef HAVE_MPI
    DeviceVector(int n_in,int nHalo_in){
	n=n_in;
	nHalo=nHalo_in;
	hipMalloc((void **)&val, sizeof(double) * (n+nHalo));
	hipMemset(val, 0, sizeof(double) * (n+nHalo));
        norm1Init();
    }
    DeviceVector(int n_in,int nHalo_in,double *v){
	n=n_in;
	nHalo=nHalo_in;
	hipMalloc((void **)&val, sizeof(double) * (n+nHalo));
	hipMemcpy(val, v, sizeof(double) * n, hipMemcpyHostToDevice);
        norm1Init();
    }
#endif
    //HostVector(double *v):
	//val(v){}
    void MallocVector(int n_in){
	n=n_in;
	hipMalloc((void **)&val, sizeof(double) * n);
	hipMemset(val, 0, sizeof(double) * n);
    }
    void MallocVector(int n_in,double *v){
	n=n_in;
	hipMalloc((void **)&val, sizeof(double) * n);
	hipMemcpy(val, v, sizeof(double) * n, hipMemcpyHostToDevice);
    }
#ifdef HAVE_MPI
    void MallocVector(int n_in,int nHalo_in){
	n=n_in;
	nHalo=nHalo_in;
	hipMalloc((void **)&val, sizeof(double) * (n+nHalo));
	hipMemset(val, 0, sizeof(double) * (n+nHalo));
    }
    void MallocVector(int n_in,int nHalo_in,double *v){
	n=n_in;
	nHalo=nHalo_in;
	hipMalloc((void **)&val, sizeof(double) * (n+nHalo));
	hipMemcpy(val, v, sizeof(double) * n, hipMemcpyHostToDevice);
    }
#endif
     DeviceVector(const DeviceVector &hstvec){
        n = hstvec.n; 
#ifdef HAVE_MPI
	nHalo = hstvec.nHalo;
	hipMalloc((void **)&val, sizeof(double) * (n+nHalo));
#else
	hipMalloc((void **)&val, sizeof(double) * n);
#endif
	hipMemcpy(val, hstvec.val, sizeof(double) * n, hipMemcpyDeviceToDevice);
    }
     HostVector &operator = (const HostVector &hstvec){
	if(this == &hstvec){
	    return *this;
	}
        this->n = hstvec.n; 
#ifdef HAVE_MPI
	this->nHalo = hstvec.nHalo;
	hipMalloc((void **)&val, sizeof(double) * (n+nHalo));
#else
	hipMalloc((void **)&val, sizeof(double) * n);
#endif
	hipMemcpy(val, hstvec.val, sizeof(double) * n, hipMemcpyDeviceToDevice);
	return *this;
    }
    void CopyVector(int n_in,double *tmp){
	n=n_in;
	hipMemcpy(val, tmp, sizeof(double) * n, hipMemcpyHostToDevice);
    }
    void SetVector(int n_in,double *tmp){
	n=n_in;
	hipMemcpy(val, tmp, sizeof(double) * n, hipMemcpyHostToDevice);
    }
    void GetVector(double *tmp){
	hipMemcpy(tmp, val, sizeof(double) * n, hipMemcpyDeviceToHost);
    }
    void UpdateVector(int n_in,double *tmp){	
	n=n_in;
	hipMemcpy(val, tmp, sizeof(double) * n, hipMemcpyHostToDevice);
    }
    void vec_dot2(DeviceVector *y,DeviceVector *q,DeviceVector *z, double *res){
	res[0]=0;res[1]=0;
	rocblas_ddot(handle,n,val,1,y->val,1,&(res[0]));
	rocblas_ddot(handle,n,q->val,1,z->val,1,&(res[1]));
#ifdef HAVE_MPI
     	communicator_sum(res,2);
#endif
    }
    double vec_dot(DeviceVector *y){
	double res=0;
	rocblas_ddot(handle,n,val,1,y->val,1,&res);
#ifdef HAVE_MPI
     	communicator_sum(res);
#endif
	return res;	
    }
    double vec_norm1(){
	double res=0;
    	hipLaunchKernelGGL((vecnorm1HIP), dim3(d_nblock_reduce), dim3(d_nthread), sizeof(double) * d_nthread,0,n,val,d_result,num_task,taskmore,allthread);
	hipMemcpy(temp_result, d_result, sizeof(double) * d_nblock_reduce, hipMemcpyDeviceToHost);
	for(int i = 0; i < d_nblock_reduce; ++i) res += temp_result[i];
#ifdef HAVE_MPI
     	communicator_sum(res);
#endif
	return res;	
    }
    void vec_copy(DeviceVector *x){
	rocblas_dcopy(handle,n,x->val,1,val,1);
    }
    void vec_axpy(double alpha, DeviceVector *x){
        rocblas_daxpy(handle, n, &alpha, x->val, 1, val,1);
    }
    void vec_scal(double alpha){
        rocblas_dscal(handle, n, &alpha, val, 1);
    }
    void vec_bicg_kernel1(double omega, double alpha, HostVector *res,HostVector *uk){
        int d_nblock = (n+d_nthread-1)/d_nthread;
	hipLaunchKernelGGL((kernel1HIP),dim3(d_nblock), dim3(d_nthread), 0, 0, n, omega, alpha, res->val, uk->val, val);
    }
    void vec_bicg_kernel2(double gama, HostVector *res,HostVector *uk){
        int d_nblock = (n+d_nthread-1)/d_nthread;
	hipLaunchKernelGGL((kernel2HIP),dim3(d_nblock), dim3(d_nthread), 0, 0, n, res->val, gama, uk->val, val);
    }
    void vec_bicg_kernel3(double gama, double alpha, HostVector *pk,HostVector *sk){
        int d_nblock = (n+d_nthread-1)/d_nthread;
	hipLaunchKernelGGL((kernel3HIP),dim3(d_nblock), dim3(d_nthread), 0, 0, n, gama, pk->val, alpha, sk->val, val);
    }
    void jacobiInit(HostVector *diag_val, double small=0){
        int d_nblock = (n+d_nthread-1)/d_nthread;
        hipLaunchKernelGGL(jacobiInitHIP, dim3(d_nblock), dim3(d_nthread),0,0, n, diag_val->val, val, small);
    } 
    void jacobiSolve(HostVector *x,HostVector *y){
        int d_nblock = (n+d_nthread-1)/d_nthread;
        hipLaunchKernelGGL((jacobiHIP), dim3(d_nblock), dim3(d_nthread), 0, 0, n, val, x->val, y->val);
    }
    void vec_print(){
	double *tmp=new double[n];
	hipMemcpy(tmp, val, sizeof(double) * n, hipMemcpyDeviceToHost);
    	for(int i=0;i<n;i++)
	    printf("val[%d] = %lg  \n",i,tmp[i]);
	delete []tmp;
    }
    void FreeVector(){
	hipFree(val); 
    }
    ~DeviceVector(){
    	 norm1Free();
    }
};
HostVector* set_vector_gpu();
HostVector* set_vector_gpu(int n);
#ifdef HAVE_MPI
HostVector* set_vector_gpu(int n,int nHalo);
#endif
#endif
