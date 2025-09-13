#include "DeviceVector.h"
__global__ void jacobiInitHIP (const int d_nIntCells, const double *d_a_p, double *d_diag, const double small)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    double tmp;
    if (icell < d_nIntCells) {
	tmp=d_a_p[icell]+small;
	/*if(tmp==0){
	    printf("The Matrix val[%d][%d]==0, Jacobi can not work!!!\n",icell,icell);
	    printf("Please change preconditioner\n");
	    //exit(0);
	}*/
    	d_diag[icell] = 1.0/tmp;
    }
}

//template <typename T>
__global__ void jacobiHIP (const int d_nIntCells, const double *d_a, const double *d_b, double *d_c)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    	d_c[icell] = d_a[icell] * d_b[icell];
    }
}
__global__ void vecnorm1HIP (const int d_nIntCells, double* __restrict__ d_a,double* __restrict__ result,int num_task, int task_more, int stepSize)
{
    HIP_DYNAMIC_SHARED(double, vals);
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;
    double temp = 0.0, temp1 = 0.0;
    if(offset<task_more)
        num_task++; 
    for(int i = 0; i < num_task; ++ i){
    	int colIndex = offset + i * stepSize;
    	temp1 = d_a[colIndex];
    	temp += fabs(temp1);
    }
    vals[threadId] = temp;
    __syncthreads();
    if(threadId < 128) vals[threadId] += vals[threadId + 128];
    __syncthreads();
    if(threadId < 64) vals[threadId] += vals[threadId + 64];
    __syncthreads();
    if(threadId < 32) vals[threadId] += vals[threadId + 32];
    if(threadId < 16) vals[threadId] += vals[threadId + 16];
    if(threadId < 8) vals[threadId] += vals[threadId + 8];
    if(threadId < 4) vals[threadId] += vals[threadId + 4];
    if(threadId < 2) vals[threadId] += vals[threadId + 2];
    if(threadId < 1) vals[threadId] += vals[threadId + 1];
    if(threadId == 0) {
    	result[blockIdx.x] = vals[0];
    }
}

__global__ void kernel1HIP(const int len,const double omega,const double alpha,const double *res,const double *uk,double *pk){
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < len) {
        pk[icell] = res[icell] + omega*(pk[icell] - alpha*uk[icell]);
    }
}
__global__ void kernel2HIP(const int len,const double *res,const double gama,const double *uk,double *sk){
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < len) {
       	sk[icell] = res[icell] - gama*uk[icell];
    }
}
__global__ void kernel3HIP(const int len,const double gama,const double *pk,const double alpha,const double *sk, double *xk){
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < len) {
        xk[icell] = xk[icell] + gama * pk[icell] + alpha * sk[icell];
    }
}
HostVector* set_vector_gpu(){
    return new DeviceVector();
}
HostVector* set_vector_gpu(int n){
    return new DeviceVector(n);
}
#ifdef HAVE_MPI
HostVector* set_vector_gpu(int n,int nHalo){
    return new DeviceVector(n,nHalo);
}
#endif
