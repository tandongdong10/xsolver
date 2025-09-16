#include "DeviceMatrix.h"


rocblas_handle handle;
hipsparseHandle_t handle1;
hipStream_t stream[13];
int d_nthread = 256;
int d_nblock;

__global__ void getsendarrayHIP (const int d_nGhstCells, const int onebase, const double *d_a, const int* d_ptr, double *d_b)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nGhstCells) {
    	d_b[icell] = d_a[d_ptr[icell]-onebase];
    }
}
void setupgpu(HostMatrix *hostmtx,int precon){
    rocblas_create_handle(&handle);
    hipsparseCreate(&handle1);//parilu
    if(precon==3){
    	for(int i=0;i<13;i++){
            hipStreamCreate(&stream[i]);
    	}//parilut
   	parilt::cold_init();
    }
    
    d_nblock = (hostmtx->n+d_nthread-1)/d_nthread;
}
void freegpu(int precon){
    hipDeviceSynchronize();
    rocblas_destroy_handle(handle);
    hipsparseDestroy(handle1);
    if(precon==3){
    	for(int i=0; i<13; i++)
    	{
            hipStreamDestroy(stream[i]);
    	}
    }
}
