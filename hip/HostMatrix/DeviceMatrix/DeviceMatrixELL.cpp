#include "DeviceMatrixELL.h"

__global__ void matMultELL( const int nrow, const int nz, const int onebase, const double *val, const int *idx, double *d_x, double *d_y)
{
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(rowIdx < nrow)
  {
    double dot = 0;
    for(int i=0; i<nz; i++)
    {
      int pos = nrow * i + rowIdx;
      double tmp = val[pos];
      int col = idx[pos]-onebase;
      dot += tmp * d_x[col];
    }
    d_y[rowIdx] = dot;
  }
}
__global__ void bmAxELL( const int nrow, const int nz, const int onebase, const double *val,  const int *idx, double *d_q, double *d_x, double *d_y)
{
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(rowIdx < nrow)
  {
    double dot = d_q[rowIdx];
    for(int i=0; i<nz; i++)
    {
      int pos = nrow * i + rowIdx;
      double tmp = val[pos];
      int col = idx[pos]-onebase;
      dot -= tmp * d_x[col];
    }
    d_y[rowIdx] = dot;
  }
}

HostMatrix* set_matrix_gpu_ell(){
    return new DeviceMatrixELL();
}

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
void DeviceMatrixELL::Update(HostMatrix *hstmtx){
    InsertZero1(hstmtx);
}

void DeviceMatrixELL::InsertZero0(HostMatrix *mtxcsr)
{	
  int nInterior=mtxcsr->m;
  int valSizes = nInterior * num_nz;
  int *idx = new int[valSizes];
  for(int i=0;i<valSizes;i++){
    idx[i]=onebase;
  }
  int *ptr=new int[nInterior+1];//(int *)malloc((nIntCells+1)*sizeof(int));//new int[nIntCells+1];
  ptr[0]=onebase;
  int nz_cnt = 0;
  int *hstmtx_ptr=mtxcsr->getptr();
  int *hstmtx_idx=mtxcsr->getidx();
  int *d_mtx_ptr,*d_mtx_idx;
  //printf("NbCell_ptr_c[0]=%d\n",NbCell_ptr_c[0]);
  //printf("NbCell_s[0]=%d\n",NbCell_s[0]);
  for(int i=0; i<nInterior; i++)
  {
    nz_cnt = hstmtx_ptr[i+1] - hstmtx_ptr[i];
    if(nz_cnt>num_nz){
	printf("Error : The number of non-zero elements in each row is wrong!!!\n");
	printf("Error : The number is larger than what you inputed!\n");
	exit(0);
    }
    ptr[i+1]=num_nz+ptr[i];
    memcpy(&idx[i*num_nz], &hstmtx_idx[hstmtx_ptr[i]-onebase], nz_cnt * sizeof(int));
  }
    hipMalloc((void **)&d_mtx_ptr, sizeof(int) * (nInterior + 1));
    hipMalloc((void **)&d_mtx_idx, sizeof(int) * valSizes);
  //for(int i=0;i<2*32;i++){
  //  printf("idx[%d]=%d,val[%d]=%f\n",i,idx[i],i,val[i]);
  //}
    hipMemcpy(d_mtx_ptr, ptr, sizeof(int) * (nInterior + 1), hipMemcpyHostToDevice);
    hipMemcpy(d_mtx_idx, idx, sizeof(int) * valSizes, hipMemcpyHostToDevice);
    dim3 threads2(num_nz, num_nz);
    dim3 block2( (nInterior+num_nz-1)/num_nz );
    if(num_nz==16) 
  	hipLaunchKernelGGL(TransposeMatrix_kernel, block2, threads2, 0, 0, d_mtx_idx, ell_idx, nInterior);
  else
  	hipLaunchKernelGGL(TransposeMatrix_kernel_32, block2, threads2, 0, 0, d_mtx_idx, ell_idx, nInterior);
  hipDeviceSynchronize();
  delete []idx;
  delete []ptr;
  hipFree(d_mtx_ptr);
  hipFree(d_mtx_idx);
}
void DeviceMatrixELL::InsertZero1(HostMatrix *mtxcsr)
{
  int nInterior=mtxcsr->m;
  int valSizes = nInterior * num_nz;
  double *val = new double[valSizes];
  memset(val,0,valSizes*sizeof(double));
  int nz_cnt = 0;
  int *hstmtx_ptr=mtxcsr->getptr();
  double *hstmtx_val=mtxcsr->getval();
  double *d_mtx_val;
  for(int i=0; i<nInterior; i++)
  {
    nz_cnt = hstmtx_ptr[i+1] - hstmtx_ptr[i];
    memcpy(&val[i*num_nz], &hstmtx_val[hstmtx_ptr[i]-onebase], nz_cnt * sizeof(double));
  }
  hipMalloc((void **)&d_mtx_val, sizeof(double) * valSizes);
  hipMemcpy(d_mtx_val, val, sizeof(double) * valSizes, hipMemcpyHostToDevice);

  hipMemset(ell_val,0,sizeof(double) * valSizes);
  dim3 threads2(num_nz, num_nz);
  dim3 block2( (nInterior+num_nz-1)/num_nz ); 
  if(num_nz==16) 
  	hipLaunchKernelGGL(TransposeMatrix_kernel, block2, threads2, 0, 0,d_mtx_val, ell_val, nInterior);
  else
  	hipLaunchKernelGGL(TransposeMatrix_kernel_32, block2, threads2, 0, 0,d_mtx_val, ell_val, nInterior);
  hipDeviceSynchronize();
  delete []val;
  hipFree(d_mtx_val);
}

void DeviceMatrixELL::SpMV(HostVector *x,HostVector *y)
{
    double *d_x=x->val;
    double *d_y=y->val;
    int d_nblock = (m+d_nthread-1)/d_nthread;
#ifdef HAVE_MPI
    hipLaunchKernelGGL(getsendarrayHIP, dim3(d_nblock),dim3(d_nthread), 0, 0, nHalo, onebase, d_x, exchange_ptr, d_x+n);
    hipMemcpy(x_nHalo+n, d_x+n, sizeof(double)*(nHalo), hipMemcpyDeviceToHost);
    for(int i=0;i<nHalo;i++)
	x_nHalo[topo_c.exchange_ptr[i]]=x_nHalo[n+i];
    communicator_p2p(x_nHalo);
    communicator_p2p_waitall();
    hipMemcpy(d_x+n, x_nHalo+n, sizeof(double)*nHalo, hipMemcpyHostToDevice);
#endif
    hipLaunchKernelGGL(matMultELL, dim3(d_nblock), dim3(d_nthread), 0, 0, m, num_nz, onebase, ell_val, ell_idx, d_x, d_y);
}
void DeviceMatrixELL::bmAx(HostVector *q, HostVector *x, HostVector *y)
{
    double *d_x=x->val;
    double *d_y=y->val;
    double *d_q=q->val;
    int d_nblock = (m+d_nthread-1)/d_nthread;
#ifdef HAVE_MPI
    hipLaunchKernelGGL(getsendarrayHIP, dim3(d_nblock),dim3(d_nthread), 0, 0, nHalo, onebase,d_x, exchange_ptr, d_x+n);
    hipMemcpy(x_nHalo+n, d_x+n, sizeof(double)*(nHalo), hipMemcpyDeviceToHost);
    for(int i=0;i<nHalo;i++)
	x_nHalo[topo_c.exchange_ptr[i]]=x_nHalo[n+i];
    communicator_p2p(x_nHalo);
    communicator_p2p_waitall();
    hipMemcpy(d_x+n, x_nHalo+n, sizeof(double)*nHalo, hipMemcpyHostToDevice);
#endif
    hipLaunchKernelGGL(bmAxELL, dim3(d_nblock),dim3(d_nthread), 0, 0, m, num_nz, onebase, ell_val, ell_idx, d_q, d_x, d_y);
}
