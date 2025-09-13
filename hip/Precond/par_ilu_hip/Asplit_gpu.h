#include "hip/hip_runtime.h"
#include "hipsparse.h"
#define T double
__global__ void split_csr2LU_L1d_Ud_gpu_pre_count(T*val,int*rows,int*cols,int*lrows,int*ucols,int n,int*uptr);

__global__ void split_csr2LU_L1d_Ud_gpu_split_lu(T*val,int*rows,int*cols,int n,\
T*lval,int*lrows,int*lcols,T*uval,int*ucols,int*urows,int*uptr);
__global__ void upre_count(T*uval,int*ucols,int*urows,int*ucols2,int nnz);
__global__ void Ucsr2Ucsc(T*uval,int*ucols,int*urows,T*uval2,int*ucols2,int*urows2,int n);

void parilu_pre(T*d_val,int*d_rows,int*d_cols,int n,int nnz,
T*&d_lval,int*&d_lrows,int*&d_lcols,int&gnnzl,
T*&d_uval,int*&d_ucols,int*&d_urows,int&gnnzu);
#undef T
