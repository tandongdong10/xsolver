#ifndef _GPUPARILUT_H_
#define _GPUPARILUT_H_
#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include "par_ilut_hip/My_computeSpmm.h"
#include <stdio.h>
#include <iostream>

extern hipsparseHandle_t handle1;
extern hipStream_t stream[13];
namespace parilt{
static inline __device__  void warp_reduce_min(int* minimum)
{
    #pragma unroll 
    for(int i = warpSize >> 1; i > 0; i >>= 1)
    {
        *minimum = min(*minimum, __shfl_xor(*minimum, i));
    }
}


static inline __device__  void warp_reduce_sum(int* minimum)
{
    #pragma unroll 
    for(int i = warpSize >> 1; i > 0; i >>= 1)
    {
        *minimum  += __shfl_xor(*minimum, i);
    }
}

void prefix_sum(int n,int*rows,int&nnz);
__global__ void temp_reverse(double*aval,int nnz);
__global__ void temp_tri_lu_nnz(double*val,int*row,int*col,int n,int nnz,\
double*lval,int*lrow,int*lcol,int nnzl,\
double*uval,int*urow,int*ucol,int nnzu);
__global__ void temp_tri_lu_nnz2(double*val,int*row,int*col,int n,int nnz,\
double*lval,int*lrow,int*lcol,int nnzl,\
double*uval,int*urow,int*ucol,int nnzu);
__global__ void temp_tri_lu(double*val,int*row,int*col,int n,int nnz,\
double*lval,int*lrow,int*lcol,int&nnzl,\
double*uval,int*urow,int*ucol,int&nnzu);
__global__ void temp_tri_lu_v2(double*val,int*row,int*col,int n,int nnz,\
double*lval,int*lrow,int*lcol,int nnzl,\
double*uval,int*urow,int*ucol,int nnzu);
__global__ void embedA_intoB_L(double*aval,int*acol,int*arow,int n,\
double*bval,int*bcol,int*brow,double*uval,int*ucol,int*urow);
__global__ void embedA_intoB_U(double*aval,int*acol,int*arow,int n,\
double*bval,int*bcol,int*brow,double*uval,int*ucol,int*urow);
__global__ void embedA_intoB_L2(double*aval,int*acol,int*arow,int n,\
double*bval,int*bcol,int*brow,double*uval,int*ucol,int*urow);
__global__ void embedA_intoB_U2(double*aval,int*acol,int*arow,int n,\
double*bval,int*bcol,int*brow);
__global__ void all_abs(double*val,int nnz);
__global__ void set_reference(int*d_rows,int n,int*d_row_reference);
__global__ void set_reference_v2(int*d_rows,int n,int nnz,int*d_row_reference);
__global__ void set_reference_v3(int*d_rows,int n,int*d_row_reference);
__global__ void thresold_remove_nnz(double*val,int*row,int*col,double thresold,int nnz,int n,\
int*out_row);
__global__ void thresold_remove_nnz2(double*val,int*row,int*col,double thresold,int nnz,int n,\
int*out_row);
__global__ void thresold_remove(double*val,int*row,int*col,double thresold,int nnz,int n,\
double*out_val,int*out_row,int*out_col);
__global__ void thresold_remove2(double*val,int*row,int*col,double thresold,int nnz,int n,\
double*out_val,int*out_row,int*out_col);
#define T  double

__global__ void L_sweep(T*lval,int*lrows,int*lcols,int n,int lnnz,int*lrow_refer,\
    T*uval,int*urows,int*ucols,\
    T*val,int*rows,int*cols);
__global__ void U_sweep(T*lval,int*lrows,int*lcols,\
    T*uval,int*urows,int*ucols,int n,int unnz,int*ucol_refer,\
    T*val,int*rows,int*cols);
__global__ void LU_sweep(T*lval,int*lrows,int*lcols,int lnnz,int*lrow_refer,\
    T*uval,int*urows,int*ucols,int unnz,int*ucol_refer,int n,\
    T*val,int*rows,int*cols);

template<typename nt>
void check(nt*tocheck,nt*orgin,int nnz,char*tprint);
void cold_init();
void parilut_clean(T*dval,int*dcol,int*drow,int n,int nnz,\
            T*&lval,int*&lcol,int*&lrow,int&nnzl,\
            T*&uval,int*&ucol,int*&urow,int&nnzu,\
	    double max_nnz_keep_rate,int sweep);

void call_parilut(T*hval,int*hcol,int*hrow,int n,int nnz,\
            T*&lval,int*&lcol,int*&lrow,int&nnzl,\
            T*&uval,int*&ucol,int*&urow,int&nnzu,\
            int sweep);
#undef T

void nytrsv_Lcsr(double*b,double*x,double*lval,int*lcol,int*lrow,int n);

void nytrsv_Ucsr(double*b,double*x,double*uval,int*ucol,int*urow,int n);
}
#endif
