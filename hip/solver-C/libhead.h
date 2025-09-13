#ifndef LIBHEAD_H
#define LIBHEAD_H

//#ifndef TYPE
//#define TYPE double
//#endif
#include "axpy.c"
#include "scal.c"
#include "copy.c"
#include "dot.c"
#include "spmv_unroll.c"
#include "bmAx_unroll.c"
#include "spmv_ell.c"
#include "bmAx_ell.c"
#include "bicg_kernel.c"

#define Max(x,y) ((x)>(y) ? (x) : (y))
#define Min(x,y) ((x)<(y) ? (x) : (y))
template<typename TYPE>
void copy(const int N, const TYPE* X, TYPE* Y);
template<typename TYPE>
TYPE dot(const int N, const TYPE* X, const TYPE* Y);
template<typename TYPE>
void axpy(const int nz, const TYPE a, const TYPE* x, TYPE* y);
template<typename TYPE>
void scal(const int nz, const TYPE a, const TYPE* x );
template<typename TYPE>
void spmv(const int rows,const int cols,const int *ptr,const TYPE *value,const int *col,const TYPE *x,TYPE *y);
template<typename TYPE>
void bmax(const int rows,const int cols,const int *ptr,const TYPE *value,const int *col,const TYPE *rhs,const TYPE *x,TYPE *y);
template<typename TYPE>
void spmv_ell(const int rows,const int cols,const TYPE* value,const int* index,const int width,const TYPE* x,TYPE* y);
template<typename TYPE>
void bmax_ell(const int rows,const int cols,const TYPE* value,const int* index,const int width,const TYPE*rhs, const TYPE* x,TYPE* y);
template<typename TYPE>
void kernel_1(const int len,const TYPE omega,const TYPE alpha,const TYPE *res,TYPE *uk,TYPE *pk);
template<typename TYPE>
void kernel_2(const int len,const TYPE *res,const TYPE gama,const TYPE *uk,TYPE *sk);
template<typename TYPE>
void kernel_3(const int len,const TYPE gama,const TYPE *pk,const TYPE alpha,const TYPE *sk, TYPE *xk);
#endif

