#ifdef _OPENMP
#include <omp.h>
#endif
//#include "partition.h"
#include <stdio.h>

//#ifndef TYPE
//#define TYPE double
//#endif
template<typename TYPE>
void
spmv_unroll2(const int rows,
		   const int cols,
		   const int *ptr,
		   const TYPE *value,
		   const int *col,
                   const TYPE* x,
                   TYPE* y)
{
    int onebase=ptr[0];
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for (int i = 0; i < rows; i++)
    {
        register TYPE tmp0 = 0, tmp1 = 0;
        int pks = ptr[i]-onebase;
        int pke = ptr[i+1]-onebase;
        int pkl = pke - pks;
        int pkl4 = pkl - 2;
        int col_ind0, col_ind1;
        const TYPE *A_val = &value[pks];
        const int *A_col = &col[pks];
        int pi;
        for (pi = 0; pi < pkl4; pi += 2)
        {
            col_ind0 = A_col[pi]-onebase;
            col_ind1 = A_col[pi + 1]-onebase;
   	    tmp0 += A_val[pi] * x[col_ind0];
            tmp1 += A_val[pi + 1] * x[col_ind1];
        }
        for (; pi < pkl; pi += 1)
        {
            tmp0 += A_val[pi] * x[A_col[pi]-onebase];
        }
        y[i] = (tmp0 + tmp1);
    }
}
template<typename TYPE>
void
spmv_unroll4(const int rows,
		   const int cols,
		   const int *ptr,
		   const TYPE *value,
		   const int *col,
                   const TYPE* x,
                   TYPE* y)
{
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for (int i = 0; i < rows; i++)
    {
        register TYPE tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;
        int pks = ptr[i];
        int pke = ptr[i+1];
        int pkl = pke - pks;
        int pkl4 = pkl - 4;
        int col_ind0, col_ind1, col_ind2, col_ind3;
        const TYPE *A_val = &value[pks];
        const int *A_col = &col[pks];
        int pi;
        for (pi = 0; pi < pkl4; pi += 4)
        {
            col_ind0 = A_col[pi];
            col_ind1 = A_col[pi + 1];
           col_ind2 = A_col[pi + 2];
           col_ind3 = A_col[pi + 3];
   	    tmp0 += A_val[pi] * x[col_ind0];
            tmp1 += A_val[pi + 1] * x[col_ind1];
            tmp2 += A_val[pi + 2] * x[col_ind2];
            tmp3 += A_val[pi + 3] * x[col_ind3];
        }
        for (; pi < pkl; pi += 1)
        {
            tmp0 += A_val[pi] * x[A_col[pi]];
        }
        y[i] = (tmp0 + tmp1 + tmp2 + tmp3);
    }
}
template<typename TYPE>
void
spmv_unroll6(const int rows,
		   const int cols,
		   const int *ptr,
		   const TYPE *value,
		   const int *col,
                   const TYPE* x,
                   TYPE* y)
{
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for (int i = 0; i < rows; i++)
    {
        register TYPE tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0,tmp4 = 0,tmp5 = 0;
        int pks = ptr[i];
        int pke = ptr[i+1];
        int pkl = pke - pks;
        int pkl4 = pkl - 6;
        int col_ind0, col_ind1, col_ind2, col_ind3, col_ind4, col_ind5;
        const TYPE *A_val = &value[pks];
        const int *A_col = &col[pks];
        int pi;
        for (pi = 0; pi < pkl4; pi += 6)
        {
            col_ind0 = A_col[pi];
            col_ind1 = A_col[pi + 1];
           col_ind2 = A_col[pi + 2];
           col_ind3 = A_col[pi + 3];
           col_ind4 = A_col[pi + 4];
           col_ind5 = A_col[pi + 5];
   	    tmp0 += A_val[pi] * x[col_ind0];
            tmp1 += A_val[pi + 1] * x[col_ind1];
            tmp2 += A_val[pi + 2] * x[col_ind2];
            tmp3 += A_val[pi + 3] * x[col_ind3];
            tmp4 += A_val[pi + 4] * x[col_ind4];
            tmp5 += A_val[pi + 5] * x[col_ind5];
        }
        for (; pi < pkl; pi += 1)
        {
            tmp0 += A_val[pi] * x[A_col[pi]];
        }
        y[i] = (tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5);
    }
}

template<typename TYPE>
void
spmv_unroll8(const int rows,
		   const int cols,
		   const int *ptr,
		   const TYPE *value,
		   const int *col,
                   const TYPE* x,
                   TYPE* y)
{
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for (int i = 0; i < rows; i++)
    {
        register TYPE tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0,tmp4 = 0,tmp5 = 0,tmp6 = 0,tmp7 = 0;
        int pks = ptr[i];
        int pke = ptr[i+1];
        int pkl = pke - pks;
        int pkl4 = pkl - 8;
        int col_ind0, col_ind1, col_ind2, col_ind3, col_ind4, col_ind5, col_ind6, col_ind7;
        const TYPE *A_val = &value[pks];
        const int *A_col = &col[pks];
        int pi;
        for (pi = 0; pi < pkl4; pi += 8)
        {
            col_ind0 = A_col[pi];
            col_ind1 = A_col[pi + 1];
           col_ind2 = A_col[pi + 2];
           col_ind3 = A_col[pi + 3];
           col_ind4 = A_col[pi + 4];
           col_ind5 = A_col[pi + 5];
           col_ind6 = A_col[pi + 6];
           col_ind7 = A_col[pi + 7];
   	    tmp0 += A_val[pi] * x[col_ind0];
            tmp1 += A_val[pi + 1] * x[col_ind1];
            tmp2 += A_val[pi + 2] * x[col_ind2];
            tmp3 += A_val[pi + 3] * x[col_ind3];
            tmp4 += A_val[pi + 4] * x[col_ind4];
            tmp5 += A_val[pi + 5] * x[col_ind5];
            tmp6 += A_val[pi + 6] * x[col_ind6];
            tmp7 += A_val[pi + 7] * x[col_ind7];
        }
        for (; pi < pkl; pi += 1)
        {
            tmp0 += A_val[pi] * x[A_col[pi]];
        }
        y[i] = (tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7);
    }
}

template<typename TYPE>
void
spmv_unroll1(const int rows,
		const int cols,
		const int *ptr,
		const TYPE *value,
		const int *col,
                const TYPE* x,
                TYPE* y)                                                         
{
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    
	for(int i=0;i<rows;i++){
        register TYPE tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;
        int pks = ptr[i];
        int pke = ptr[i+1];
        int pkl = pke - pks;
        int col_ind0;
        const TYPE *A_val = &value[pks];
        const int *A_col = &col[pks];
        int pi;
        y[i] = 0.0;
        for (pi = 0; pi < pkl; pi ++)
        {
            col_ind0 = A_col[pi];
   	    y[i] += A_val[pi] * x[col_ind0];
        }
        }
    
}

template<typename TYPE>
void spmv(const int rows,
		const int cols,
		const int *ptr,
		const TYPE *value,
	        const int *col,
                const TYPE *x,
                TYPE *y)
{
	spmv_unroll2(rows, cols, ptr, value, col, x, y);
}
