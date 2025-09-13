#ifdef _OPENMP
#include <omp.h>
#endif
#include "partition.h"
#include <stdio.h>

#ifndef TYPE
#define TYPE double
#endif

void
spmv_unroll4(const int rows,
		   const int cols,
		   const int *ptr,
		   const TYPE *value,
		   const int *col,
                   const TYPE* x,
                   TYPE* y,
                   int lrs,
                   int lre)                                                         
{
    int m = rows;
    for (int i = lrs; i < lre; i++)
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

void
spmv_omp(const int rows,
		const int cols,
		const int *ptr,
		const TYPE *value,
		const int *col,
                const TYPE* x,
                TYPE* y)                                                         
{
    int m = rows;

    int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        num_threads = omp_get_num_threads();
    }
    int partition[num_threads + 1];
    const int *rows_end = ptr+1;
    balanced_partition_row_by_nnz(rows_end, m, num_threads, partition);
    //csr_row_partition(m,num_threads,partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        int tid = omp_get_thread_num();

        int local_m_s = partition[tid];
        int local_m_e = partition[tid + 1];
        spmv_unroll4(rows,cols,ptr,value,col,x,y,local_m_s,local_m_e);
    }
}

void spmv(const int rows,
		const int cols,
		const int *ptr,
		const TYPE *value,
	        const int *col,
                const TYPE *x,
                TYPE *y)
{
	spmv_omp(rows, cols, ptr, value, col, x, y);
}
