#ifdef _OPENMP
//#include <omp.h>
#include "/usr/lib/gcc/x86_64-redhat-linux/4.8.2/include/omp.h"
#endif
//#include "partition.h"
#include <stdio.h>

//#ifndef TYPE
//#define TYPE double
//#endif
template<typename TYPE>
void
bmAx_unroll2(const int rows,
		   const int cols,
		   const int *ptr,
		   const TYPE *value,
		   const int *col,
                   const TYPE* b,
                   const TYPE* x,
                   TYPE* y)
{
//#ifdef _OPENMP
//#pragma omp parallel for 
//#endif
    for (int i = 0; i < rows; i++)
    {
        register TYPE tmp0 = 0, tmp1 = 0, tmp2=0;
        int pks = ptr[i];
        int pke = ptr[i+1];
        int pkl = pke - pks;
        int pkl4 = pkl - 2;
        int col_ind0, col_ind1;
        const TYPE *A_val = &value[pks];
        const int *A_col = &col[pks];
        int pi;
        for (pi = 0; pi < pkl4; pi += 2)
        {
            col_ind0 = A_col[pi];
            col_ind1 = A_col[pi + 1];
   	    tmp0 += A_val[pi] * x[col_ind0];
            tmp1 += A_val[pi + 1] * x[col_ind1];
        }
        for (; pi < pkl; pi += 1)
        {
            tmp0 += A_val[pi] * x[A_col[pi]];
        }
        y[i] = b[i]-(tmp0 + tmp1);
    }
}
template<typename TYPE>
void bmax(const int rows,
		const int cols,
		const int *ptr,
		const TYPE *value,
	        const int *col,
                const TYPE *b,
                const TYPE *x,
                TYPE *y)
{
	bmAx_unroll2(rows, cols, ptr, value, col, b, x, y);
}
