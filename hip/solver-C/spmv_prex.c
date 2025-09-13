#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>

#ifndef TYPE
#define TYPE double
#endif


void
spmv_omp(const int rows,
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
	    __asm__ volatile(
                "prfm pldl3strm, [%[value]]\n\t"
                "prfm pldl3strm, [%[col]]\n\t"
                "prfm pldl3strm, [%[x]]\n\t"
                :
                :[value] "r"(value+i), [col] "r"(col+i),[x] "r"(x+col[i]));
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
            if(pi % 8 == 0 && pkl4 - pi > 8){
	    	__asm__ volatile(
                	"prfm pldl3strm, [%[A_val]]\n\t"
                	"prfm pldl3strm, [%[A_col]]\n\t"
                	"prfm pldl3strm, [%[x]]\n\t"
                	:
                	:[A_val] "r"(A_val+pi+8), [A_col] "r"(A_col+pi+8), [x] "r"(x[A_col[pi+8]]));
            }
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
