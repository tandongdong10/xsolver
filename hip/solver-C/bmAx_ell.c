#ifdef _OPENMP
//#include <omp.h>
#include "/usr/lib/gcc/x86_64-redhat-linux/4.8.2/include/omp.h"
#endif
#include <stdio.h>


template<typename TYPE>
void bmax_ell_unroll4(const int rows,const int cols,const TYPE* value,const int* index,const int width,const TYPE* rhs, const TYPE* x,TYPE* y)                {
#ifdef _OPENMP
#pragma omp parallel for 
#endif  
    for(int i=0;i<rows;i++){
        register TYPE tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;
        int pks = i * width;
        int pke = (i+1) * width;
        int pkl4 = width - 4;
        int col_ind0, col_ind1, col_ind2, col_ind3;
       	const TYPE *A_val = &value[pks];
       	const int *A_col = &index[pks];
       	int pi;
       	for(pi = 0; pi < pkl4; pi += 4)
       	{
            if(A_col[pi+3] == -1)break;
            col_ind0 = A_col[pi];
            col_ind1 = A_col[pi + 1];
       	    col_ind2 = A_col[pi + 2];
       	    col_ind3 = A_col[pi + 3];
       	    tmp0 += A_val[pi] * x[col_ind0];
       	    tmp1 += A_val[pi + 1] * x[col_ind1];
       	    tmp2 += A_val[pi + 2] * x[col_ind2];
       	    tmp3 += A_val[pi + 3] * x[col_ind3];
        }
       	   for (; pi < width; pi += 1)
       	   {
            	if(A_col[pi] == -1)break;
       	       tmp0 += A_val[pi] * x[A_col[pi]];
       	   }
       	   y[i] = rhs[i]- (tmp0 + tmp1 + tmp2 + tmp3);
	}		   
}
template<typename TYPE>
void bmax_ell_unroll2(const int rows,const int cols,const TYPE* value,const int* index,const int width,const TYPE *rhs, const TYPE* x,TYPE* y)                {
#ifdef _OPENMP
#pragma omp parallel for 
#endif  
    for(int i=0;i<rows;i++){
        register TYPE tmp0 = 0, tmp1 = 0;
        int pks = i * width;
        int pke = (i+1) * width;
        int pkl4 = width - 2;
        int col_ind0, col_ind1;
       	const TYPE *A_val = &value[pks];
       	const int *A_col = &index[pks];
       	int pi;
       	for(pi = 0; pi < pkl4; pi += 2)
       	{
            if(A_col[pi+1] == -1)break;
            col_ind0 = A_col[pi];
            col_ind1 = A_col[pi + 1];
       	    tmp0 += A_val[pi] * x[col_ind0];
       	    tmp1 += A_val[pi + 1] * x[col_ind1];
        }
       	   for (; pi < width; pi += 1)
       	   {
            	if(A_col[pi] == -1)break;
       	       tmp0 += A_val[pi] * x[A_col[pi]];
       	   }
       	   y[i] = rhs[i]-(tmp0 + tmp1);
	}		   
}

template<typename TYPE>
void bmax_ell_0(const int rows,const int cols,const TYPE* value,const int* index,const int width,const TYPE*rhs, const TYPE* x,TYPE* y)
{
#ifdef _OPENMP
#pragma omp parallel for 
#endif  
    for(int i=0;i<rows;i++){
        register TYPE tmp0 = 0;
        int pks = i * width;
        int pke = (i+1) * width;
       	const TYPE *A_val = &value[pks];
       	const int *A_col = &index[pks];
       	int pi;
       	for (pi=0; pi < width; pi += 1)
        {
       	    tmp0 += A_val[pi] * x[A_col[pi]];
       	}
        y[i] = rhs[i]-tmp0;
    }
}		   
template<typename TYPE>
void bmax_ell(const int rows,const int cols,const TYPE* value,const int* index,const int width,const TYPE*rhs, const TYPE* x,TYPE* y){
	bmax_ell_unroll2(rows,cols,value,index,width,rhs,x,y);
}
