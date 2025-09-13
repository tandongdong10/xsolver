#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>

#ifndef TYPE
#define TYPE double
#endif

int lower_bound(const int *ptr,int start,int end,int val){
    while(start < end){
        int m = (start + end) >> 1;
        if(ptr[m] < val){
            start = m + 1;
        }else{
            end = m;
        }
    }
    return start;
}


void partition_nnz(const int *ptr, int row, int thread_num, int *partition){
    int nnz = ptr[row-1];
    int avr = nnz / thread_num;
    partition[0] = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for(int i = 1;i < thread_num;i++){
        partition[i] = lower_bound(ptr,0,row-1,(avr*i));
    }
    partition[thread_num] = row;
}

void spmv_unroll(const int rows,const int cols,const int *ptr,const TYPE *value,const int *col,const TYPE *x,TYPE *y,int lcs,int lce){
    for (int i = lcs; i < lce; i++){
	__asm__ volatile(
            "prfm pldl3strm, [%[value]]\n\t"
            "prfm pldl3strm, [%[col]]\n\t"
            :
            :[value] "r"(value+i), [col] "r"(col+i));
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
               	:
               	:[A_val] "r"(A_val+pi+8), [A_col] "r"(A_col+pi+8));
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

void spmv_omp(const int rows,const int cols,const int *ptr,const TYPE *value,const int *col,const TYPE *x,TYPE *y)
{
    int thread_num = 1;
    #ifdef _OPENMP
    #pragma omp parallel 
    #endif
{
    thread_num = omp_get_num_threads();
}
    int part[thread_num+1];
    partition_nnz(ptr+1,rows,thread_num,part);
    #ifdef _OPENMP
    #pragma omp parallel num_threads(thread_num)
    #endif
    {
        int tid = omp_get_thread_num();
        int local_st = part[tid];
        int local_ed = part[tid+1];
        spmv_unroll(rows,cols,ptr,value,col,x,y,local_st,local_ed);
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
