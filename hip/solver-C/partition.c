#include <math.h>
#include <limits.h>
#include <stdint.h>

#ifndef TYPE
#define TYPE double
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

int lower_bound_int(const int *t, int l, int r, int value)
{
    while (r > l)
    {
        int m = (l + r) / 2;
        if (t[m] < value)
        {
            l = m + 1;
        }
        else
        {
            r = m;
        }
    }
    return l;
}


void balanced_partition_row_by_nnz(const int *acc_sum_arr, int rows, int num_threads, int *partition)
{
    int nnz = acc_sum_arr[rows - 1];
    int ave = nnz / num_threads;
    partition[0] = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int i = 1; i < num_threads; i++)
    {
        partition[i] = lower_bound_int(acc_sum_arr, 0, rows - 1, (ave * i));
    }
    partition[num_threads] = rows;
}

void csr_row_partition( int rows, int num_threads, int *partition){
     partition[0] = 0;
     int ave = rows/num_threads;
     int last = rows % num_threads;
     for(int i = 1;i < num_threads;i++){
         if(i <= last){
             partition[i] = partition[i-1] + ave + 1;
         }else{
             partition[i] = partition[i-1] + ave;
         }
     }
      partition[num_threads] = rows;
} 
    
