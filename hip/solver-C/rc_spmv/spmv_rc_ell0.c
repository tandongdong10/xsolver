#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef TYPE
#define TYPE double
#endif

#define NUM_THREADS_COL 2
#define NUM_THREADS_ROW 2

//ell格式按照4*8（numa节点个数*cacheline数据数）做padding；行列都是32的倍数 
void spmv(const int rows,const int cols,const double* value,const int* index,const int width,\
                 const double* x,double* y)
{
   int r,c,br,bc;
   int br_size = rows / NUM_THREADS_ROW,bc_size = cols / NUM_THREADS_COL;
   double *y_ = (double *) malloc (sizeof(double)*(rows * NUM_THREADS_COL));
   memset(y_,0,sizeof(double)*(rows*NUM_THREADS_COL));
#pragma omp parallel for schedule (static,1) collapse(2)
   for(bc = 0;bc < NUM_THREADS_COL;bc ++){
      for(br = 0;br < NUM_THREADS_ROW;br ++){
           int id = omp_get_thread_num();
           printf("threadid = %d ,bc = %d, br = %d\n",id,bc,br);
          for(r = 0;r < rows/NUM_THREADS_ROW;r++){
              for(c = 0;c < cols/NUM_THREADS_COL;c++){
                  int r_ = br * br_size + r;
                  int c_ = bc * bc_size + c;
                  int idx = r_ * width + c_;
                  y_[r_ + bc * rows] += value[idx] * x[index[idx]];
              }
          }
      }
  }

#pragma omp parallel for
  for(int i = 0;i < rows;i++){
      y[i] = 0;
      for(int j = 0;j < NUM_THREADS_COL;j++){
          y[i] += y_[i + j * rows];
      }
  }
}

