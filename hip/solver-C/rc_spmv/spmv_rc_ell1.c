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
void spmv(const int rows,const int cols,const TYPE* value,const int* index,const int width,\
                 const TYPE* x,TYPE* y)
{
   int r,c,br,bc;
   int br_size = rows / NUM_THREADS_ROW,bc_size = cols / NUM_THREADS_COL;
   double *y_ = (double *) malloc (sizeof(double)*(rows * NUM_THREADS_COL));
   memset(y_,0,sizeof(double)*(rows*NUM_THREADS_COL));

#pragma omp parallel for schedule (static,1) collapse(2)
  for(bc = 0;bc < NUM_THREADS_COL;bc ++){
      for(br = 0;br < NUM_THREADS_ROW;br ++){
          for(r = 0;r < br_size;r++){
              int r_ = br * br_size + r;
              int c_1,c_2,c_3,c_4,idx1,idx2,idx3,idx4;
              double ty1=0,ty2=0,ty3=0,ty4=0;
              for(c = 0;c < bc_size;c+=4){
                  c_1 = bc * bc_size + c;
                  c_2 = bc * bc_size + c + 1;
                  c_3 = bc * bc_size + c + 2;
                  c_4 = bc * bc_size + c + 3;
                  idx1 = r_ * width + c_1;
                  idx2 = r_ * width + c_2;
                  idx3 = r_ * width + c_3;
                  idx4 = r_ * width + c_4;
                  ty1 += value[idx1] * x[index[idx1]];
                  ty2 += value[idx2] * x[index[idx2]];
                  ty3 += value[idx3] * x[index[idx3]];
                  ty4 += value[idx4] * x[index[idx4]];
              }
              y_[r_ + bc * rows] = (ty1+ty2+ty3+ty4);
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

