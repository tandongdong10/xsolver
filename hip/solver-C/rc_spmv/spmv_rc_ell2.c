#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<string.h>
#ifndef TYPE
#define TYPE double
#endif

#define NUM_THREADS_COL 1
#define NUM_THREADS_ROW 24

//ell格式按照4*8（numa节点个数*cacheline数据数）做padding；行列都是32的倍数 
void spmv(const int rows,const int cols,const TYPE* value,const int* index,const int width,\
                 const TYPE* x,TYPE* y)
{
   int r,c,br,bc;
   int br_size = rows / NUM_THREADS_ROW,bc_size = cols / NUM_THREADS_COL;
   double *y_ = (double *) malloc (sizeof(double)*(rows * NUM_THREADS_COL));
   memset(y_,0,sizeof(double) * (rows*NUM_THREADS_COL));

//#pragma omp parallel for schedule (static,1) collapse(2)
  for(bc = 0;bc < NUM_THREADS_COL;bc ++){
      for(br = 0;br < NUM_THREADS_ROW;br ++){
          int y_offset = bc * rows;
          int c_offset = bc * bc_size;
          for(r = 0;r < br_size;r++){
              int r_ = br * br_size + r;
              int idx_offset = r_ * width;
              int c_1,c_2,c_3,c_4,idx1,idx2,idx3,idx4;
              register double ty1=0,ty2=0,ty3=0,ty4=0;
              for(c = 0;c < bc_size;c+=4){
                  c_1 = c_offset + c;
                  c_2 = c_offset + c + 1;
                  c_3 = c_offset + c + 2;
                  c_4 = c_offset + c + 3;
                  idx1 = idx_offset + c_1;
                  idx2 = idx_offset + c_2;
                  idx3 = idx_offset + c_3;
                  idx4 = idx_offset + c_4;
                  if(index[idx4] == -1)break;
                  ty1 += value[idx1] * x[index[idx1]];
                  ty2 += value[idx2] * x[index[idx2]];
                  ty3 += value[idx3] * x[index[idx3]];
                  ty4 += value[idx4] * x[index[idx4]];
              }
              for(;c < bc_size;c++){
                  int c_ = c_offset + c;
                  int idx = idx_offset + c_;
                  if(index[idx] == -1)break;
                  ty1 += value[idx] * x[index[idx]];
              }
              y_[r_ + y_offset] = (ty1+ty2+ty3+ty4);
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

