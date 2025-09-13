#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef TYPE
#define TYPE double
#endif

#define NUM_THREADS1 2
#define NUM_THREADS2 2

void col_part(const int rows, const int row_num,const int col_st,const int col_ed,const double *value,const int *index,const double *x,double *y,const int id){
     int col_num = col_ed - col_st;
     printf("col_part\n");
     for(int i = 0;i < row_num;i++){
         int offset = i * rows + id*col_num;
         register double t0=0,t1=0,t2=0,t3=0;
         int id0,id1,id2,id3;
         int j;
         for(int j = 0;j < col_num - 4;j+=4){
             if(index[offset + j + 3] == -1)break;
             id0 = index[offset + j];
             id1 = index[offset + j + 1];
             id2 = index[offset + j + 2];
             id3 = index[offset + j + 3];
             t0 += value[offset + j] * x[id0];
             t1 += value[offset + j + 1] * x[id1];
             t2 += value[offset + j + 2] * x[id2];
             t3 += value[offset + j + 3] * x[id3];
         }
         for(;j < col_num;j++){
             if(index[offset + j] == -1)break;
             t0 += value[offset+j] * x[index[offset+j]];
         }
         y[i] += (t0 + t1 + t2 + t3);
     }      
}
void row_part(const int rows,const int* col_next,const int r_st, const int r_ed,const double *value,const int *index,const double *x,double *y)
{
    int row_num = r_ed - r_st;
    double *y_ = (double*) malloc (sizeof(double) * row_num * NUM_THREADS2);
    #pragma omp parallel num_threads(NUM_THREADS2)
    { 
        
       int t_n = omp_get_thread_num();
       printf("row_part : threadid : %d\n",t_n);
       double *_y_ = y_ + t_n * row_num;
       const double *x_ = x + col_next[t_n];
       col_part(rows,row_num,col_next[t_n],col_next[t_n+1],value,index,x_,_y_,t_n);
    }
    #pragma omp parallel for num_threads(NUM_THREADS2)
    for(int i = 0;i < row_num;i++){
       for(int j = 0;j < NUM_THREADS2;j++){
          y[i] += y_[i + j * row_num];
       }
    } 
}
void count_rc(const int rows,const int width, int*row_next, int*col_next){
    int row_avr = rows / NUM_THREADS1;
    int row_last = rows % NUM_THREADS1;
    row_next[0] = 0;
    for(int j = 1;j <= NUM_THREADS1;j++){
        row_next[j] = row_next[j-1] + row_avr;
        if(j <= row_last)row_next[j]++;
    }
        
    int col_avr = width / NUM_THREADS2;
    int col_last = width % NUM_THREADS2;
    col_next[0] = 0;
    for(int j = 1;j <= NUM_THREADS2;j++){
        col_next[j] = col_next[j-1] + col_avr;
        if(j <= col_last)col_next[j]++;
    }
}
 
void spmv_rc_ell(const int rows,const int cols,const double* value,const int* index,const int width,\
                 const double* x,double* y,const int*row_next,const int*col_next)
{
    int k=omp_get_nested();
    omp_set_nested(1);//设置支持嵌套并行
    k=omp_get_nested();
        
#pragma omp parallel num_threads(NUM_THREADS1)
   {
       int t_n = omp_get_thread_num();
       printf("spmv_totol : threadid : %d\n",t_n);
       const double *val = value + (row_next[t_n] * width);
       const int *idx = index + (row_next[t_n] * width);
       row_part(rows,col_next,row_next[t_n],row_next[t_n+1],val,idx,x,y+(row_next[t_n]));
   }		   
}

void spmv(const int rows,const int cols,const double* value,const int* index,const int width,const double* x,double* y){
    int *row_next = (int *) malloc(sizeof(int) * (NUM_THREADS1+1));
    int *col_next = (int *) malloc(sizeof(int) * (NUM_THREADS2+1));
    count_rc(rows,width,row_next,col_next);
    spmv_rc_ell(rows,cols,value,index,width,x,y,row_next,col_next);
}
