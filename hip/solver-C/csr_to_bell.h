#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define TYPE double
//#define Max(x,y) ((x)>(y) ? (x) : (y))
//#define Min(x,y) ((x)<(y) ? (x) : (y))

#define WID 8 //cacheline_size
#define ROW 12 //thread_num
int lower_bound(const int*ptr,int start,int end,int val_tmp,int *val){
    bool flag = true;
    if(start == end)flag = false;
    while(start < end){
        int m = (start + end) >> 1;
        if(ptr[m] + m < val_tmp){
            start = m + 1;
        }else{
            end = m;
        }
    }
    if(flag){
        int in_size = 0,over_size = 0;
        over_size = ptr[start] + start - val_tmp;
        if(start > 0) in_size = (ptr[start] - ptr[start-1] + 1) - over_size;
        if(over_size > in_size)start--;
        else *val += over_size; 
    } 
    return start;
}
//ell_ptr 大小为 ROW+1,留一行做缓冲，线程数开ROW个，最后一行又最后一个线程负责
void partition(const int *ptr, int rows,int ori_ell_cols,int *ell_ptr,int *ell_cols){
    int tmp = *ell_cols;
    for(int i = 0;i < ROW;i++){
         int val = ori_ell_cols * (i+1);
         ell_ptr[i] = lower_bound(ptr,0,rows,val,&tmp);
         *ell_cols = Max(*ell_cols,tmp); 
    }
    ell_ptr[ROW] = rows;
}
//计算ell_col实际需要的长度，ell_rows = ROW+1;
void count_bell(const int rows,const int nnz,const int *ptr,int *ell_cols,int *ell_rows,int *ell_ptr){ 
   int t_ell_cols = (nnz + rows + ROW-1) / ROW;//负载均衡
   t_ell_cols = ((t_ell_cols + WID - 1) / WID)*WID;//内存对齐
   *ell_cols = t_ell_cols;
   partition(ptr,rows,t_ell_cols,ell_ptr,ell_cols);
   *ell_cols =((*ell_cols + WID - 1)/WID) * WID;//调整后内存对齐
}
 
//将ptr的数值（一行有几个非零元存到value中）
void csr_to_bell(const int rows,const int nnz,const int *ptr,const int *col,const TYPE *val,\
                 const int ell_cols,const int ell_rows,int *ell_ptr,int *index,TYPE *value){
   int i = 0,ei = 0,j = 0;//ei为bell行号；
   for(;ei <= ROW;ei++){
       //确定每一行容纳的原始行始末位置
       int st = 0,ed = ell_ptr[ei];
       if(ei != 0)st = ell_ptr[ei-1];
       //j为bell行中的坐标，i为原始行的行号，用于定位原始行的值所在范围
       if(st == ed){
           for(j = (ei)*ell_cols;j < (ei+1)*ell_cols;j++){
               value[j] = 0;
               index[j] = -1;
           }
       }else{
       	   for(i = st,j = (ei) * ell_cols;i < ed,j < (ei+1) * ell_cols;){
       	       //原始行每一行起止
       	       value[j] = ptr[i+1] - ptr[i];
       	       index[j] = 0;
       	       j++;
       	      for(int k = ptr[i];k < ptr[i+1];k++){
       	          value[j] = val[k];
       	          index[j] = col[k];
       	          j++;
       	      }
       	      if(i == ed-1){
       	          for(;j < (ei+1)*ell_cols;j++){
       	              value[j] = 0;
       	              index[j] = -1;
       	          }
       	          break;
       	      }
       	      i++;
       	   }
       }
   }
}
/*
rows =  8, ell_rows = 4,ell_cols = 16
ei = 0: 3 * * * 4 * * * * 3 * * * -1 -1 -1
     1: 9 * * * * * * * * * 5 * * * * * -1
     3: 10 * * * * * * * * * * -1 -1 -1 -1
     4: 10 * * * * * * * * * * 3 * * * -1
*/
void spmv_bell4(const int rows,const int cols,const int*ell_ptr,const TYPE* value,const int* index,\
               const int ell_rows,const int ell_cols,const TYPE* x,TYPE* y)
{
    int r,i,j;
#pragma omp parallel for schedule (static,1)
    for(r = 0;r < ell_rows;r++){
        int st = 0,ed = ell_ptr[r];
        if(r != 0)st = ell_ptr[r-1];
        for(i = st,j = (r) * ell_cols;i < ed,j < (r+1) * ell_cols;){
            int rnnz = value[j++];
            //printf("rnnz = %d\n",rnnz);
            int k;
            double t0=0,t1=0,t2=0,t3=0;
            for(k = 0;k < rnnz-3;k+=4){
                t0 += value[j]*x[index[j]];
                t1 += value[j+1]*x[index[j+1]];
                t2 += value[j+2]*x[index[j+2]];
                t3 += value[j+3]*x[index[j+3]];
                j+=4;
            }
            for(;k<rnnz;k++){
                t0 += value[j]*x[index[j]];
                j++;
            }
            y[i] = (t0+t1+t2+t3);
            if(index[j] == -1){break;}
            i++;
        }
    }
}
void spmv_bell2(const int rows,const int cols,const int*ell_ptr,const TYPE* value,const int* index,\
               const int ell_rows,const int ell_cols,const TYPE* x,TYPE* y)
{
    int r,i,j;
#pragma omp parallel for schedule (static,1)
    for(r = 0;r < ell_rows;r++){
        int st = 0,ed = ell_ptr[r];
        if(r != 0)st = ell_ptr[r-1];
        for(i = st,j = (r) * ell_cols;i < ed,j < (r+1) * ell_cols;){
            int rnnz = value[j++];
            int k;
            double t0=0,t1=0;
            for(k = 0;k < rnnz-1;k+=2){
                t0 += value[j]*x[index[j]];
                t1 += value[j+1]*x[index[j+1]];
                j+=2;
            }
            for(;k<rnnz;k++){
                t0 += value[j]*x[index[j]];
                j++;
            }
            y[i] = (t0+t1);
            if(index[j] == -1){break;}
            i++;
        }
    }
}

void spmv_bell1(const int rows,const int cols,const int*ell_ptr,const TYPE* value,const int* index,\
               const int ell_rows,const int ell_cols,const TYPE* x,TYPE* y)
{
    int r,i,j;
#pragma omp parallel for schedule (static,1)
    for(r = 0;r < ell_rows;r++){
        int st = 0,ed = ell_ptr[r];
        if(r != 0)st = ell_ptr[r-1];
        for(i = st,j = (r) * ell_cols;i < ed,j < (r+1) * ell_cols;){
            int rnnz = value[j++];
            //printf("rnnz = %d\n",rnnz);
            int k;
            double t0=0,t1=0;
            for(k = 0;k < rnnz;k++){
                t0 += value[j]*x[index[j]];
                j++;
            }
            y[i] = t0;
            if(index[j] == -1){break;}
            i++;
        }
    }
}
