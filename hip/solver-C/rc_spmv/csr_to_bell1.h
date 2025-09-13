#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define TYPE double
//#define Max(x,y) ((x)>(y) ? (x) : (y))
//#define Min(x,y) ((x)<(y) ? (x) : (y))

#define WID 8 //cacheline_size
#define ROW 48 //thread_num
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
       	      //原始行换行标志
       	      value[j] = 0;
       	      index[j] = -2;
       	      j++,i++;
       	   }
       }
   }
}
/*
rows =  8, ell_rows = 4,ell_cols = 16
ei = 0: * * * -2 * * * * -2 *  *  * -1 -1 -1 -1
     1: * * *  * * * * * *  -2 *  *  *  *  * -1 
     3: * * *  * * * * * *  * -1 -1 -1 -1 -1 -1
     4: * * *  * * * * * *  * -2  *  *  * -1 -1 
*/

void spmv_bell(const int rows,const int cols,const int*ell_ptr,const TYPE* value,const int* index,\
               const int ell_rows,const int ell_cols,const TYPE* x,TYPE* y)
{
    int r,i,j;
    for(i = 0;i < rows;i++){y[i] = 0;}
#pragma omp parallel for schedule (static,1)
    for(r = 0;r < ell_rows;r++){
            int st = 0,ed = ell_ptr[r];
            if(r != 0)st = ell_ptr[r-1];
            for(i = st,j = (r) * ell_cols;i < ed,j < (r+1) * ell_cols;){
                    if(index[j] == -2){j++;i++;continue;}
                    if(index[j] == -1){break;}
                    y[i] += value[j]*x[index[j]];
                    j++;
            }
    }
}
/*
 //int idx1,idx2,idx3,idx4;
//double ty1=0,ty2=0,ty3=0,ty4=0;
                  idx1 = c;
                  idx2 = c + 1;
                  idx3 = c + 2;
                  idx4 = c + 3;
                  ty1 += value[idx1] * x[index[idx1]];
                  ty2 += value[idx2] * x[index[idx2]];
                  ty3 += value[idx3] * x[index[idx3]];
                  ty4 += value[idx4] * x[index[idx4]];
*/
