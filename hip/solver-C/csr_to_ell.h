#include <stdlib.h>
#include <math.h>

#define TYPE double
void count_width(const int rows,const int cols,const int *ptr,int *width,int *minwid){
   int i = 0;
   *width = 0;
   *minwid = cols;
   for(i = 1;i <= rows;i++){
      int tmp = ptr[i] - ptr[i-1];
      *width = Max(*width,tmp);
      *minwid = Min(*minwid,tmp);
       //printf("width : %d tmp : %d\n",*width,tmp);
   }
}
//行优先存储
void csr_to_ell(const int rows,const int cols,const int *ptr,const int *col,const TYPE *val,const int width,int *index,TYPE *value){
   int i = 0;
   for(i = 1;i <= rows;i++){
       int st = ptr[i-1],ed = ptr[i];
       for(int j = (i-1) * width;j < i * width;j++){
           if (st < ed){
	      value[j] = val[st];
              index[j] = col[st];
           }
           else{
              value[j] = 0;
              index[j] = -1;
           }
           st++;
       }
   }
}

void csr_to_ell_0(const int rows,const int cols,const int *ptr,const int *col,const TYPE *val,const int width,int *index,TYPE *value){
   int i = 0;
   for(i = 1;i <= rows;i++){
       int st = ptr[i-1],ed = ptr[i];
       for(int j = (i-1) * width;j < i * width;j++){
           if (st < ed){
	      value[j] = val[st];
              index[j] = col[st];
           }
           else{
              value[j] = 0;
              index[j] = 0;
           }
           st++;
       }
   }
}

