#include <stdlib.h>
#include <math.h>

#define TYPE double
#define Max(x,y) ((x)>(y) ? (x) : (y))
#define Min(x,y) ((x)<(y) ? (x) : (y))

#define MIN_WID 32 //8*4(列线程数）
#define MIN_ROW 24 //24（行线程数）
void exp_width(const int rows,const int cols,const int *ptr,int *width,int *minwid,int *exp_rows){
   int i = 0;
   *width = 0;
   *minwid = cols;
   for(i = 1;i <= rows;i++){
      int tmp = ptr[i] - ptr[i-1];
      *width = Max(*width,tmp);
      *minwid = Min(*minwid,tmp);
       //printf("width : %d tmp : %d\n",*width,tmp);
   }
   *width = ((*width + MIN_WID-1) / MIN_WID)*MIN_WID;
   *exp_rows = ((rows + MIN_ROW-1) / MIN_ROW)*MIN_ROW;
}

void csr_to_expell(const int rows,const int pad_rows,const int *ptr,const int *col,const TYPE *val,const int width,int *index,TYPE *value){
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
   for(i = rows+1;i <= pad_rows;i++){
       for(int j = (i-1)*width;j < i*width;j++){
           value[j] = 0;
           index[j] = -1;
       }
   }
}
//扩展ell格式使得能够做二维划分，行列都为32的倍数


