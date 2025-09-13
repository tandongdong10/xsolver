#include <stdlib.h>
#include <math.h>
//#include <immintrin.h>

#define TYPE double
#define PlusDot4(sum,r,xv,xi,nnz) { \
    if (nnz > 0) { \
      int nnz2=nnz,rem=nnz&0x3; \
      switch (rem) { \
      case 3: sum += *xv++ *r[*xi++]; \
      case 2: sum += *xv++ *r[*xi++]; \
      case 1: sum += *xv++ *r[*xi++]; \
        nnz2      -= rem;} \
      while (nnz2 > 0) { \
        sum +=  xv[0] * r[xi[0]] + xv[1] * r[xi[1]] + \
                xv[2] * r[xi[2]] + xv[3] * r[xi[3]]; \
        xv += 4; xi += 4; nnz2 -= 4; \
      } \
      xv -= nnz; xi -= nnz; \
    } \
  }
#define PlusDot2(sum,r,xv,xi,nnz) { \
    int __i,__i1,__i2; \
    for (__i=0; __i<nnz-1; __i+=2) {__i1 = xi[__i]; __i2=xi[__i+1]; \
                                    sum += (xv[__i]*r[__i1] + xv[__i+1]*r[__i2]);} \
    if (nnz & 0x1) sum += xv[__i] * r[xi[__i]];}

void Dot(double *sum,const double *x,const double *val,const int *index,int n) {
    double tmp0=0.,tmp1=0.;
    int i;
    for(i = 0;i < n-1;i+=2){
        tmp0 += val[i] * x[index[i]];
        tmp1 += val[i+1] * x[index[i+1]];
    }
    if(i == n-1){
        tmp0 += val[i] * x[index[i]];
    }
    *sum += (tmp0+tmp1);
}
//void Dot_AVX512(double *sum,const double *x,const double *val,const int *index,int n)
//{
//    __m512d  vec_x,vec_y,vec_vals;
//    __m256i  vec_idx;
//    __mmask8 mask;
//    int j;
//
//    vec_y = _mm512_setzero_pd();
//    for (j=0; j<(n>>3); j++) {
//        vec_idx  = _mm256_loadu_si256((__m256i const*)index);
//        vec_vals = _mm512_loadu_pd(val);
//        vec_x    = _mm512_i32gather_pd(vec_idx,x,8);
//        vec_y    = _mm512_fmadd_pd(vec_x,vec_vals,vec_y);
//        index += 8; val += 8;
//    }
//    /* masked load does not work on KNL, it requires avx512vl */
//    if ((n&0x07)>2) {
//        mask     = (__mmask8)(0xff >> (8-(n&0x07)));
//        vec_idx  = _mm256_loadu_si256((__m256i const*)index);
//        vec_vals = _mm512_loadu_pd(val);
//        vec_x    = _mm512_mask_i32gather_pd(vec_x,mask,vec_idx,x,8);
//        vec_y    = _mm512_mask3_fmadd_pd(vec_x,vec_vals,vec_y,mask);
//    } else if ((n&0x07)==2) {
//        *sum += val[0]*x[index[0]];
//        *sum += val[1]*x[index[1]];
//    } else if ((n&0x07)==1) {
//        *sum += val[0]*x[index[0]];
//    }
//    if (n>2) *sum += _mm512_reduce_add_pd(vec_y);
///*
//  for (j=0;j<(n&0x07);j++) *sum += val[j]*x[index[j]];
//*/
//}
void spmv_hong4(const int rows,const int cols,const int *ptr,const TYPE *value,const int *index,const TYPE* x,TYPE* y)
{
    const double *val_ ;val_ = value;
    const int *idx_ ; idx_ = index;
    int n = 0;
    double sum;
    for(int i = 0;i < rows;i++){
        val_ += n;
        idx_ += n;
        n = ptr[i+1] - ptr[i];
        sum = 0.;
        PlusDot4(sum,x,val_,idx_,n);
        y[i] = sum;
    }
}
void spmv_hong2(const int rows,const int cols,const int *ptr,const TYPE *value,const int *index,const TYPE* x,TYPE* y)
{
    const double *val_ ;val_ = value;
    const int *idx_ ; idx_ = index;
    int n = 0;
    double sum;
    for(int i = 0;i < rows;i++){
        val_ += n;
        idx_ += n;
        n = ptr[i+1] - ptr[i];
        sum = 0.;
        PlusDot2(sum,x,val_,idx_,n);
        y[i] = sum;
    }
}
void spmv_noavx(const int rows,const int cols,const int *ptr,const TYPE *value,const int *index,const TYPE* x,TYPE* y)
{
    const double *val_ ;val_ = value;
    const int *idx_ ; idx_ = index;
    int n = 0;
    double sum;
    for(int i = 0;i < rows;i++){
        val_ += n;
        idx_ += n;
        n = ptr[i+1] - ptr[i];
        sum = 0.;
        Dot(&sum,x,val_,idx_,n);
        y[i] = sum;
    }
}
//void spmv_avx(const int rows,const int cols,const int *ptr,const TYPE *value,const int *index,const TYPE* x,TYPE* y)
//{
//    const double *val_ ;val_ = value;
//    const int *idx_ ;idx_ = index;
//    double sum;
//    for(int i = 0;i < rows;i++){
//        int n = ptr[i+1] - ptr[i];
//        val_ += ptr[i];
//        idx_ += ptr[i];
//        sum = 0.;
//        Dot_AVX512(&sum,x,val_,idx_,n);
//        y[i] = sum;
//    }
//}

