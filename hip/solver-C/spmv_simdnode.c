#include <stdio.h>
#include <stdbool.h>
#include <arm_neon.h>

#define DATA_SIZE 1 //double 8 BYTE
#define LEVEL1_DCACHE_LINESIZE 32 //L1cache_line size BYTE
#define PETSC_PREFETCH_HINT_NTA 0 //prefech time locality
#define PrefetchBlock(a,n,rw,t) do {                               \
    const char *_p = (const char*)(a),*_end = (const char*)((a)+(n * DATA_SIZE));   \
    for (; _p < _end; _p += LEVEL1_DCACHE_LINESIZE) __builtin_prefetch(_p,(rw),(t)); \
  } while (0)


void spmv_node_simd(const int rows,const int cols,const int *ptr,const double *value,const int *col_idx, const double *x, double *y,\
               const int node_max, const int *ns)
{
  double       sum1,sum2,sum3,sum4,sum5,tmp0,tmp1;
  const double   *v1,*v2,*v3,*v4,*v5;
  double x_tmp[4], v_tmp[4];
  int          i1,i2,i3,i4,n,i,row,nsz,sz;
  const int    *idx,*ii;
 
  #pragma disjoint(*x,*y,*v1,*v2,*v3,*v4,*v5)
  
  v1  = value;
  idx = col_idx;
  ii  = ptr;

  for (i = 0,row = 0; i< node_max; ++i) {
    nsz         = ns[i];
    n           = ii[1] - ii[0];
    ii         += nsz;
    PrefetchBlock(idx+nsz*n,n,0,PETSC_PREFETCH_HINT_NTA);    /* Prefetch the indices for the block row after the current one */
    PrefetchBlock(v1+nsz*n,nsz*n,0,PETSC_PREFETCH_HINT_NTA); /* Prefetch the values for the block row after the current one  */
    //Prefetch_arm(idx+nsz*n,v1+nsz*n,n);
    sz = n;                     /* Number of non zeros in this row */
                                /* Switch on the size of Node */
    
    switch (nsz) {               /* Each loop in 'case' is unrolled */
    case 1:
      sum1 = 0.;
      for (n = 0; n< sz-1; n+=4) {
        i1    = idx[0];         /* The instructions are ordered to */
        i2    = idx[1];         /* make the compiler's job easy */
        i3    = idx[2];
        i4    = idx[3];
        x_tmp[0] = x[i1];
        x_tmp[1] = x[i2];
        x_tmp[2] = x[i3];
        x_tmp[3] = x[i4];
        idx  += 4;
        sum1 += v1[0]*x_tmp[0] + v1[1]*x_tmp[1] + v1[2]*x_tmp[2]+v1[3]*x_tmp[3];v1 += 4;
      }

      while (n <= sz-1) {          /* Take care of the last nonzero  */
        x_tmp[0]  = x[*idx++];
        sum1 += *v1++ *x_tmp[0];
        n++;
      }
      y[row++]=sum1;
      break;
    case 2:
        //每一行的两列放置在一个向量寄存器中累加
      sum1 = 0.;
      sum2 = 0.;
      v2   = v1 + n;
      float64x2_t val_v,x1_v,x2_v,y1_v,y2_v;
      y1_v[0] = 0.,y1_v[1] = 0.,y2_v[0]=0.,y2_v[1]=0.;
      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        x_tmp[0]  = x[i1];
        x_tmp[1]  = x[i2];
        val_v = vld1q_f64(x_tmp);
        x1_v = vld1q_f64(v1);
        x2_v = vld1q_f64(v2);
        y1_v = vfmaq_f64(y1_v,x1_v,val_v);
        y2_v = vfmaq_f64(y2_v,x2_v,val_v);
        //sum1 += v1[0] * tmp0 + v1[1] * tmp1;
        //sum2 += v2[0] * tmp0 + v2[1] * tmp1;
        v1 += 2;v2 += 2;
      }
      sum1 = y1_v[0]+y1_v[1];
      sum2 = y2_v[0]+y2_v[1];
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      v1      =v2;              // Since the next block to be processed starts there
      idx    +=sz;
      break;
      /*
       case 2:
        //每两行的一列放置在一个向量寄存器中累加
      sum1 = 0.;
      sum2 = 0.;
      v2   = v1 + n;
      float64x1_t x_1,x_2;
      float64x2_t val_v,x1_v,x2_v,y1_v,y2_v;
      y1_v[0] = 0.,y1_v[1] = 0.,y2_v[0]=0.,y2_v[1]=0.;
      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        x_1  = vld1_f64(&x[i1]);
        x_2  = vld1_f64(&x[i2]);
        v_tmp[0] = v1[0];v_tmp[1] = v2[0];
        v_tmp[2] = v1[1];v_tmp[3] = v2[1];
        x1_v = vld1q_f64(v_tmp);
        x2_v = vld1q_f64(&v_tmp[2]);
        y1_v = vfmaq_lane_f64(y1_v,x1_v,x_1,0);
        y2_v = vfmaq_lane_f64(y2_v,x2_v,x_2,0);
        v1 += 2;v2 += 2;
      }
      sum1 = y1_v[0]+y2_v[0];
      sum2 = y1_v[1]+y2_v[1];
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      v1      =v2;              // Since the next block to be processed starts there
      idx    +=sz;
      break;
    */
    case 3:
      sum1 = 0.;
      sum2 = 0.;
      sum3 = 0.;
      v2   = v1 + n;
      v3   = v2 + n;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
      }
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      v1      =v3;              /* Since the next block to be processed starts there*/
      idx    +=2*sz;
      break;
      /*
        case 4:
            sum1 = 0.;
            sum2 = 0.;
            sum3 = 0.;
            sum4 = 0.;
            v2   = v1 + n;
            v3   = v2 + n;
            v4   = v3 + n;

            for (n = 0; n< sz-1; n+=2) {
                i1    = idx[0];
                i2    = idx[1];
                idx  += 2;
                tmp0  = x[i1];
                tmp1  = x[i2];
                sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
                sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
                sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
                sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
            }
            if (n == sz-1) {
                tmp0  = x[*idx++];
                sum1 += *v1++ * tmp0;
                sum2 += *v2++ * tmp0;
                sum3 += *v3++ * tmp0;
                sum4 += *v4++ * tmp0;
            }
            y[row++]=sum1;
            y[row++]=sum2;
            y[row++]=sum3;
            y[row++]=sum4;
            v1      =v4;              // Since the next block to be processed starts there
            idx    +=3*sz;
            break;
      */
    case 4:
      sum1 = 0.;
      sum2 = 0.;
      sum3 = 0.;
      sum4 = 0.;
      v2   = v1 + n;
      v3   = v2 + n;
      v4   = v3 + n;

      for (n = 0; n< sz-1; n+=1) {
        i1    = idx[0];
        idx  += 1;
        tmp0  = x[i1];
        sum1 += v1[0] * tmp0; v1 ++;
        sum2 += v2[0] * tmp0; v2 ++;
        sum3 += v3[0] * tmp0; v3 ++;
        sum4 += v4[0] * tmp0; v4 ++;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      v1      =v4;              /* Since the next block to be processed starts there*/
      idx    +=3*sz;
      break;
    case 5:
      sum1 = 0.;
      sum2 = 0.;
      sum3 = 0.;
      sum4 = 0.;
      sum5 = 0.;
      v2   = v1 + n;
      v3   = v2 + n;
      v4   = v3 + n;
      v5   = v4 + n;

      for (n = 0; n<=sz-1; n+=1) {
        i1    = idx[0];
        //i2    = idx[1];
        idx  += 1;
        tmp0  = x[i1];
        //tmp1  = x[i2];
        sum1 += v1[0] * tmp0 ; v1 ++;
        sum2 += v2[0] * tmp0 ; v2 ++;
        sum3 += v3[0] * tmp0 ; v3 ++;
        sum4 += v4[0] * tmp0 ; v4 ++;
        sum5 += v5[0] * tmp0 ; v5 ++;
      }
      /*
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
        sum4 += *v4++ * tmp0;
        sum5 += *v5++ * tmp0;
      }
      */
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      y[row++]=sum5;
      v1      =v5;       /* Since the next block to be processed starts there */
      idx    +=4*sz;
      break;
    default:
      printf("ERROR : Node size not yet supported");
    }
  }  
}