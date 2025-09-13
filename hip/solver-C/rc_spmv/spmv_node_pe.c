#include <stdio.h>
#include <stdbool.h>

#define DATA_SIZE 1 //double 8 BYTE
#define LEVEL1_DCACHE_LINESIZE 32 //L1cache_line size BYTE
#define PETSC_PREFETCH_HINT_NTA 0 //prefech time locality
#define PrefetchBlock(a,n,rw,t) do {                               \
    const char *_p = (const char*)(a),*_end = (const char*)((a)+(n * DATA_SIZE));   \
    for (; _p < _end; _p += LEVEL1_DCACHE_LINESIZE) __builtin_prefetch(_p,(rw),(t)); \
  } while (0)

void Prefetch_arm(const int *idx,const double *value,const int cnt,int idx_cnt){
     int count_val = cnt * 8 ;
     int count_idx = idx_cnt * 4 ;
     int i;
     for(i = 0;i < count_idx;i += 64){
       __asm__ volatile("prfm pldl3strm, [%[idx]]\n\t""prfm pldl3strm, [%[value]]\n\t": :[idx]"r"(idx+i),[value]"r"(value+i));                   
     }
     for( ; i < count_val;i += 64){
       __asm__ volatile("prfm pldl3strm, [%[value]]\n\t": :[value]"r"(value+i));                   
     }
}
void spmv_csr_node(const int rows,const int cols,const int *ptr,const double *value,const int *col_idx, const double *x, double *y,\
               const int node_max, const int *ns)
{
    double       sum1,sum2,sum3,sum4,sum5,tmp0,tmp1;
    const double   *v1,*v2,*v3,*v4,*v5;
    int          i1,i2,n,i,row,nsz,sz;
    //int nonzerorow=0;
    const int    *idx,*ii;

#pragma disjoint(*x,*y,*v1,*v2,*v3,*v4,*v5)

    v1  = value;
    idx = col_idx;
    ii  = ptr;

    for (i = 0,row = 0; i< node_max; ++i) {
        nsz         = ns[i];
        n           = ii[1] - ii[0];
        //nonzerorow += (n>0)*nsz;
        ii         += nsz;
        PrefetchBlock(idx+nsz*n,n,0,PETSC_PREFETCH_HINT_NTA);    /* Prefetch the indices for the block row after the current one */
        PrefetchBlock(v1+nsz*n,nsz*n,0,PETSC_PREFETCH_HINT_NTA); /* Prefetch the values for the block row after the current one  */
        //Prefetch_arm(idx+nsz*n,v1+nsz*n,n);
        sz = n;                     /* Number of non zeros in this row */
        /* Switch on the size of Node */

        switch (nsz) {               /* Each loop in 'case' is unrolled */
            case 1:
                sum1 = 0.;

                for (n = 0; n< sz-1; n+=2) {
                    i1    = idx[0];         /* The instructions are ordered to */
                    i2    = idx[1];         /* make the compiler's job easy */
                    idx  += 2;
                    tmp0  = x[i1];
                    tmp1  = x[i2];
                    sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
                }

                if (n == sz-1) {          /* Take care of the last nonzero  */
                    tmp0  = x[*idx++];
                    sum1 += *v1++ *tmp0;
                }
                y[row++]=sum1;
                break;
            case 2:
                sum1 = 0.;
                sum2 = 0.;
                v2   = v1 + n;

                for (n = 0; n< sz-1; n+=2) {
                    i1    = idx[0];
                    i2    = idx[1];
                    idx  += 2;
                    tmp0  = x[i1];
                    tmp1  = x[i2];
                    sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
                    sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
                }
                if (n == sz-1) {
                    tmp0  = x[*idx++];
                    sum1 += *v1++ * tmp0;
                    sum2 += *v2++ * tmp0;
                }
                y[row++]=sum1;
                y[row++]=sum2;
                v1      =v2;              /* Since the next block to be processed starts there*/
                idx    +=sz;
                break;
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

                for (n = 0; n<sz-1; n+=2) {
                    i1    = idx[0];
                    i2    = idx[1];
                    idx  += 2;
                    tmp0  = x[i1];
                    tmp1  = x[i2];
                    sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
                    sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
                    sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
                    sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
                    sum5 += v5[0] * tmp0 + v5[1] *tmp1; v5 += 2;
                }
                if (n == sz-1) {
                    tmp0  = x[*idx++];
                    sum1 += *v1++ * tmp0;
                    sum2 += *v2++ * tmp0;
                    sum3 += *v3++ * tmp0;
                    sum4 += *v4++ * tmp0;
                    sum5 += *v5++ * tmp0;
                }
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
void spmv_ell_node(const int rows,const int cols,const int width, const double *value,const int *col_idx, const double *x, double *y,\
               const int node_max, const int *ns)
{
  double       sum1,sum2,sum3,sum4,sum5,tmp0,tmp1;
  const double   *v1,*v2,*v3,*v4,*v5,*v1_;
  int          i1,i2,n,i,row,nsz,sz;
  //int nonzerorow=0;
  const int    *idx,*idx_;
 
  #pragma disjoint(*x,*y,*v1,*v2,*v3,*v4,*v5)
  
  v1  = value;
  idx = col_idx;

  for (i = 0,row = 0; i< node_max; ++i) {
    v1_ = v1;
    idx_ = idx;
    nsz         = ns[i];
    n           = width;
    //nonzerorow += (n>0)*nsz;
    PrefetchBlock(idx+nsz*n,n,0,PETSC_PREFETCH_HINT_NTA);    /* Prefetch the indices for the block row after the current one */
    PrefetchBlock(v1+nsz*n,nsz*n,0,PETSC_PREFETCH_HINT_NTA); /* Prefetch the values for the block row after the current one  */
    //Prefetch_arm(idx+nsz*n,v1+nsz*n,n,n);
    sz = n;                     /* Number of non zeros in this row */
                                /* Switch on the size of Node */
    
    switch (nsz) {               /* Each loop in 'case' is unrolled */
    case 1:
      sum1 = 0.;

      for (n = 0; n< sz-1; n+=2) {
          if(idx[1]==-1){*idx--;break;}
          i1    = idx[0];         /* The instructions are ordered to */
          i2    = idx[1];         /* make the compiler's job easy */
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
      }

      if (n <= sz-1) {/* Take care of the last nonzero  */
          if(*idx++ == -1)break;
          tmp0  = x[*idx];
          sum1 += *v1++ *tmp0;
          
      }
      y[row++]=sum1;
      v1      = v1_  + sz;
      idx     = idx_ + sz;
      break;
    case 2:
      sum1 = 0.;
      sum2 = 0.;
      v2   = v1 + n;

      for (n = 0; n< sz-1; n+=2) {
          if(idx[1]==-1){*idx--;break;}//
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
      }
      if (n <= sz-1) {
          if(*idx++ == -1)break;
          tmp0  = x[*idx];
          sum1 += *v1++ * tmp0;
          sum2 += *v2++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      v1      =v1_  + 2*sz;              /* Since the next block to be processed starts there*/
      idx     =idx_ + 2*sz;
      break;
    case 3:
      sum1 = 0.;
      sum2 = 0.;
      sum3 = 0.;
      v2   = v1 + n;
      v3   = v2 + n;

      for (n = 0; n< sz-1; n+=2) {
          if(idx[1]==-1){*idx--;break;}
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          sum3 += v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
      }
      if (n <= sz-1) {
          if(*idx++ == -1)break;
          tmp0  = x[*idx];
          sum1 += *v1++ * tmp0;
          sum2 += *v2++ * tmp0;
          sum3 += *v3++ * tmp0;
          
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      v1      =v1_ + 3*sz;              /* Since the next block to be processed starts there*/
      idx     =idx_ + 3*sz;
      break;
    case 4:
      sum1 = 0.;
      sum2 = 0.;
      sum3 = 0.;
      sum4 = 0.;
      v2   = v1 + n;
      v3   = v2 + n;
      v4   = v3 + n;

      for (n = 0; n< sz-1; n+=2) {
          if(idx[1]==-1){*idx--;break;}
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
      if (n <= sz-1) {
          if(*idx++ == -1)break;
          tmp0  = x[*idx];
          sum1 += *v1++ * tmp0;
          sum2 += *v2++ * tmp0;
          sum3 += *v3++ * tmp0;
          sum4 += *v4++ * tmp0;
          
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      v1     = v1_  + 4*sz;              /* Since the next block to be processed starts there*/
      idx    = idx_ + 4*sz;
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

      for (n = 0; n<sz-1; n+=2) {
          if(idx[1]==-1){*idx--;break;}
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
          sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
          sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
          sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
          sum5 += v5[0] * tmp0 + v5[1] *tmp1; v5 += 2;
      }
      if (n <= sz-1) {
          if(*idx++ == -1)break;
          tmp0  = x[*idx];
          sum1 += *v1++ * tmp0;
          sum2 += *v2++ * tmp0;
          sum3 += *v3++ * tmp0;
          sum4 += *v4++ * tmp0;
          sum5 += *v5++ * tmp0;
          
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      y[row++]=sum5;
      v1     = v1_  + 5*sz;              /* Since the next block to be processed starts there*/
      idx    = idx_ + 5*sz;
      break;
    default:
      printf("ERROR : Node size not yet supported");
    }
  }  
}

