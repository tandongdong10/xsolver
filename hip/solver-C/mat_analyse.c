#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define Max(x,y) ((x)>(y) ? (x) : (y))
#define Min(x,y) ((x)<(y) ? (x) : (y))
void mtx_dis_anl(const int rows,const int cols,const int nnz,const int *ptr,int *width,int *minwid,\
                 int *a,int *b,int *c,int *d,int *e){
   int i = 0,avr_nnz = (nnz + rows - 1)/rows;
   *width = 0;*minwid = cols;
   *a = 0;
   *b = 0;
   *c = 0;
   *d = 0;
   *e = 0;
   for(i = 1;i <= rows;i++){
      int tmp = ptr[i] - ptr[i-1];
      if(tmp <= 1.1 * avr_nnz && tmp >= 0.9 * avr_nnz){(*a) += 1;}
      if(tmp <= 1.2 * avr_nnz && tmp >= 0.8 * avr_nnz){(*b) += 1;}
      if(tmp <= 1.3 * avr_nnz && tmp >= 0.7 * avr_nnz){(*c) += 1;}
      if(tmp <= 1.4 * avr_nnz && tmp >= 0.6 * avr_nnz){(*d) += 1;}
      if(tmp <= 1.5 * avr_nnz && tmp >= 0.5 * avr_nnz){(*e) += 1;}
      *width = Max(*width,tmp);
      *minwid = Min(*minwid,tmp);
   }
}
void Memcmp(const void *str1,const void *str2,size_t len,bool *e)
{
  int r;
  if (len > 0 && !str1) printf("ERROR: Trying to compare at a null pointer\n");
  if (len > 0 && !str2) printf("ERROR: Trying to compare at a null pointer\n");
  r = memcmp((char*)str1,(char*)str2,len);
  if (!r) *e = true;
  else    *e = false;
}

//#define  Arraycmp(str1,str2,cnt,e) Memcmp(str1,str2,(size_t)(cnt)*sizeof(*(str1)),e))

void mtx_analyse(const int rows,const int cols,const int *ptr,const double *value,const int *col_idx,\
                int *node_max, int *ns)
{
    int MAX_NODE = 5;
    int       i,j,nzx,nzy,blk_size,node_count;
    bool      flag;
    const int *idx,*idy;
    idx = col_idx;
    i          = 0;
    node_count = 0;
    while (i < rows) {                /* For each row */
        nzx = ptr[i+1] - ptr[i];       /* Number of nonzeros */
        /* Limits the number of elements in a node to 'a->inode.limit' */
        for (j=i+1,idy=idx,blk_size=1; j<rows && blk_size < MAX_NODE; ++j,++blk_size) {
            nzy = ptr[j+1] - ptr[j];     /* Same number of nonzeros */
            if (nzy != nzx) break;
            idy += nzx;              /* Same nonzero pattern */
            //Arraycmp(idx,idy,nzx,&flag);
            Memcmp(idx,idy,(size_t)(nzx)*sizeof(*(idx)),&flag);
            if (!flag) break;
        }
        ns[node_count++] = blk_size;
        idx             += blk_size*nzx;
        i                = j;
    }
    *node_max = node_count; 
    /* If not enough inodes found,, do not use inode version of the routines */
    if (!rows || node_count > .8*rows) {
        //printf("ATENTION: please use the original spmv\n");
    } else {
        //printf("ATENTION: please use the inode spmv\n");
    }

}
void mtx_analyse_muti(const int rows,const int nnz,const int *ptr,const double *value,const int *col_idx,\
                int *node_max, int *ns,int *part_ns,int *part_row,int thread_num)
{
    int thread_avr_nnz = (nnz + thread_num-1) / thread_num;
    int MAX_NODE = 5;
    int       th_i,i,j,nzx,nzy,blk_size,node_count,node_nnz;
    bool      flag;
    const int *idx,*idy;
    idx = col_idx;
    i          = 0;
    node_count = 0;
    node_nnz = 0;
    th_i = 0;
    part_row[th_i] = 0,part_ns[th_i] = 0;th_i++;
    while (i < rows) {                /* For each row */
        nzx = ptr[i+1] - ptr[i];       /* Number of nonzeros */
        /* Limits the number of elements in a node to 'a->inode.limit' */
        for (j=i+1,idy=idx,blk_size=1; j<rows && blk_size < MAX_NODE; ++j,++blk_size) {
            nzy = ptr[j+1] - ptr[j];     /* Same number of nonzeros */
            if (nzy != nzx) break;
            idy += nzx;              /* Same nonzero pattern */
            //Arraycmp(idx,idy,nzx,&flag);
            Memcmp(idx,idy,(size_t)(nzx)*sizeof(*(idx)),&flag);
            if (!flag) break;
        }
        node_nnz += blk_size*nzx;
        idx             += blk_size*nzx;
        ns[node_count++] = blk_size;
        if(th_i * thread_avr_nnz <= node_nnz){
            part_ns[th_i] = node_count;
            part_row[th_i] = j;
            th_i++;
        }
        i = j;
    }
    for(;th_i <= thread_num;th_i++){
        part_ns[th_i] = node_count;
        part_row[th_i] = rows;
    }
    *node_max = node_count;
    /* If not enough inodes found,, do not use inode version of the routines */
    if (!rows || node_count > .8*rows) {
        //printf("ATENTION: please use the original spmv\n");
    } else {
        //printf("ATENTION: please use the inode spmv\n");
    }
}
