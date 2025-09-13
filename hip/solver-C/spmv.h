#ifndef SPMV_H
#define SPMV_H

#ifndef TYPE
#define TYPE double
#endif
void spmv( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_unroll1( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_unroll2( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_unroll4( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_csr( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_pre( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_ell( const int rows, const int cols, const TYPE *value, const int *index, const int width, const TYPE *x, TYPE *y);
void spmv_ell_unroll2( const int rows, const int cols, const TYPE *value, const int *index, const int width, const TYPE *x, TYPE *y);
void spmv_ell_0( const int rows, const int cols, const TYPE *value, const int *index, const int width, const TYPE *x, TYPE *y);
void spmv_csc(const int rows,const int cols,const int *ptr,const double*value,const int*row_idx,const double *x,double*y);
void spmv_csr_node(const int rows,const int cols,const int *ptr,const double *value,const int *col_idx, const double *x, double *y,\
               const  int node_max, const int *ns);
void spmv_csr_node_pe(const int rows,const int cols,const int *ptr,const double *value,const int *col_idx, const double *x, double *y,\
               const  int node_max, const int *ns);
void spmv_csr_node_muti(const int rows,const int cols,const int *ptr,const double *value,const int *col_idx, const double *x, double *y,\
               const  int node_max, const int *ns, const int *part_ns, const int *part_row,int thread_num);
void spmv_node_simd(const int rows,const int cols,const int *ptr,const double *value,const int *col_idx, const double *x, double *y,\
               const  int node_max, const int *ns);
void spmv_ell_node(const int rows,const int cols,const int width,const double *value,const int *col_idx, const double *x, double *y,\
               const  int node_max, const int *ns);
void spmv_ell_node_muti(const int rows,const int cols,const int width,const double *value,const int *col_idx, const double *x, double *y,\
               const  int node_max, const int *ns,const int *part_ns, const int *part_row,int thread_num);
void mtx_dis_anl(const int rows,const int cols,const int nnz,const int *ptr,int *width,int *minwid,int *a,int *b,int *c,int *d,int *e);
void mtx_analyse(const int rows,const int cols,const int *ptr,const double *value,const int *col_idx,\
                int *node_max, const int *ns);
void mtx_analyse_muti(const int rows,const int nnz,const int *ptr,const double *value,const int *col_idx,\
                int *node_max, const int *ns, const int *part_ns, int *part_row,int thread_num);
void spmv_rcell(const int rows,const int cols,const double* value,const int* index,const int width,const double* x,double* y);
void spmv_2_pre( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_4_pre( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_unroll2_pre( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
void spmv_unroll4_pre( const int rows, const int cols, const int *ptr, const TYPE *value, const int *col, const TYPE *x, TYPE *y);
#endif

