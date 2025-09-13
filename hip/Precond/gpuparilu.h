#ifndef _GPUPARILU_H_
#define _GPUPARILU_H_
#include "par_ilu_hip/Asplit_gpu.h"
#define T double
__global__ void empty_launch(int n);
void empty_launch_gpu(int n);
__global__ void set_reference(T*d_val,int*d_rows,int*d_cols,int n,int*d_row_reference);

void parilu_pre_set(T*d_val,int*d_rows,int*d_cols,int n,int nnz,
T*&d_lval,int*&d_lrows,int*&d_lcols,int&gnnzl,
T*&d_uval,int*&d_ucols,int*&d_urows,int&gnnzu,
int*&d_row_reference);
__global__ void ilu0_ij_aw(T*lval,int*lrows,int*lcols,T*uval,int*urows,int*ucols,int n,\
    int*rows,int*cols,int*row_reference,T*val,int nnz);
__global__ void ilu0_aij_Ldiag1_Udiag_try_faster2(T*lval,int*lrows,int*lcols,\
    T*uval,int*urows,int*ucols,int n,\
    int*rows,int*cols,int*row_reference,T*val,int nnz);
void parilu_fact(T*&vald,int*&rowsd,int*&colsd,int*&row_referenced,\
                T*&lvald,int*&lrowsd,int*&lcolsd,\
                T*&uvald,int*&ucolsd,int*&urowsd,int n,int nnz,int sweep);
__global__ void ilu0_aij_Ldiag1_Udiag_try_faster2_shared(T*lval,int*lrows,int*lcols,T*uval,int*urows,int*ucols,int n,\
    int*rows,int*cols,int*row_reference,T*val,int nnz);
void parilu_fact_shared(T*&vald,int*&rowsd,int*&colsd,int*&row_referenced,\
                T*&lvald,int*&lrowsd,int*&lcolsd,\
                T*&uvald,int*&ucolsd,int*&urowsd,int n,int nnz,int sweep);
__global__ void Lcsr_trsv_sync_free(T*lval,int*lrows,int*lcols,int n,T*x,T*y,volatile int*set);

void lcsr_trsv(T*&lvald,int*&lrowsd,int*&lcolsd,T*&xd,T*&yd,int n);
void lcsr_trsv_setd(T*&lvald,int*&lrowsd,int*&lcolsd,T*&xd,T*&yd,int n,int*setd);

__global__ void Ucsc_trsv_sync_free_pre(int*ucols,int*urows,int n,int*d_indegree);
__global__ void Ucsc_trsv_sync_free(T*uval,int*ucols,int*urows,int n,T*y,T*x,T*d_left_sum,volatile int*d_indegree);

void ucsc_trsv(T*&uvald,int*&ucolsd,int*&urowsd,T*&x,T*&y,int n,int nnzu);

void ucsc_trsv_leftsum_indegree(T*&uvald,int*&ucolsd,int*&urowsd,T*&x,T*&y,int n,int nnzu,\
            T*d_left_sum,int*d_indegree);
void free_on_gpu(T*&lvald,int*&lrowsd,int*&lcolsd,\
                T*&uvald,int*&ucolsd,int*&urowsd,\
                int*&row_referenced);
void do_factorization(T*val,int*rows,int*cols,int n,int nnz,\
										T*&lvald,int*&lrowsd,int*&lcolsd,int&nnzl,\
                                        T*&uvald,int*&ucolsd,int*&urowsd,int&nnzu,\
                                        int*&row_referenced,int sweep);

void do_solve(T*x,T*y,int n,\
						T*lvald,int*lrowsd,int*lcolsd,\
                        T*uvald,int*ucolsd,int*urowsd,int nnzu);

#undef T
#endif
