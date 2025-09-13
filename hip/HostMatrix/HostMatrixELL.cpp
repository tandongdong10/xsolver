#include "HostMatrixELL.h"
void HostMatrixELL::getdiag(double *a_p){
    int num_nz=nnz/m;
    for(int i=0;i<m;i++){
        for(int j=i*num_nz;j<(i+1)*num_nz;j++)
    	if(colidx[j]==i)
    	    a_p[i]=val[j];
    }
}
void HostMatrixELL::SpMV(HostVector *x, HostVector *y)
{
    double one=1,zero=0;

#ifdef HAVE_MPI
    communicator_p2p(x->val);
    communicator_p2p_waitall();
#endif
    spmv_ell ( m, n, val, colidx, nnz/m, x->val, y->val);
}

void HostMatrixELL::bmAx(HostVector *rhs, HostVector *x, HostVector *y)
{
#ifdef HAVE_MPI
    communicator_p2p(x->val);
    communicator_p2p_waitall();
#endif
    bmax_ell ( m, n, val, colidx, nnz/m, rhs->val, x->val, y->val);
}
HostMatrix* set_matrix_ell(){
    return new HostMatrixELL();
}
