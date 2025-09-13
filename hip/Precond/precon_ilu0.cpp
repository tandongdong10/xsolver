#include "precon_ilu0.h"
void Precond_ilu0::preconditioner_init(){
	int nnzLU=hostmtx->nnz;
	hostmtxL=new HostMatrixCSR();
	hostmtxL->MallocMatrix(n_p,nnzLU);
	hostmtxU=new HostMatrixCSR();
	hostmtxU->MallocMatrix(n_p,nnzLU);
	hostmtx->seqilu0(hostmtxL,hostmtxU,diag_val);
}
void Precond_ilu0::preconditioner(HostVector *x_vec,HostVector *y_vec){
	hostmtxL->LUsolve(hostmtxU,diag_val,x_vec,y_vec);
}
void Precond_ilu0::preconditioner_free(){
        delete []diag_val;
}
Precond* precond_set_ilu0(){
    return new Precond_ilu0();
}
