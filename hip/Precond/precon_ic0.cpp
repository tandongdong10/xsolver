#include "precon_ic0.h"
void Precond_ic0::preconditioner_init(){
	int nnzLU=hostmtx->nnz;
	hostmtxL=new HostMatrixCSR();
	hostmtxL->MallocMatrix(n_p,nnzLU);
	hostmtxU=new HostMatrixCSR();
	hostmtxU->MallocMatrix(n_p,nnzLU);
	hostmtx->seqic0(hostmtxL,hostmtxU,diag,smallnum,droptol);
}
void Precond_ic0::preconditioner(HostVector *x_vec,HostVector *y_vec){
	hostmtxL->LUsolve(hostmtxU,diag,x_vec,y_vec);
}
void Precond_ic0::preconditioner_free(){
	hostmtxL->FreeMatrix();
	delete hostmtxL;
	hostmtxU->FreeMatrix();
	delete hostmtxU;
        if(diag!=NULL){delete[]diag;diag=NULL;}
}
Precond* precond_set_ic0(){
    return new Precond_ic0();
}
