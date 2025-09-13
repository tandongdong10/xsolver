#include "precon_ilup.h"
void Precond_ilup::preconditioner_init(){
	int nnzLU=maxfil*n_p;
	hostmtxL=new HostMatrixCSR();
	hostmtxL->MallocMatrix(n_p,nnzLU);
	hostmtxU=new HostMatrixCSR();
	hostmtxU->MallocMatrix(n_p,nnzLU);
	hostmtx->seqilut(hostmtxL,hostmtxU,diag,maxfil,ilut_tol);
}
void Precond_ilup::preconditioner(HostVector *x,HostVector *y){
	hostmtxL->LUsolve(hostmtxU,diag,x,y);
}
void Precond_ilup::preconditioner_free(){
	hostmtxL->FreeMatrix();
	delete hostmtxL;
	hostmtxU->FreeMatrix();
	delete hostmtxU;
        if(diag!=NULL){delete[]diag;diag=NULL;}
}
Precond* precond_set_ilup(){
    return new Precond_ilup();
}
