#include "precon_ict.h"
void Precond_ict::preconditioner_init(){
	if(maxfil>n_p-1)
	    maxfil=n_p-1;
	int nnzLU=maxfil*n_p+n_p;
	hostmtxL=new HostMatrixCSR();
	hostmtxL->MallocMatrix(n_p,nnzLU);
	hostmtxU=new HostMatrixCSR();
	hostmtxU->MallocMatrix(n_p,nnzLU);
	hostmtx->seqict(hostmtxL,hostmtxU,diag,smallnum,droptol,maxfil);
}
void Precond_ict::preconditioner(HostVector *x_vec,HostVector *y_vec){
	hostmtxL->LUsolve(hostmtxU,diag,x_vec,y_vec);
}
void Precond_ict::preconditioner_free(){
	hostmtxL->FreeMatrix();
	delete hostmtxL;
	hostmtxU->FreeMatrix();
	delete hostmtxU;
        if(diag!=NULL){delete[]diag;diag=NULL;}
	
}
Precond* precond_set_ict(){
    return new Precond_ict();
}
