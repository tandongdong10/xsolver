#include "precon_ilu0_mkl.h"
void Precond_ilu0_mkl::preconditioner_init(){
	int nnz=hostmtx->nnz;
	bilu0= new double[nnz];
	hostmtxLU=new HostMatrixCSR();
	hostmtxLU->create_matrix(n_p,bilu0,hostmtx->getptr(),hostmtx->getidx());
	hostmtx->seqilu0_mkl(hostmtxLU);
	xtmp_vec=new HostVector();
	xtmp_vec->MallocVector(n_p);
}
void Precond_ilu0_mkl::preconditioner(HostVector *x_vec,HostVector *y_vec){
	hostmtxLU->Lsolve(x_vec,xtmp_vec);
	hostmtxLU->Usolve(xtmp_vec,y_vec);
}
void Precond_ilu0_mkl::preconditioner_free(){
	delete []bilu0;
	xtmp_vec->FreeVector();
	delete hostmtxLU;
	delete xtmp_vec;
}
Precond* precond_set_ilu0_mkl(){
    return new Precond_ilu0_mkl();
}
