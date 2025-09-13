#include "precon_ilup_mkl.h"
void Precond_ilup_mkl::preconditioner_init(){
	int nnzLU=(2*maxfil+1)*n_p-maxfil*(maxfil+1)+1;
	hostmtxLU=new HostMatrixCSR();
	hostmtxLU->MallocMatrix(n_p,nnzLU);
	hostmtx->seqilut_mkl(hostmtxLU,maxfil,ilut_tol);
	xtmp_vec=new HostVector();
	xtmp_vec->MallocVector(n_p);
}
void  Precond_ilup_mkl::preconditioner(HostVector *x,HostVector *y){
	hostmtxLU->Lsolve(x,xtmp_vec);
	hostmtxLU->Usolve(xtmp_vec,y);
}
void Precond_ilup_mkl::preconditioner_free(){
	hostmtxLU->FreeMatrix();
	delete hostmtxLU;
	xtmp_vec->FreeVector();
	delete xtmp_vec;
}
Precond* precond_set_ilup_mkl(){
    return new Precond_ilup_mkl();
}
