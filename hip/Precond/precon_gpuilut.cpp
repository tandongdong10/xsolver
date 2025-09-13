#include "precon_gpuilut.h"
void Precond_gpuilut::preconditioner_init(){
	devicemtxL=new DeviceMatrixCSR();
	devicemtxU=new DeviceMatrixCSR();
	int sweep=10;
	diag_U=new DeviceVector();
	diag_U->MallocVector(n_p);
        //hostmtx->parilut(devicemtxL,devicemtxU,sweep);//parilut <->Lsolve Usolve
        hostmtx->parilut_csr(devicemtxL,devicemtxU,diag_U,sweep);//parilut_csr <->Lsolve_iter Usolve_iter
	xtmp_vec=new DeviceVector();
	xtmp_vec->MallocVector(n_p);
	tmp=new DeviceVector();
	tmp->MallocVector(n_p);
    	//printf("nnzl=%d nnzu=%d nnzA=%d\n",devicemtxL->nnz,devicemtxU->nnz,hostmtx->nnz);
}
void Precond_gpuilut::preconditioner(HostVector *x_vec,HostVector *y_vec){
	devicemtxL->Lsolve_iter(x_vec,xtmp_vec,tmp,10);
	devicemtxU->Usolve_iter(xtmp_vec,y_vec,tmp,diag_U,15);
}
void Precond_gpuilut::preconditioner_free(){
	xtmp_vec->FreeVector();
	tmp->FreeVector();
	diag_U->FreeVector();
	devicemtxL->FreeMatrix();
	devicemtxU->FreeMatrix();
}
Precond* precond_set_gpuilut(){
    return new Precond_gpuilut();
}
