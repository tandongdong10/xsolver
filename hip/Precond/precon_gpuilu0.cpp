#include "precon_gpuilu0.h"
void Precond_gpuilu0::preconditioner_init(){
	devicemtxL=new DeviceMatrixCSR();
	devicemtxU=new DeviceMatrixCSR();
	int sweep=20;
	diag_U=new DeviceVector();
	diag_U->MallocVector(n_p);
        //hostmtx->parilu(devicemtxL,devicemtxU,&row_referenced,sweep);//parilu <->Lsolve Usolve
        hostmtx->parilu_csr(devicemtxL,devicemtxU,&row_referenced,diag_U,sweep);//parilu_csr <->Lsolve_iter Usolve_iter
			
	xtmp_vec=new DeviceVector();
	xtmp_vec->MallocVector(n_p);
	tmp=new DeviceVector();
	tmp->MallocVector(n_p);
}
void Precond_gpuilu0::preconditioner(HostVector *x_vec,HostVector *y_vec){
	devicemtxL->Lsolve_iter(x_vec,xtmp_vec,tmp,20);
	devicemtxU->Usolve_iter(xtmp_vec,y_vec,tmp,diag_U,20);
}
void Precond_gpuilu0::preconditioner_free(){
	xtmp_vec->FreeVector();
	tmp->FreeVector();
	diag_U->FreeVector();
	devicemtxL->FreeMatrix();
	devicemtxU->FreeMatrix();
	delete xtmp_vec;
	delete tmp;
	delete diag_U;
	delete devicemtxL;
	delete devicemtxU;
}
Precond* precond_set_gpuilu0(){
    return new Precond_gpuilu0();
}
