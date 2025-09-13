#include "precon_gpujacobi.h"
void Precond_gpujacobi::create_precond(int n_in, HostMatrix *hstmtx){
	n_p=n_in;
	diag_val=new DeviceVector();
    	diag_val->MallocVector(n_p);
	hstmtx->getdiag(diag_val);
}
void Precond_gpujacobi::create_precond(int n_in, double *diag_val_in){
	n_p=n_in;	
	diag_val=new DeviceVector();
    	diag_val->MallocVector(n_p,diag_val_in);
}
void Precond_gpujacobi::preconditioner_init(){
	diag=new DeviceVector();
    	diag->MallocVector(n_p);
    	diag->jacobiInit(diag_val,small);
}
void Precond_gpujacobi::preconditioner_update(HostMatrix *hstmtx){
	hstmtx->getdiag(diag_val);
    	diag->jacobiInit(diag_val,small);
}
void Precond_gpujacobi::preconditioner_update(double *diag_val_in){
    	diag_val->UpdateVector(n_p,diag_val_in);
    	diag->jacobiInit(diag_val,small);
}
void Precond_gpujacobi::preconditioner(HostVector *x,HostVector *y){
    	diag->jacobiSolve(x,y);
}
void Precond_gpujacobi::preconditioner_free(){
	diag->FreeVector();
}
Precond* precond_set_gpujacobi(){
    return new Precond_gpujacobi();
}
