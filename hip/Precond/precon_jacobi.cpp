#include "precon_jacobi.h"
void Precond_jacobi::create_precond(int n_in, HostMatrix *hstmtx){
	n_p=n_in;
	diag_val=new HostVector();
	diag_val->MallocVector(n_p);
	hstmtx->getdiag(diag_val);
}
void Precond_jacobi::create_precond(int n_in, double *diag_val_in){
	n_p=n_in;	
	diag_val=new HostVector();
	diag_val->MallocVector(n_p,diag_val_in);
}
void Precond_jacobi::preconditioner_init(){
	diag=new HostVector();
    	diag->MallocVector(n_p);
    	diag->jacobiInit(diag_val,small);
}
void Precond_jacobi::preconditioner_update(HostMatrix *hstmtx){
	hstmtx->getdiag(diag_val);
    	diag->jacobiInit(diag_val,small);
}
void Precond_jacobi::preconditioner_update(double *diag_val_in){
    	diag_val->CopyVector(n_p,diag_val_in);
    	diag->jacobiInit(diag_val,small);
}
void Precond_jacobi::preconditioner(HostVector *x,HostVector *y){
    	diag->jacobiSolve(x,y);
}
void Precond_jacobi::preconditioner_free(){
	diag->FreeVector();
}
Precond* precond_set_jacobi(){
    return new Precond_jacobi();
}
