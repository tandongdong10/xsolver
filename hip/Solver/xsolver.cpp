#include "xsolver.h"
extern int pam[3];
extern Precond* precond_set_jacobi();
extern Precond* precond_set_gpujacobi();
extern Precond* precond_set_ilu0();
extern Precond* precond_set_ilu0_mkl();
extern Precond* precond_set_gpuilu0();
extern Precond* precond_set_ilup();
extern Precond* precond_set_ilup_mkl();
extern Precond* precond_set_gpuilut();
extern Precond* precond_set_ic0();
extern Precond* precond_set_ict();
void Xsolver::preconditioner_set(PRECON precon){
	if(precon==1&&pam[2]!=1)
	    precond=precond_set_jacobi();
	if(precon==1&&pam[2]==1){
	    precond=precond_set_gpujacobi();
	}
	if(precon==2&&pam[2]!=1)
	    precond=precond_set_ilu0();
	if(precon==2&&pam[2]==1)
	    precond=precond_set_gpuilu0();
	if(precon==3&&pam[2]!=1)
	    precond=precond_set_ilup();
	if(precon==3&&pam[2]==1)
	    precond=precond_set_gpuilut();
	if(precon==7)
	    precond=precond_set_ilu0_mkl();
	if(precon==8)
	    precond=precond_set_ilup_mkl();
	if(precon==9)
	    precond=precond_set_ic0();
	if(precon==10)
	    precond=precond_set_ict();
}
template <typename VectorType>
extern Xsolver* solver_set_bicgstab();
template <typename VectorType>
extern Xsolver* solver_set_igcr();
template <typename VectorType>
extern Xsolver* solver_set_cg();
template <typename VectorType>
Xsolver* solver_set(SOLVER solver){
    if(solver==1)
	return solver_set_bicgstab<VectorType>();	
    else if(solver==2)
	return solver_set_igcr<VectorType>();	
    else if(solver==4)
	return solver_set_cg<VectorType>();	
    else{
	printf("Solver Set Wrong!!!\n");
	return NULL;
    }
}
