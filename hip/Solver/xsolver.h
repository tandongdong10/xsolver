#ifndef _XSOLVER_H
#define _XSOLVER_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "../Precond/preconditioner.h"
#include "../HostMatrix/HostMatrix.h"
#include "../HostVector/HostVector.h"
#include "../HostVector/DeviceVector/DeviceVector.h"
enum SOLVER{ BICGSTAB=1, IGCR=2, GMRES=3, CG=4};
class Xsolver{
public:
    int maxiter;
    int restart;
    double tol;
    double absolute_tol;
    int n;
#ifdef HAVE_MPI
    int nHalo;
#endif
    HostMatrix *hostmtx;
    HostVector *q;
    HostVector *phi;
    double res0;
    double *resvec;
    int usediter;
    int *usediter_pointer;
    Precond *precond;
    Xsolver(){
	maxiter=0;
	restart=0;
	tol=0;
	absolute_tol=0;
	n=0;
	q=NULL;
	phi=NULL;
	res0=0;
	usediter=0;
	resvec=NULL;
    }
    virtual void set_xsolver_absolute_tol(double absolute_tol){}
    virtual void set_xsolver(double tol_in, int maxiter_in, int &usediter_in, double *resvec_in){};
    virtual void set_xsolver(double tol_in, int maxiter_in, int &usediter_in, double *resvec_in,int restart){};
    virtual void create_xsolver(int n_in, HostMatrix *hostmtx_in, double *q_in, double *phi_in){};
    virtual void create_xsolver(int maxiter_in, int restart_in, double tol_in, int n_in, HostMatrix *hostmtx_in, double *resvec=NULL){};
    virtual void create_xsolver(int maxiter_in, int restart_in, double tol_in, int n_in, HostMatrix *hostmtx_in, double *q_in, double *phi_in, double *resvec=NULL){};
    virtual void create_xsolver(int maxiter_in, double tol_in, int n_in, HostMatrix *hostmtx_in, double *q_in, double *phi_in, double *resvec=NULL){};
    virtual void create_xsolver(int maxiter_in, int restart_in, double tol_in, int n_in, double *diag_val_in, HostMatrix *hostmtx_in, double *q_in, double *phi_in, double *resvec=NULL){};
    virtual void create_xsolver(int maxiter_in, double tol_in, int n_in, double *diag_val_in, HostMatrix *hostmtx_in, double *q_in, double *phi_in, double *resvec=NULL){};
    void preconditioner_set(PRECON precon);
    void create_precond(int maxiter_in,int restart_in, double tol_in,int n_in, HostMatrix *hostmtx_in){
	precond->create_precond(maxiter_in,restart_in,tol_in,n_in, hostmtx_in);
    }
    void create_precond(int n_in, HostMatrix *hostmtx_in){
	precond->create_precond(n_in, hostmtx_in);
    }
    void create_precond(int n_in, double *diag_val_in){
	precond->create_precond(n_in, diag_val_in);
    }
    void preconditioner_init(){
	precond->preconditioner_init();
    }
    void preconditioner_free(){
	precond->preconditioner_free();
    }
    virtual void xsolver_init(){};
    virtual void xsolver(){};
    virtual void xsolver_free(){}; 
    virtual ~Xsolver(){};
};
template <typename VectorType>
Xsolver* solver_set(SOLVER solver);
#endif
