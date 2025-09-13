#ifndef _XSOLVER_CG_H
#define _XSOLVER_CG_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "xsolver.h"
template <typename VectorType>
class Xsolver_cg:public Xsolver{
public:
    Xsolver_cg(){
	maxiter=0;
	tol=0;
	absolute_tol=0;
	n=0;
#ifdef HAVE_MPI
	nHalo=0;
#endif
	q=NULL;
	phi=NULL;
	phi_old=NULL;
	res0=0;
	resvec=NULL;
	usediter=0;
    }
    void set_xsolver_absolute_tol(double absolute_tol_in){
	absolute_tol=absolute_tol_in;
    }
    void set_xsolver(double tol_in, int maxiter_in, int &usediter_in, double *resvec_in){
	tol=tol_in;
	maxiter=maxiter_in;
	usediter_pointer=&usediter_in;
	resvec=resvec_in;
    }
    void create_xsolver(int n_in, HostMatrix *hostmtx_in, double *q_in, double *phi_in){
        n=n_in;
	hostmtx=hostmtx_in;
#ifdef HAVE_MPI
	nHalo=hostmtx->nHalo;
#endif
        q= new VectorType();
	phi_old=phi_in;
	phi= new VectorType();
#ifdef HAVE_MPI
	phi->MallocVector(n,nHalo,phi_in);
#else
	phi->MallocVector(n,phi_in);
#endif
	q->MallocVector(n,q_in);
    }
    void create_xsolver(int maxiter_in, double tol_in, int n_in, HostMatrix *hostmtx_in, double *q_in, double *phi_in, double *resvec_in=NULL){
	maxiter=maxiter_in;
	tol=tol_in;
        n=n_in;
	hostmtx=hostmtx_in;
        q= new VectorType();
	phi_old=phi_in;
	phi= new VectorType();
#ifdef HAVE_MPI
	nHalo=hostmtx->nHalo;
	phi->MallocVector(n,nHalo,phi_in);
#else 
	phi->MallocVector(n,phi_in);
#endif
	q->MallocVector(n,q_in);
	resvec=resvec_in;
    }
    void xsolver_init();
    void xsolver();
    void xsolver_free(); 
    ~Xsolver_cg(){
    };
private:
    VectorType *pk;
    VectorType *res;
    VectorType *zk;
    VectorType *Apk;
    VectorType *qk;
    VectorType *phi;
    double small = 0;
    double *phi_old=NULL;
};
template <typename VectorType>
Xsolver* solver_set_cg();
#endif
