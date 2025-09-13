#ifndef _HOSTVECTOR_H_
#define _HOSTVECTOR_H_
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "../solver-C/libhead.h"
#include "../mpicpu.h"
#include <string.h>
class HostVector{
public:
    int n;
#ifdef HAVE_MPI
    int nHalo=0;
#endif
    double *val;
    HostVector(){
	n=0;
	val=NULL;
    }
    HostVector(int Num){
	n=Num;
	val=new double[n];
	memset(val,0,n*sizeof(double));
    }
#ifdef HAVE_MPI
    HostVector(int n_in,int nHalo_in){
	n=n_in;
	nHalo=nHalo_in;
	val=new double[n+nHalo];
	memset(val,0,(n+nHalo)*sizeof(double));
    }
    HostVector(int n_in,int nHalo_in,double *v){
	n=n_in;
	nHalo=nHalo_in;
	val=new double[n+nHalo];
	memcpy(val,v,n*sizeof(double));
    }
#endif
     HostVector(const HostVector &hstvec){
        n = hstvec.n; 
#ifdef HAVE_MPI
	nHalo = hstvec.nHalo;
	val=new double[n+nHalo];
#else
	val=new double[n];
#endif
	memcpy(val,hstvec.val,n*sizeof(double));
    }
     HostVector &operator = (const HostVector &hstvec){
	if(this == &hstvec){
	    return *this;
	}
        this->n = hstvec.n; 
#ifdef HAVE_MPI
	this->nHalo = hstvec.nHalo;
	this->val=new double[n+nHalo];
#else
	this->val=new double[n];
#endif
	memcpy(this->val,hstvec.val,n*sizeof(double));
	return *this;
    }
    HostVector(double *v):
	val(v){}
    HostVector(int n_in,double *v){
	n=n_in;
	val=new double[n];
	memcpy(val,v,n*sizeof(double));
    }
    virtual void MallocVector(int n_in);
    virtual void MallocVector(int n_in,double *v);
#ifdef HAVE_MPI
    virtual void MallocVector(int n_in,int nHalo_in);
    virtual void MallocVector(int n_in,int nHalo_in,double *v);
#endif
    virtual void SetVector(int n_in,double *tmp);
    virtual void CopyVector(int n_in,double *tmp);
    virtual void GetVector(double *tmp);
    virtual void UpdateVector(int n_in,double *tmp);
    virtual double vec_dot(HostVector *y);
    virtual void vec_dot2(HostVector *y,HostVector *q,HostVector *z, double *res);
    virtual double vec_norm1();
    virtual void vec_copy(HostVector *x);
    virtual void vec_axpy(double alpha, HostVector *x);
    virtual void vec_scal(double alpha);
    virtual void vec_bicg_kernel1(double omega, double alpha, HostVector *res,HostVector *uk);
    virtual void vec_bicg_kernel2(double gama, HostVector *res,HostVector *uk);
    virtual void vec_bicg_kernel3(double gama, double alpha, HostVector *pk,HostVector *sk);
    virtual void jacobiInit(HostVector *diag, double small=0);
    virtual void jacobiSolve(HostVector *x,HostVector *y);
    virtual void vec_print();
    virtual void FreeVector();
    ~HostVector(){
    }
};
HostVector* set_vector_cpu();
HostVector* set_vector_cpu(int n);
#ifdef HAVE_MPI
HostVector* set_vector_cpu(int n,int nHalo);
#endif
HostVector* NewVector();
HostVector* NewVector(int n);
#ifdef HAVE_MPI
HostVector* NewVector(int n, int nHalo);
#endif
#endif
