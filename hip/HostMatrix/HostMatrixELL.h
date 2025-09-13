#ifndef _HOSTMATRIXELL_H_
#define _HOSTMATRIXELL_H_
#include <stdio.h>
#include <stdlib.h>
#include "../solver-C/libhead.h"
#include "HostMatrix.h"
#include <string.h>
#include <sys/time.h>
#include <stdio.h>

class HostMatrixELL:public HostMatrix{
public:
#ifdef HAVE_MPI
    int *exchange_ptr;
#endif
    double *val;
    int *colidx;
    HostMatrixELL(){
	//n=0;
        //nnz=0;
        val=NULL;
        colidx=NULL;
    }
    HostMatrixELL(int n,int nnz,double *val, int *colidx):
	//n(n),
        //nnz(nnz),
        val(val),
        colidx(colidx){}
    MATRIXFORMAT getmatrixformat(){
	return ELL;
    }
    double* getval(){
	return val;}
    int* getidx(){
	return colidx;}
    void create_matrix(int n_in, int nnz_in, double *val_in, int *colidx_in){
	m=n=n_in;
	nnz=nnz_in;
	val=val_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int n_in,int nnz_in){
	m=n=n_in;
	nnz=nnz_in;
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#ifdef HAVE_MPI
    void create_matrix(int m_in, int nHalo_in, int nnz_in, double *val_in, int *colidx_in){
	m=n=m_in;
	nHalo=nHalo_in;
	nnz=nnz_in;
	val=val_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int m_in,int nHalo_in,int nnz_in){
	m=n=m_in;
	nHalo=nHalo_in;
	nnz=nnz_in;
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#else
    void create_matrix(int m_in, int n_in, int nnz_in, double *val_in,int *colidx_in){
	m=m_in;
	n=n_in;
	nnz=nnz_in;
	val=val_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int m_in,int n_in,int nnz_in){
	m=m_in;
	n=n_in;
	nnz=nnz_in;
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#endif
    void CopyMatrix(HostMatrix *hostmtx){
	if(hostmtx->getmatrixformat()!=ELL){
	    printf("Wrong!!! copy matrix is not Ell!!!\n");
	    exit(0);
	}
	m=hostmtx->m;
	n=hostmtx->n;
	nnz=hostmtx->nnz;
	memcpy(val,hostmtx->getval(),nnz*sizeof(double));
	memcpy(colidx,hostmtx->getidx(),nnz*sizeof(int));
    }
    void FreeMatrix(){
	if(colidx!=NULL) {free(colidx);colidx=NULL;}
	if(val!=NULL) {free(val);val=NULL;}
    }
    void update(double *val_new){
	val=val_new;
    }
    void getdiag(double *a_p);
    void SpMV(HostVector *x,HostVector *y);
    void bmAx(HostVector *rhs, HostVector *x, HostVector *y);
    ~HostMatrixELL(){}
};
HostMatrix* set_matrix_ell();
#endif
