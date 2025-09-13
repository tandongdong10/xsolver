#ifndef _HOSTMATRIXMCSR_H_
#define _HOSTMATRIXMCSR_H_
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string.h>
#include "HostMatrix.h"
//#include "../Tool/MatrixTrans.h"
extern void permuteMatrix(HostMatrix *hostmtx);
class HostMatrixMCSR:public HostMatrix{
public:
    double *diag_val;
    double *val;
    int *rowptr;
    int *colidx;
    HostMatrixMCSR(){
        diag_val=NULL;
        val=NULL;
        rowptr=NULL;
        colidx=NULL;
    }
    HostMatrixMCSR(int n,int nnz,double *diag_val,double *val,int *rowptr, int *colidx):
	//n(n),
        //nnz(nnz),
        diag_val(diag_val),
        val(val),
        rowptr(rowptr),
        colidx(colidx){}
    MATRIXFORMAT getmatrixformat(){
	return MCSR;
    }
    void getdiag(double *a_p){
	a_p=diag_val;
    }
    double* getdiagval(){
	return diag_val;}
    double* getval(){
	return val;}
    int* getptr(){
	return rowptr;}
    int* getidx(){
	return colidx;}
    void setdiagval(double *diag_val_in){
	memcpy(diag_val,diag_val_in,n*sizeof(double));
    }
    void create_matrix(int n_in,double *diag_val_in,double *val_in,int *rowptr_in, int *colidx_in){
	m=n=n_in;
	nnz=rowptr_in[m]-rowptr_in[0];
	diag_val=diag_val_in;
	val=val_in;
	rowptr=rowptr_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int n_in,int nnz_in){
	m=n=n_in;
	nnz=nnz_in;
	diag_val=(double *)malloc(n*sizeof(double));
    	rowptr=(int *)malloc((m+1)*sizeof(int));
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#ifdef HAVE_MPI
    void create_matrix(int m_in, int nHalo_in,double *diag_val_in,double *val_in,int *rowptr_in, int *colidx_in){
	m=n=m_in;
	nHalo=nHalo_in;
	nnz=rowptr_in[m]-rowptr_in[0];
	diag_val=diag_val_in;
	val=val_in;
	rowptr=rowptr_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int m_in,int nHalo_in,int nnz_in){
	m=n=m_in;
	nHalo=nHalo_in;
	nnz=nnz_in;
	diag_val=(double *)malloc(n*sizeof(double));
    	rowptr=(int *)malloc((m+1)*sizeof(int));
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#else
    void create_matrix(int m_in, int n_in,double *diag_val_in,double *val_in,int *rowptr_in, int *colidx_in){
	m=m_in;
	n=n_in;
	nnz=rowptr_in[m]-rowptr_in[0];
	diag_val=diag_val_in;
	val=val_in;
	rowptr=rowptr_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int m_in,int n_in,int nnz_in){
	m=m_in;
	n=n_in;
	nnz=nnz_in;
	diag_val=(double *)malloc(n*sizeof(double));
    	rowptr=(int *)malloc((m+1)*sizeof(int));
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#endif
    void FreeMatrix(){
	if(rowptr!=NULL) {free(rowptr);rowptr=NULL;}
	if(colidx!=NULL) {free(colidx);colidx=NULL;}
	if(val!=NULL) {free(val);val=NULL;}
	if(diag_val!=NULL) {free(diag_val);diag_val=NULL;}
    }
    void update(double *diag_val_new,double *val_new){
	diag_val=diag_val_new;
	val=val_new;
    }
    void MCSRTOCSR(HostMatrix *hostmtxcsr);
    void MCSRTOCSR(double *diag_val_in,HostMatrix *hostmtxcsr);
    void SpMV(HostVector *x,HostVector *y);
    void bmAx(HostVector *rhs, HostVector *x, HostVector *y);
    ~HostMatrixMCSR(){}
};
HostMatrix* set_matrix_mcsr();
#endif
