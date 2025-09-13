#ifndef _HOSTMATRIXCSC_H_
#define _HOSTMATRIXCSC_H_
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string.h>
#include "HostMatrix.h"
#include "HostMatrixCSR.h"
class HostMatrixCSC:public HostMatrix{
public:
    double *val;
    int *colptr;
    int *rowidx;
    HostMatrixCSC(){
	//n=0;
        //nnz=0;
        val=NULL;
        colptr=NULL;
        rowidx=NULL;
    }
    HostMatrixCSC(int n,double *val,int *colptr, int *rowidx):
	//n(n),
        val(val),
        colptr(colptr),
        rowidx(rowidx){}
    MATRIXFORMAT getmatrixformat(){
	return CSC;
    }
    double* getval(){
	return val;}
    int* getptr(){
	return colptr;}
    int* getidx(){
	return rowidx;}
    void create_matrix(int n_in,double *val_in,int *colptr_in, int *rowidx_in){
	m=n=n_in;
	nnz=colptr_in[n]-colptr_in[0];
	val=val_in;
	colptr=colptr_in;
	rowidx=rowidx_in;
    }
    void create_matrix(int m_in, int n_in,double *val_in,int *colptr_in, int *rowidx_in){
	m=m_in;
	n=n_in;
	nnz=colptr_in[n]-colptr_in[0];
	val=val_in;
	colptr=colptr_in;
	rowidx=rowidx_in;
    }
    void MallocMatrix(int n_in,int nnz_in){
	m=n=n_in;
	nnz=nnz_in;
    	colptr=(int *)malloc((n+1)*sizeof(int));
    	rowidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
    void MallocMatrix(int m_in,int n_in,int nnz_in){
	m=m_in;
	n=n_in;
	nnz=nnz_in;
    	colptr=(int *)malloc((n+1)*sizeof(int));
    	rowidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
    void FreeMatrix(){
	if(colptr!=NULL) {free(colptr);colptr=NULL;}
	if(rowidx!=NULL) {free(rowidx);rowidx=NULL;}
	if(val!=NULL) {free(val);val=NULL;}
    }
    void update(double *val_new){
	val=val_new;
    }
    void getdiag(double *a_p);
    void CSCTOCSR(HostMatrix *hostmtxcsr);
    void SpMV(HostVector *x,HostVector *y);
    void bmAx(HostVector *rhs, HostVector *x, HostVector *y);
    ~HostMatrixCSC(){}
};
HostMatrix* set_matrix_csc();
#endif
