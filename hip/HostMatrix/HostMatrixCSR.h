#ifndef _HOSTMATRIXCSR_H_
#define _HOSTMATRIXCSR_H_
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string.h>
#include "HostMatrix.h"
#include "HostMatrixCSC.h"
#include "../Precond/ilut_diag_in_csr.h"
#include "../Precond/ic0.h"
#include "../Precond/ict_l.h"
class HostMatrixCSR:public HostMatrix{
public:
#ifdef HAVE_MPI
    int *exchange_ptr;
#endif
    double *val;
    int *rowptr;
    int *colidx;
    HostMatrixCSR(){
	//n=0;
        //nnz=0;
        val=NULL;
        rowptr=NULL;
        colidx=NULL;
#ifdef HAVE_MPI
	exchange_ptr=NULL;
#endif
    }
    HostMatrixCSR(int n,int nnz,double *val,int *rowptr, int *colidx):
	//n(n),
        //nnz(nnz),
        val(val),
        rowptr(rowptr),
        colidx(colidx){}
    MATRIXFORMAT getmatrixformat(){
	return CSR;
    }
    double* getval(){
	return val;}
    int* getptr(){
	return rowptr;}
    int* getidx(){
	return colidx;}
    int getonebase(){
	return rowptr[0];}
    void create_matrix(int n_in,double *val_in,int *rowptr_in, int *colidx_in){
	m=n=n_in;
	nnz=rowptr_in[m]-rowptr_in[0];
	val=val_in;
	rowptr=rowptr_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int n_in,int nnz_in){
	m=n=n_in;
	nnz=nnz_in;
    	rowptr=(int *)malloc((m+1)*sizeof(int));
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#ifdef HAVE_MPI
    void create_matrix(int m_in, int nHalo_in,double *val_in,int *rowptr_in, int *colidx_in){
	m=n=m_in;
	nHalo=nHalo_in;
	nnz=rowptr_in[m]-rowptr_in[0];
	val=val_in;
	rowptr=rowptr_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int m_in,int nHalo_in,int nnz_in){
	m=n=m_in;
	nHalo=nHalo_in;
	nnz=nnz_in;
    	rowptr=(int *)malloc((m+1)*sizeof(int));
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#else
    void create_matrix(int m_in, int n_in,double *val_in,int *rowptr_in, int *colidx_in){
	m=m_in;
	n=n_in;
	nnz=rowptr_in[m]-rowptr_in[0];
	val=val_in;
	rowptr=rowptr_in;
	colidx=colidx_in;
    }
    void MallocMatrix(int m_in,int n_in,int nnz_in){
	m=m_in;
	n=n_in;
	nnz=nnz_in;
    	rowptr=(int *)malloc((m+1)*sizeof(int));
    	colidx=(int *)malloc(nnz*sizeof(int));
    	val=(double *)malloc(nnz*sizeof(double));
    }
#endif
    void CopyMatrix(HostMatrix *hostmtx){
	if(hostmtx->getmatrixformat()!=CSR){
	    printf("Wrong!!! copy matrix is not csr!!!\n");
	    exit(0);
	}
	m=hostmtx->m;
	n=hostmtx->n;
	nnz=hostmtx->nnz;
	memcpy(val,hostmtx->getval(),nnz*sizeof(double));
	memcpy(rowptr,hostmtx->getptr(),(m+1)*sizeof(int));
	memcpy(colidx,hostmtx->getidx(),nnz*sizeof(int));
    }
    void FreeMatrix(){
	if(rowptr!=NULL) {free(rowptr);rowptr=NULL;}
	if(colidx!=NULL) {free(colidx);colidx=NULL;}
	if(val!=NULL) {free(val);val=NULL;}
    }
    void update(double *val_new){
	val=val_new;
    }
    void CSRTOCSC(HostMatrix *hostmtxcsc);
    void CSRTOELL(HostMatrix *hostmtxell);
    void update(double *diag_val_new,double *val_new);
    void getdiag(HostVector *a_p);
    void getdiag(double *a_p);
    void ToDiagMatrix(HostMatrix *hostmtx);
    void SpMV(HostVector *x,HostVector *y);
    void SpMM(HostMatrixCSR *B,HostMatrixCSR *C);
    void bmAx(HostVector *rhs, HostVector *x, HostVector *y);
    void seqilu0(HostMatrix *hstmtxL,HostMatrix *hstmtxU,double *&diag_val);
    void seqilu0_mkl(HostMatrix *hstmtxLU);
    void seqilut_mkl(HostMatrix *hstmtxLU,int maxfil,double ilut_tol);
    void seqilut(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag,int maxfil,double ilut_tol);
    void seqic0(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag_val,double smallnum, double droptol);
    void seqict(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag_val,double smallnum, double droptol, int maxfil);
    void LUsolve(HostMatrix *mtxU, double *diag, HostVector *x_vec,HostVector *y_vec);
    void Lsolve(HostVector *x,HostVector *y);
    void Usolve(HostVector *x,HostVector *y);
    ~HostMatrixCSR(){}
};
HostMatrix* set_matrix_csr();
#endif
