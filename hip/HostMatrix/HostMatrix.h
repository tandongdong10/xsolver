#ifndef _HOSTMATRIX_H_
#define _HOSTMATRIX_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../mpicpu.h"
#include "../HostVector/HostVector.h"
#include "../HostVector/DeviceVector/DeviceVector.h"
class HostMatrix;
class DeviceMatrixCSR;
class HostMatrixCSR;
class HostMatrixCSC;
enum MATRIXFORMAT{ MCSR=1, CSR=2, CSC=3, GPUCSR=4, ELL=5, GPUELL=6, WRONGFORMAT=7};
class HostMatrix{
public:
    int m;
    int n;
    int nnz;
#ifdef HAVE_MPI
    int nHalo;
    int *exchange_ptr;
#endif
    HostMatrix(){
	m=0;
	n=0;
        nnz=0;
    }
    HostMatrix(int n,int nnz):
	m(n),
	n(n),
        nnz(nnz){}
    HostMatrix(int m,int n,int nnz):
	m(m),
	n(n),
        nnz(nnz){}
    virtual MATRIXFORMAT getmatrixformat(){return WRONGFORMAT;};
    virtual double* getval(){return nullptr;};
    virtual int* getptr(){return nullptr;};
    virtual int* getidx(){return nullptr;};
    virtual int getonebase(){return 0;};
    virtual void create_matrix(int n_in,double *val_in,int *colptr_in, int *rowidx_in){};
    virtual void create_matrix(int n_in,double *diag_val_in,double *val_in,int *rowptr_in, int *colidx_in){};
    virtual void MallocMatrix(int n_in,int nnz_in){};
#ifdef HAVE_MPI
    virtual void create_matrix(int n_in,int nHalo,double *val_in,int *colptr_in, int *rowidx_in){};
    virtual void create_matrix(int m_in, int nHalo_in,double *diag_val_in,double *val_in,int *rowptr_in, int *colidx_in){};
    virtual void MallocMatrix(int n_in,int nHalo, int nnz_in){};
#else
    virtual void create_matrix(int m_in, int n_in, double *val_in,int *colptr_in, int *rowidx_in){};
    virtual void MallocMatrix(int m_in, int n_in,int nnz_in){};
#endif
    virtual void CopyMatrix(HostMatrix *hostmtx){};
    virtual void based1To0Matrix(){};
    virtual void getdiag(double *a_p){};
    virtual void getdiag(HostVector *a_p_vec){};
    virtual double* getdiagval(){return nullptr;};
    virtual void FreeMatrix(){};
    virtual void CSRTOCSC(HostMatrix *hostmtxcsc){};
    virtual void CSRTOELL(HostMatrix *hostmtxell){};
    virtual void CSCTOCSR(HostMatrix *hostmtxcsr){};
    virtual void MCSRTOCSR(HostMatrix *hostmtxcsr){};
    virtual void MCSRTOCSR(double *diag_val_in,HostMatrix *hostmtxcsr){};
    virtual void update(double *val_new){};
    virtual void update(double *diag_val_new,double *offdiag_val_new){};
    virtual void ToDiagMatrix(HostMatrix *hostmtx){};
    virtual void SpMV(HostVector *x,HostVector *y){};
    virtual void SppMV(HostVector *x,HostVector *y){};
    virtual void bmAx(HostVector *rhs, HostVector *x, HostVector *y){};
    virtual void parilu(HostMatrix *mtxL,HostMatrix *mtxU,int **row_referenced, int sweep=5){};
    virtual void parilu(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU,int **row_referenced, int sweep){};
    virtual void parilut(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU, int sweep){};
    virtual void parilu_csr(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU,int **row_referenced, HostVector *diag_U,int sweep){};
    virtual void parilut_csr(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU, HostVector *diag_U,int sweep){};
    virtual void seqilu0(HostMatrix *hstmtxL,HostMatrix *hstmtxU,double *&diag_val){};
    virtual void seqilu0_mkl(HostMatrix *hstmtxLU){};
    virtual void seqilut_mkl(HostMatrix *hstmtxLU,int maxfil,double ilut_tol){};
    virtual void seqilut(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag,int maxfil,double ilut_tol){};
    virtual void seqic0(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag_val,double smallnum, double droptol){};
    virtual void seqict(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag_val,double smallnum, double droptol, int maxfil){};
    virtual void LUsolve(HostMatrix *mtxU, double *diag, HostVector *x_vec,HostVector *y_vec){};
    virtual void Lsolve(HostVector *x,HostVector *y){};
    virtual void Usolve(HostVector *x,HostVector *y){};
    virtual void Lsolve(DeviceVector *x,DeviceVector *y){};
    virtual void Usolve(DeviceVector *x,DeviceVector *y){};
    virtual void Lsolve_iter(HostVector *x,HostVector *y,HostVector *tmp,int maxiter){};
    virtual void Usolve_iter(HostVector *x,HostVector *y,HostVector *tmp,HostVector *diag_U, int maxiter){};
    virtual ~HostMatrix(){};
    virtual void ToDeviceMatrix(HostMatrix *hstmtx){};
    virtual void SpMV(double *x, double *d_x,double *d_y){};
    virtual void bmAx(double *d_q, double *x, double *d_x,double *d_y){};
};
HostMatrix* matrixform_set(const char* fmt);
#endif
