#ifndef _PRECOND_ILUP_MKL_H
#define _PRECOND_ILUP_MKL_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "preconditioner.h"
#include "../HostMatrix/HostMatrixCSR.h"
class Precond_ilup_mkl:public Precond{
public:
    HostMatrix *hostmtx;
    HostMatrix *hostmtxLU;
    int maxfil;
    HostVector *xtmp_vec;
    double ilut_tol;
    Precond_ilup_mkl(){
	n_p=0;
 	hostmtx=NULL;
	maxfil=20;
	ilut_tol=1e-5;
    }
    void create_precond(int n_in, HostMatrix *hostmtx_in){
	n_p=n_in;
	hostmtx=hostmtx_in;
    }
    void set_ilut(double tol_in, int maxfil_in){
	ilut_tol=tol_in;
	maxfil=maxfil_in;
    }
    void preconditioner_init();
    void preconditioner(HostVector *x,HostVector *y);
    void preconditioner_free();
    ~Precond_ilup_mkl(){}
};
Precond* precond_set_ilup_mkl();
#endif
