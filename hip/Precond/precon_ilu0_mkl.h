#ifndef _PRECOND_ILU0_MKL_H
#define _PRECOND_ILU0_MKL_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "preconditioner.h"
#include "../HostMatrix/HostMatrixCSR.h"
class Precond_ilu0_mkl:public Precond{
public:
    HostMatrix *hostmtx;
    HostMatrix *hostmtxLU;
    HostVector *xtmp_vec;
    double *bilu0;
    Precond_ilu0_mkl(){
	n_p=0;
	bilu0=NULL;
 	hostmtx=NULL;
    }
    void create_precond(int n_in, HostMatrix *hostmtx_in){
	n_p=n_in;
	hostmtx=hostmtx_in;
    }
    void preconditioner_init();
    void preconditioner(HostVector *x_vec,HostVector *y_vec);
    void preconditioner_free();
    ~Precond_ilu0_mkl(){};
};
Precond* precond_set_ilu0_mkl();
#endif
