#ifndef _PRECOND_JACOBI_H
#define _PRECOND_JACOBI_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "preconditioner.h"
#include "../HostMatrix/HostMatrixCSR.h"
class Precond_jacobi:public Precond{
public:
    double small;
    HostVector *diag;
    HostVector *diag_val;
    Precond_jacobi(){
	n_p=0;
	diag_val=NULL;
	small=0;
    }
    void create_precond(int n_in, HostMatrix *hstmtx);
    void create_precond(int n_in, double *diag_val_in);
    void preconditioner_init();
    void preconditioner_update(HostMatrix *hstmtx);
    void preconditioner_update(double *diag_val_in);
    void preconditioner(HostVector *x,HostVector *y);
    void preconditioner_free();
    ~Precond_jacobi(){
	diag_val->FreeVector();
    };
};
Precond* precond_set_jacobi();
#endif
