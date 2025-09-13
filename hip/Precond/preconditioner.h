#ifndef _PRECONDITIONER_H
#define _PRECONDITIONER_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "../HostMatrix/HostMatrix.h"
#include "../HostVector/HostVector.h"
#include "../HostVector/DeviceVector/DeviceVector.h"
//#include "HostMatrixCSC.h"
//#include "HostMatrixCSR.h"
enum PRECON{ JACOBI=1, ILU0=2, ILUP=3, ILU0_MKL=7, ILUP_MKL=8, IC0=9, ICT=10};
class Precond{
public:
    int n_p;
    Precond();
    virtual void set_ilut(double tol_in, int maxfil_in){};
    virtual void set_ict(double tol_in, int maxfil_in){};
    virtual void create_precond(int maxiter_in,int restart, double tol_in,int n_in, HostMatrix *hostmtx_in){};
    //virtual void create_precond(int n_in, HostMatrix *hostmtx_in,int maxiter_in=200, int restart_in=5,double tol_in=1e-4){};
    //virtual void create_precond(int Num_in, HostMatrixCSR **hostmtx_in,HostMatrixtotal *hostmtx_total_in){}
    virtual void create_precond(int n_in, HostMatrix *hostmtx_in){};
    virtual void create_precond(int n_in, double *diag_val_in){};
    virtual void preconditioner_init(){};
    virtual void preconditioner_update(double *diag_val_in){};
    virtual void preconditioner(HostVector *x,HostVector *y){};
    virtual void preconditioner_free(){};
    virtual ~Precond(){};
};
#endif
