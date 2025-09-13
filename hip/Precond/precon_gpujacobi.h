#ifndef _PRECOND_GPUJACOBI_H
#define _PRECOND_GPUJACOBI_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "preconditioner.h"
#include "hip/hip_runtime.h"
#include <rocblas.h>
#include <sys/time.h>
class Precond_gpujacobi:public Precond{
public:
    double small;
    DeviceVector *diag;
    DeviceVector *diag_val;
    Precond_gpujacobi(){
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
    ~Precond_gpujacobi(){}
};
Precond* precond_set_gpujacobi();
#endif
