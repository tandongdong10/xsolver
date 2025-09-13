#ifndef _PRECOND_GPUILU0_H
#define _PRECOND_GPUILU0_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "preconditioner.h"
#include "gpuparilu.h"
#include "../HostMatrix/DeviceMatrix/DeviceMatrixCSR.h"
class Precond_gpuilu0:public Precond{
public:
    HostMatrix *hostmtx;
    DeviceMatrixCSR *devicemtxL;
    DeviceMatrixCSR *devicemtxU;
    int *row_referenced;
    HostVector *diag_U;
    HostVector *xtmp_vec;
    HostVector *tmp;
    Precond_gpuilu0(){
	n_p=0;
 	hostmtx=NULL;
    }
    void create_precond(int n_in, HostMatrix *hostmtx_in){
	n_p=n_in;
	hostmtx=hostmtx_in;
    }
    void preconditioner_init();
    void preconditioner(HostVector *x_vec,HostVector *y_vec);
    void preconditioner_free();
    ~Precond_gpuilu0(){};
};
Precond* precond_set_gpuilu0();
#endif
