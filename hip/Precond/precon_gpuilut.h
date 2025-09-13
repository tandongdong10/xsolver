#ifndef _PRECOND_GPUILUT_H
#define _PRECOND_GPUILUT_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "preconditioner.h"
#include "../HostMatrix/DeviceMatrix/DeviceMatrixCSR.h"
class Precond_gpuilut:public Precond{
public:
    HostMatrix *hostmtx;
    DeviceMatrixCSR *devicemtxL;
    DeviceMatrixCSR *devicemtxU;
    int *row_referenced;
    HostVector *xtmp_vec;
    HostVector *tmp;
    HostVector *diag_U;
    Precond_gpuilut(){
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
    ~Precond_gpuilut(){};
};
Precond* precond_set_gpuilut();
#endif
