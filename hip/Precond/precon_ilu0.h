#ifndef _PRECOND_ILU0_H
#define _PRECOND_ILU0_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "preconditioner.h"
#include <sys/time.h>
#include <stdio.h>
#include "../HostMatrix/HostMatrixCSR.h"

class Precond_ilu0:public Precond{
public:
    HostMatrix *hostmtx;
    HostMatrix *hostmtxL;
    HostMatrix *hostmtxU;
    double *diag_val;
    Precond_ilu0(){
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
    ~Precond_ilu0(){};
};
Precond* precond_set_ilu0();
#endif
