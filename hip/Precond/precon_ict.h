#ifndef _PRECOND_ICT_H
#define _PRECOND_ICT_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "preconditioner.h"
#include "../HostMatrix/HostMatrixCSR.h"
class Precond_ict:public Precond{
public:
    HostMatrix *hostmtx;
    HostMatrix *hostmtxL;
    HostMatrix *hostmtxU;
    double smallnum=1e-8;
    double droptol=1e-8;
    int maxfil=100;
    double *diag;
    
    Precond_ict(){
	n_p=0;
 	hostmtx=NULL;
    }
    void create_precond(int n_in, HostMatrix *hostmtx_in){
	n_p=n_in;
	hostmtx=hostmtx_in;
    }
    void set_ict(double tol_in, int maxfil_in){
	smallnum=droptol=tol_in;
	maxfil=maxfil_in;
    }
    void preconditioner_init();
    void preconditioner(HostVector *x_vec,HostVector *y_vec);
    void preconditioner_free();
    ~Precond_ict(){}
};
Precond* precond_set_ict();
#endif
