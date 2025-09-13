#ifndef _XSOLVER_C_H_
#define _XSOLVER_C_H_
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Tool/MatrixTrans.h"
#include "mpicpu.h"
#include "hip/hip_runtime.h"
#include "HostVector/DeviceVector/DeviceVector.h"
#include "HostMatrix/HostMatrixCSC.h"
#include "HostMatrix/HostMatrixCSR.h"
#include "HostMatrix/HostMatrixELL.h"
#include "HostMatrix/HostMatrixMCSR.h"
#include "HostMatrix/DeviceMatrix/DeviceMatrixCSR.h"
#include "HostMatrix/DeviceMatrix/DeviceMatrixELL.h"
#include "Solver/xsolver_bicgstab.h"
#include "Solver/xsolver_cg.h"
#include "Solver/xsolver_igcr.h"
#include "Precond/precon_jacobi.h"
#include "Precond/precon_gpujacobi.h"
#include "Precond/precon_ilu0.h"
#include "Precond/precon_ilu0_mkl.h"
#include "Precond/precon_gpuilu0.h"
#include "Precond/precon_gpuilut.h"
#include "Precond/precon_ilup.h"
#include "Precond/precon_ilup_mkl.h"
#include "Precond/precon_ic0.h"
#include "Precond/precon_ict.h"
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#ifdef HAVE_MPI
#include "Tool/distribute_mtx.h"
#endif

//template <typename T>
/*extern "C" */void xsolver_mat_setup(int nInterior, int *rowptr, int *colidx, double *val, int *advanced_pam);
/*extern "C" */void xsolver_ksp_settype(char *fmt);
/*extern "C" */void xsolver_ksp_set_absolute_tol(double absolute_tol);
/*extern "C" */void xsolver_ksp_setoption(double tol, int maxiter, int &usediter, double *resvec, int restart=0);
/*extern "C" */void xsolver_pc_settype(char *fmt);
/*extern "C" */void xsolver_pc_setilut(double tol, int max_fill);
/*extern "C" */void xsolver_pc_setict(double tol, int max_fill);
/*extern "C" */void xsolver_solve(double *rhs, double *sol);
/*extern "C" */void xsolver_free();
#ifdef HAVE_MPI
/*extern "C" */void scatter_mpi(double *q);
/*extern "C" */void gather_mpi(double *phi);
#endif
#endif
