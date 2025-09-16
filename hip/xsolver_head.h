#include "mpicpu.h"
extern void xsolver_mat_setup(int nInterior, int *rowptr, int *colidx, double *val, int *advanced_pam);
extern void xsolver_ksp_settype(char *fmt);
extern void xsolver_ksp_set_absolute_tol(double absolute_tol);
extern void xsolver_ksp_setoption(double tol, int maxiter, int &usediter, double *resvec, int restart=0);
extern void xsolver_pc_settype(char *fmt);
extern void xsolver_pc_setilut(double tol, int max_fill);
extern void xsolver_solve(double *rhs, double *sol);
extern void xsolver_free();
#ifdef HAVE_MPI
extern void scatter_mpi(double *q);
extern void gather_mpi(double *phi);
#endif
