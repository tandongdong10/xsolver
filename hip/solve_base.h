#include "hip/hip_runtime.h"
#include "xsolver_c.h"

void solve_base(int n, int *rowPtr, int *colIdx, double *csrVal, double *q, double *phi,int maxIteration, double tol,double &res0, int &usediter)
{
    int advanced_pam[3];
    advanced_pam[0]=0;
    advanced_pam[1]=0;
    advanced_pam[2]=1;
    xsolver_mat_setup(n,rowPtr,colIdx,csrVal,advanced_pam);
    xsolver_ksp_settype("bicgstab");
    double *resvec=new double[maxIteration+1];
    memset(resvec,0,(maxIteration+1)*sizeof(double));
    xsolver_ksp_setoption(tol,maxIteration, usediter, resvec, maxIteration);
    xsolver_pc_settype("ilut");
    xsolver_pc_setilut(1e-4,5);
    xsolver_pc_setict(1e-4,5);
    xsolver_solve(q,phi);
    xsolver_free();
    res0=resvec[0];
    for(int i=0;(i<maxIteration+1)&&resvec[i]!=0;i++)
	printf("X-Solver Iter %d,  error =%lf\n",i,resvec[i]);
    delete []resvec;
}
