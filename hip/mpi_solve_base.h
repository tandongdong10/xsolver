#include "xsolver_head.h"
//#include "Tool/distribute_mtx.h"

#ifdef HAVE_MPI
extern struct topology_c topo_c;
#endif
void solve_base(int n, int *rowPtr, int *colIdx, double *csrVal, double *q, double *phi,int maxIteration, double tol,double &res0, int &usediter)
{
    struct timeval time1, time2;
    int advanced_pam[3];
    advanced_pam[0]=1;
    advanced_pam[1]=0;
    advanced_pam[2]=0;
    xsolver_mat_setup(n,rowPtr,colIdx,csrVal,advanced_pam);
    char ksptype[10]="bicgstab";
    xsolver_ksp_settype(ksptype);
    double *resvec=new double[maxIteration+1];
    memset(resvec,0,(maxIteration+1)*sizeof(double));
    xsolver_ksp_setoption(tol,maxIteration, usediter, resvec, maxIteration);
    //xsolver_ksp_set_absolute_tol(0.5);
    //char pctype[20]="noprecond";
    char pctype[20]="jacobi";
    //char pctype[20]="ict_l";
    xsolver_pc_settype(pctype);
    //xsolver_pc_setict_l(1e-11,5);
    xsolver_pc_setilut(1e-11,20);
    scatter_mpi(q);
    gettimeofday(&time1, NULL); 
    xsolver_solve(q,phi);
    gettimeofday(&time2, NULL); 
    double elapsed_time = (time2.tv_sec - time1.tv_sec) * 1000. +(time2.tv_usec - time1.tv_usec) / 1000.; 
    if(topo_c.myid<3)
    	printf("Elapsed Solve time: %lf(ms)\n", elapsed_time); 
    gather_mpi(phi);
    xsolver_free();//!!!!!!!!!!!
    res0=resvec[0];
    //if(topo_c.myid==0)
   //	printf("X-Solver Iter %d,  error =%lg\n",usediter,resvec[usediter]);
    //cout residual
    if(topo_c.myid==0){
    for(int i=0;i<usediter+1;i++)
        printf("X-Solver Iter %d,  error =%lg\n",i,resvec[i]);
    }
    xsolver_communicator_distroy();
    delete []resvec;
    return;
}
