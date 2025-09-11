#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <string.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <sys/time.h>
#include "device_launch_parameters.h"
#ifdef HAVE_MPI
extern struct topology_c topo_c;
#endif
#include "mpicpu.h"
#include "HostMatrix.h"
#include "DeviceMatrixCSR.h"
#include "DeviceMatrixELL.h"

void applyplanerotationc(double &dx, double &dy, double csx, double snx);
void generateplanerotationc(double dx, double dy, double &csx, double &snx);



//extern void  communicator_sum(double &value);
//extern void communicator_sum(double *value,int n);
//extern void  communicator_sum(float &value);
//extern void communicator_sum(float *value,int n);
//extern void  communicator_sum(int &value);
//extern void communicator_sum(int *value,int n);
//extern cublasHandle_t handle;
//extern HostMatrix hostmtx;
//extern DeviceMatrix *devicemtx;
//extern PRECON precon;
//__global__ static void gputogpu (const int d_nIntCells, const double *d_a, double *d_b)
//{
//    int icell = blockIdx.x*blockDim.x + threadIdx.x;
//    if (icell < d_nIntCells) {
//    	d_b[icell] = d_a[icell];
//    }
//}


int *d_exchange_ptr;
double *d_phi;
double *d_res;
double *d_qk;
double *d_pk;
double *d_Apk;
double *d_reso;
double *d_uk;
double *d_vk;
double *d_zk;
double *d_sk;
double *d_tk;
double *d_vkm;
double *d_tkm;
double *d_resm;
double *d_vko;
double *d_ukr;
double *d_ckr;

double *d_V;
double *d_Z;
double *d_q;
double *d_w;

void gpu_cg_malloc(const int &nIntCells, const int &nGhstCells)
{
    cudaMalloc((void **)&d_phi, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_res, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_qk, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_zk, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_pk, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_Apk, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_q, sizeof(double) * nIntCells);
}
void gpu_bicgstab_malloc(const int &nIntCells, const int &nGhstCells)
{
    cudaMalloc((void **)&d_phi, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_res, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_qk, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_pk, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_reso, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_uk, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_vk, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_zk, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_sk, sizeof(double) * nIntCells);

    cudaMalloc((void **)&d_tk, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_vkm, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_tkm, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_resm, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_vko, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_q, sizeof(double) * nIntCells);
}
void gpu_gcr_malloc(const int &nIntCells, const int &nGhstCells, const int &restart)
{
    cudaMalloc((void **)&d_phi, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_res, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_pk, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_ukr, sizeof(double) * restart*(nIntCells + nGhstCells));
    cudaMalloc((void **)&d_ckr, sizeof(double) * restart*nIntCells);
    cudaMalloc((void **)&d_q, sizeof(double) * nIntCells);
}

void gpu_gmres_malloc(const int &nIntCells, const int &nGhstCells, const int &restart)
{
    cudaMalloc((void **)&d_phi, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_res, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_pk, sizeof(double) * (nIntCells + nGhstCells));
    cudaMalloc((void **)&d_V, sizeof(double) * (restart+1)*(nIntCells + nGhstCells));
    cudaMalloc((void **)&d_Z, sizeof(double) * (restart+1)*(nIntCells + nGhstCells));
    cudaMalloc((void **)&d_w, sizeof(double) * nIntCells);
    cudaMalloc((void **)&d_q, sizeof(double) * nIntCells);
}

void gpu_cg_free()
{
    cudaFree(d_phi);
    cudaFree(d_res);
    cudaFree(d_qk);
    cudaFree(d_zk);
    cudaFree(d_pk);
    cudaFree(d_Apk);
    cudaFree(d_q);
}
void gpu_bicgstab_free()
{
    cudaFree(d_phi);
    cudaFree(d_res);
    cudaFree(d_pk);
    cudaFree(d_qk);
    cudaFree(d_reso);
    cudaFree(d_uk);
    cudaFree(d_vk);
    cudaFree(d_zk);
    cudaFree(d_sk);
    cudaFree(d_tk);
    cudaFree(d_vkm);
    cudaFree(d_tkm);
    cudaFree(d_resm);
    cudaFree(d_vko);
    cudaFree(d_q);
}
void gpu_gcr_free()
{
    cudaFree(d_phi);
    cudaFree(d_res);
    cudaFree(d_pk);
    cudaFree(d_ckr);
    cudaFree(d_ukr);
    cudaFree(d_q);
}
void gpu_gmres_free()
{
    cudaFree(d_phi);
    cudaFree(d_res);
    cudaFree(d_pk);
    cudaFree(d_V);
    cudaFree(d_Z);
    cudaFree(d_q);
    cudaFree(d_w);
}

extern "C" void gpu_cg_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{
//left preconditioned cg method with Jacobi preconditioner
     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_cg_malloc(nIntCells, nGhstCells);
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;

     double *pk = new double[nIntCells+nGhstCells]; 
     double *res = new double[nIntCells];
     //double *diag = new double[nIntCells];
     double *Apk = new double[nIntCells];
     double *qk = new double[nIntCells];
     double sigma, alpha, minalpha, taoo, tao, resl, rsm, beta;
     int iiter;
     //double small = 0, one = 1.0, minone = -1.0;
     double small = 1e-20, one = 1.0, minone = -1.0;
//     for(icell=0;icell<nIntCells;icell++)
//{
//       tmp = a_p[icell] + small;
//       diag[icell] = one/tmp;
//}
     //jacobiInitCUDA<<<dim3(d_nblock), dim3(d_nthread),0,0>>>(nIntCells, d_a_p, d_diag, small);
     if(sol_init==NULL)
	sol_init=sol;
     cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_res, res, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     //cudaMemcpy(d_diag, diag, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_pk, pk, sizeof(double) * (nIntCells+nGhstCells), cudaMemcpyHostToDevice);
     cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
     preconditioner (d_res, d_res);
     //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_res, d_res);
     //applying the preconditioner
     /*for(icell=0;icell<nIntCells;icell++)
     {
        res[icell] = diag[icell]*res[icell];
     }
      
     res0 = cblas_ddot(nIntCells,res,intone,res,intone);*/
     cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &res0);
     communicator_sum(res0);
     //cblas_dcopy(nIntCells,res,intone,pk,intone); 
     cublasDcopy(handle,nIntCells,d_res,1,d_pk,1); 
     taoo = res0;
     res0 = sqrt(res0);


    for(iiter=0;iiter<maxiter;iiter++)
{
    usediter = iiter;    


    devicemtx->SpMV(pk, d_pk,d_Apk);
    //matmultvecc(nIntCells,nGhstCells, ell_val, ell_idx, d_a_p, pk, d_pk,d_Apk);
    preconditioner (d_Apk, d_qk);
    //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_Apk, d_qk);
    cublasDdot(handle, nIntCells, d_pk, 1, d_qk, 1, &sigma);
     communicator_sum(sigma);
     alpha = taoo/(sigma+small);
     minalpha = minone*alpha;
       cublasDaxpy(handle, nIntCells, &alpha, d_pk, 1, d_phi, 1);
       cublasDaxpy(handle, nIntCells, &minalpha, d_qk, 1, d_res, 1);
     //check convergence 
    cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &resl);
     communicator_sum(resl);
     tao = resl;
     resl = sqrt(resl);
     rsm = resl/(res0+small);
     //resvec[iiter]=rsm;
     //printf("cg iter=%d, rsm= % 4.5f\n", iiter, rsm);
     if(resvec!=NULL)
	 resvec[iiter]=rsm;
     if(rsm<tol)
     {
       usediter = iiter;
       cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
       gpu_cg_free();
       delete []pk; 
       delete []res; 
       delete []Apk; 
       delete []qk; 
       return;
     }
     
     beta = tao/(taoo+small);
     taoo = tao;

     cublasDscal(handle, nIntCells, &beta, d_pk, 1);
     cublasDaxpy(handle, nIntCells, &one, d_res, 1, d_pk, 1);


} 
    cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    gpu_cg_free();
    delete []pk; 
    delete []res; 
    delete []Apk; 
    delete []qk; 
}     



extern "C" void gpu_icg_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{

//left preconditioned improved cg method with Jacobi preconditioner
     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_cg_malloc(nIntCells, nGhstCells);
     //double *a_p=hostmtx.diag_val;
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;
     double *pk = new double[nIntCells+nGhstCells]; 
     
     double *res = new double[nIntCells];
     double *Apk = new double[nIntCells];
     double *qk = new double[nIntCells];
     double *tmp_v = new double[3];
     
     double sigma, alpha, minalpha, taoo, tao, resl, rsm, beta, rho, theta;
     int iiter;
     
     //double small = 0, one = 1.0, minone = -1.0;
     double small = 1e-20, one = 1.0, minone = -1.0;


//     for(icell=0;icell<nIntCells;icell++)
//{
//       tmp = a_p[icell] + small;
//       diag[icell] = one/tmp;
//}
     //jacobiInitCUDA<<<dim3(d_nblock), dim3(d_nthread),0,0>>>(nIntCells, d_a_p, d_diag, small);
     if(sol_init==NULL)
	sol_init=sol;

    cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_diag, diag, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pk, pk, sizeof(double) * (nIntCells+nGhstCells), cudaMemcpyHostToDevice);
    devicemtx->bmAx(d_q, sol, d_phi,d_res);
    preconditioner (d_res, d_res);
    //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_res, d_res);
     //applying the preconditioner
     /*for(icell=0;icell<nIntCells;icell++)
     {
        res[icell] = diag[icell]*res[icell];
     }
     
     res0 = cblas_ddot(nIntCells,res,intone,res,intone);*/
    cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &res0);
     communicator_sum(res0);
     //cblas_dcopy(nIntCells,res,intone,pk,intone); 
     cublasDcopy(handle,nIntCells,d_res,1,d_pk,1); 
     taoo = res0;
     res0 = sqrt(res0);


    for(iiter=0;iiter<maxiter;iiter++)
{
    usediter = iiter;    
    devicemtx->SpMV(pk, d_pk,d_Apk);
    //matmultvecc(nIntCells,nGhstCells, ell_val, ell_idx, d_a_p, pk, d_pk,d_Apk);
    preconditioner(d_Apk, d_qk);
    cublasDdot(handle, nIntCells, d_pk, 1, d_qk, 1, tmp_v);
    cublasDdot(handle, nIntCells, d_res, 1, d_qk, 1, tmp_v+1);
    cublasDdot(handle, nIntCells, d_qk, 1, d_qk, 1, tmp_v+2);
     communicator_sum(tmp_v,3);
     sigma=tmp_v[0];
     rho=tmp_v[1];
     theta=tmp_v[2];
     alpha = taoo/(sigma+small);
     minalpha = minone*alpha;
       cublasDaxpy(handle, nIntCells, &alpha, d_pk, 1, d_phi, 1);
       cublasDaxpy(handle, nIntCells, &minalpha, d_qk, 1, d_res, 1);
     //check convergence 
    cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &resl);
     communicator_sum(resl);
     tao = resl;
     resl = sqrt(resl);
     rsm = resl/(res0+small);
     if(resvec!=NULL)
	 resvec[iiter]=rsm;
     if(rsm<tol)
     {
       usediter = iiter;
       cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
       gpu_cg_free();
       delete []pk; 
       delete []res; 
       delete []Apk; 
       delete []qk; 
       delete []tmp_v; 
       return;
     }
     
     tao  = taoo - 2.0*alpha*rho + alpha*alpha*theta;
     beta = tao/(taoo+small);
     taoo = tao;

     cublasDscal(handle, nIntCells, &beta, d_pk, 1);
     cublasDaxpy(handle, nIntCells, &one, d_res, 1, d_pk, 1);

} 
    cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    gpu_cg_free();
    delete []pk; 
    delete []res; 
    delete []Apk; 
    delete []qk; 
    delete []tmp_v; 
}     

extern "C" void gpu_cgm_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{
//left preconditioned cg method with Jacobi preconditioner
     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_cg_malloc(nIntCells, nGhstCells);
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;

     double *pk = new double[nIntCells+nGhstCells]; 
     double *res = new double[nIntCells];
     double *zk = new double[nIntCells];
     double *Apk = new double[nIntCells];
     double *qk = new double[nIntCells];
     double sigma, alpha, minalpha, taoo, tao, resl, rsm, beta;
     int iiter;
     //double small = 0, one = 1.0, minone = -1.0;
     double small = 1e-20, one = 1.0, minone = -1.0;
//     for(icell=0;icell<nIntCells;icell++)
//{
//       tmp = a_p[icell] + small;
//       diag[icell] = one/tmp;
//}
     //jacobiInitCUDA<<<dim3(d_nblock), dim3(d_nthread),0,0>>>(nIntCells, d_a_p, d_diag, small);
     if(sol_init==NULL)
	sol_init=sol;
     cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_res, res, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     //cudaMemcpy(d_diag, diag, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_pk, pk, sizeof(double) * (nIntCells+nGhstCells), cudaMemcpyHostToDevice);
     cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
     preconditioner (d_res, d_zk);
     //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_res, d_res);
     //applying the preconditioner
     /*for(icell=0;icell<nIntCells;icell++)
     {
        res[icell] = diag[icell]*res[icell];
     }
      
     res0 = cblas_ddot(nIntCells,res,intone,res,intone);*/
     cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &res0);
     cublasDdot(handle, nIntCells, d_res, 1, d_zk, 1, &tao);
     communicator_sum(res0);
     communicator_sum(tao);
     //cblas_dcopy(nIntCells,res,intone,pk,intone); 
     cublasDcopy(handle,nIntCells,d_zk,1,d_pk,1); 
     res0 = sqrt(res0);


    for(iiter=0;iiter<maxiter;iiter++)
{
    usediter = iiter;    

    taoo = tao;

    devicemtx->SpMV(pk, d_pk,d_Apk);
    //matmultvecc(nIntCells,nGhstCells, ell_val, ell_idx, d_a_p, pk, d_pk,d_Apk);
    //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_Apk, d_qk);
    cublasDdot(handle, nIntCells, d_pk, 1, d_Apk, 1, &sigma);
     communicator_sum(sigma);
     alpha = tao/(sigma+small);
     minalpha = minone*alpha;
       cublasDaxpy(handle, nIntCells, &alpha, d_pk, 1, d_phi, 1);
       cublasDaxpy(handle, nIntCells, &minalpha, d_Apk, 1, d_res, 1);
     //check convergence 
    preconditioner (d_res, d_zk);
    cublasDdot(handle, nIntCells, d_res, 1, d_zk, 1, &tao);
     communicator_sum(tao);
     
     beta = tao/(taoo+small);

     cublasDscal(handle, nIntCells, &beta, d_pk, 1);
     cublasDaxpy(handle, nIntCells, &one, d_zk, 1, d_pk, 1);
     cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &resl);
     communicator_sum(resl);
     resl = sqrt(resl);
     rsm = resl/(res0+small);
     //resvec[iiter]=rsm;
     if(resvec!=NULL)
	 resvec[iiter]=rsm;
     //printf("cg iter=%d, rsm= % 4.5f\n", iiter, rsm);
     if(rsm<tol)
     {
       usediter = iiter;
       cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
       gpu_cg_free();
       delete []pk; 
       delete []res; 
       delete []Apk; 
       delete []qk; 
       delete []zk; 
       return;
     }
} 
    cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    gpu_cg_free();
    delete []pk; 
    delete []res; 
    delete []Apk; 
    delete []qk; 
    delete []zk; 
}
extern "C" void gpu_icgm_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{

//left preconditioned improved cg method with Jacobi preconditioner
     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_cg_malloc(nIntCells, nGhstCells);
     //double *a_p=hostmtx.diag_val;
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;
     double *pk = new double[nIntCells+nGhstCells]; 
     
     double *res = new double[nIntCells];
     double *Apk = new double[nIntCells];
     double *tmp_v = new double[5];
     
     double sigma, alpha, minalpha, tauo, tau, resl, rsm, beta, rho, theta, fai;
     int iiter;
     
     //double small = 0, one = 1.0, minone = -1.0;
     double small = 1e-20, one = 1.0, minone = -1.0;

     if(sol_init==NULL)
	sol_init=sol;

    cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_diag, diag, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pk, pk, sizeof(double) * (nIntCells+nGhstCells), cudaMemcpyHostToDevice);
    devicemtx->bmAx(d_q, sol, d_phi,d_res);
    preconditioner (d_res, d_zk);
    //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_res, d_res);
     //applying the preconditioner
     /*for(icell=0;icell<nIntCells;icell++)
     {
        res[icell] = diag[icell]*res[icell];
     }
     
     res0 = cblas_ddot(nIntCells,res,intone,res,intone);*/
    cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &res0);
    cublasDdot(handle, nIntCells, d_res, 1, d_zk, 1, &tau);
     communicator_sum(res0);
     communicator_sum(tau);
     //cblas_dcopy(nIntCells,res,intone,pk,intone); 
     cublasDcopy(handle,nIntCells,d_zk,1,d_pk,1); 
     res0 = sqrt(res0);


    for(iiter=0;iiter<maxiter;iiter++)
{
    usediter = iiter;    
    devicemtx->SpMV(pk, d_pk,d_Apk);
    //matmultvecc(nIntCells,nGhstCells, ell_val, ell_idx, d_a_p, pk, d_pk,d_Apk);
    preconditioner(d_Apk, d_qk);
    cublasDdot(handle, nIntCells, d_pk, 1, d_Apk, 1, tmp_v);
    cublasDdot(handle, nIntCells, d_res, 1, d_qk, 1, tmp_v+1);
    cublasDdot(handle, nIntCells, d_Apk, 1, d_zk, 1, tmp_v+2);
    cublasDdot(handle, nIntCells, d_qk, 1, d_Apk, 1, tmp_v+3);
    cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, tmp_v+4);
     communicator_sum(tmp_v,5);
     sigma=tmp_v[0];
     rho=tmp_v[1];//omega
     fai=tmp_v[2];
     theta=tmp_v[3];//delta
     
     alpha = tau/(sigma+small);
     minalpha = minone*alpha;
       cublasDaxpy(handle, nIntCells, &alpha, d_pk, 1, d_phi, 1);
       cublasDaxpy(handle, nIntCells, &minalpha, d_Apk, 1, d_res, 1);
    preconditioner(d_res, d_zk);
     tauo = tau;
     tau  = tau - alpha*fai - alpha*rho + alpha*alpha*theta;
     beta = tau/(tauo+small);

     cublasDscal(handle, nIntCells, &beta, d_pk, 1);
     cublasDaxpy(handle, nIntCells, &one, d_zk, 1, d_pk, 1);
     //check convergence 
     resl=tmp_v[4];
     resl = sqrt(resl);
     rsm = resl/(res0+small);
     if(resvec!=NULL)
	 resvec[iiter]=rsm;
     if(rsm<tol)
     {
       usediter = iiter;
       cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
       gpu_cg_free();
       delete []pk; 
       delete []res; 
       delete []Apk; 
       delete []tmp_v; 
       return;
     }
     

} 
    cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    gpu_cg_free();
    delete []pk; 
    delete []res; 
    delete []Apk; 
    delete []tmp_v; 
}     

extern "C" void gpu_bicgstab_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{

//right preconditioned classical BICGStab method with Jacobi preconditioner
     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_bicgstab_malloc(nIntCells, nGhstCells);
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;

     double *zk = new double[nIntCells+nGhstCells]; 
     
     double *res = new double[nIntCells];
     double *pk = new double[nIntCells];
     double *uk = new double[nIntCells];
     double *vk = new double[nIntCells];
     double *sk = new double[nIntCells];
     double *reso = new double[nIntCells];
     double *tmp_v = new double[2];
     
     double alpha, minalpha, omega, resl, rsm, beta, beto, gama, mingama,tmp;
     int iiter;
     
     double small = 1e-20, one=1.0, minone = -1.0;


//     for(icell=0;icell<nIntCells;icell++)
//{
//       tmp = a_p[icell] + small;
//       diag[icell] = one/tmp;
//}
     //jacobiInitCUDA<<<dim3(d_nblock), dim3(d_nthread),0,0>>>(nIntCells, d_a_p, d_diag, small);
    cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);

    cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_diag, diag, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pk, pk, sizeof(double) * (nIntCells+nGhstCells), cudaMemcpyHostToDevice);
    devicemtx->bmAx(d_q, sol, d_phi,d_res);
    cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &res0);
     communicator_sum(res0);
     res0 = sqrt(res0);
     cublasDcopy(handle, nIntCells,d_res,1,d_reso,1); 
     cudaMemset(d_uk, 0, sizeof(double) * nIntCells);
     cudaMemset(d_vk, 0, sizeof(double) * nIntCells);
     cudaMemset(d_zk, 0, sizeof(double) * (nIntCells + nGhstCells));
     cudaMemset(d_pk, 0, sizeof(double) * nIntCells);
     cudaMemset(d_sk, 0, sizeof(double) * nIntCells);
     alpha = 1.0;
     beto  = 1.0;
     gama  = 1.0;

    for(iiter=0;iiter<maxiter;iiter++)
{
    usediter = iiter;    
 
     cublasDdot(handle, nIntCells, d_res, 1, d_reso, 1, &beta);
     communicator_sum(beta);
     omega = (beta*gama)/(alpha*beto+small);
     beto = beta;

     minalpha = minone*alpha;
     
     cublasDaxpy(handle,nIntCells,&minalpha,d_uk,1,d_pk,1);
     cublasDscal(handle,nIntCells,&omega,d_pk,1);
     cublasDaxpy(handle,nIntCells,&one,d_res,1,d_pk,1);

     // applying the preconditioner 
     //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_pk, d_zk);
     preconditioner (d_pk, d_zk);
     devicemtx->SpMV(zk, d_zk,d_uk);
     //matmultvecc(nIntCells,nGhstCells, ell_val, ell_idx, d_a_p, zk, d_zk,d_uk);

     cublasDdot(handle, nIntCells,d_uk,1,d_reso,1,&tmp);
     communicator_sum(tmp);
     gama = beta/(tmp+small);
     mingama = minone*gama;
     cublasDcopy(handle, nIntCells,d_res,1,d_sk,1); 
     cublasDaxpy(handle, nIntCells,&mingama,d_uk,1,d_sk,1);

     // applying the preconditioner 
     //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_sk, d_zk);
     preconditioner (d_sk, d_zk);
     devicemtx->SpMV(zk, d_zk,d_vk);
     //matmultvecc(nIntCells,nGhstCells, ell_val, ell_idx, d_a_p, zk, d_zk,d_vk);

     cublasDdot(handle, nIntCells,d_vk,1,d_sk,1,tmp_v);
     cublasDdot(handle, nIntCells,d_vk,1,d_vk,1,tmp_v+1);
     communicator_sum(tmp_v,2);
     alpha = tmp_v[0]/(tmp_v[1]+small);

     cublasDaxpy(handle, nIntCells,&gama,d_pk,1,d_phi,1);
     cublasDaxpy(handle, nIntCells,&alpha,d_sk,1,d_phi,1);
     
     minalpha = minone*alpha;
     cublasDcopy(handle, nIntCells,d_sk,1,d_res,1); 
     cublasDaxpy(handle, nIntCells,&minalpha,d_vk,1,d_res,1);

     //check convergence 
     cublasDdot(handle, nIntCells,d_res,1,d_res,1,&resl);    
     communicator_sum(resl);
     resl = sqrt(resl);
     rsm = resl/(res0+small);
     if(resvec!=NULL)
	 resvec[iiter]=rsm;
     //printf("bicgstab iter=%d, rsm= % 4.5f\n", iiter, rsm);
     if(rsm<tol)
     {
       usediter = iiter;
       //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_phi, d_phi);
       preconditioner (d_phi, d_phi);
       cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
       gpu_bicgstab_free();
       delete []zk; 
       delete []res; 
       delete []pk; 
       delete []uk; 
       delete []vk; 
       delete []sk; 
       delete []reso; 
       delete []tmp_v; 
       return;
     }
     
} 
       // applying the preconditioner 
       preconditioner (d_phi, d_phi);
       //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_phi, d_phi);
       cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
       gpu_bicgstab_free();
    delete []zk; 
    delete []res; 
    delete []pk; 
    delete []uk; 
    delete []vk; 
    delete []sk; 
    delete []reso; 
    delete []tmp_v; 
}     



extern "C" void gpu_ibicgstab_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{
     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_bicgstab_malloc(nIntCells, nGhstCells);
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;
//right preconditioned improved BICGStab method with Jacobi preconditioner

     double *vkm = new double[nIntCells+nGhstCells]; 
     double *tkm = new double[nIntCells+nGhstCells]; 
     double *resm = new double[nIntCells+nGhstCells]; 
     
     double *res = new double[nIntCells];
     double *tk = new double[nIntCells];
     double *pk = new double[nIntCells];
     double *qk = new double[nIntCells];
     double *zk = new double[nIntCells];
     double *uk = new double[nIntCells];
     double *vk = new double[nIntCells];
     double *vko = new double[nIntCells];
     double *sk = new double[nIntCells];
     double *reso = new double[nIntCells];
     double *tmp_v = new double[6];
     
     double pi,tau,fai,sigma,alpha,minalpha,alphao,rou,rouo,omega,minomega;
     double beta,delta,mindelta,theta,kappa,mu,nu;
     double resl, rsm, tmp1, tmp2, mintmp2;
     int iiter;
     
     double small = 1e-20, one = 1.0, minone = -1.0, zero = 0.0;
     cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_res, res, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
     cublasDdot(handle, nIntCells, d_res, 1, d_res, 1, &res0);
     communicator_sum(res0);
     res0 = sqrt(res0);
     
     cublasDcopy(handle, nIntCells,d_res,1,d_reso,1); 
     cudaMemset(d_qk, 0, sizeof(double) * nIntCells);
     cudaMemset(d_vk, 0, sizeof(double) * nIntCells);
     cudaMemset(d_zk, 0, sizeof(double) * nIntCells);
     
     alpha = 1.0;
     alphao = 1.0;
     rou  = 1.0;
     rouo = 1.0;
     omega  = 1.0;
     pi = 0.0;
     tau = 0.0;
     mu = 0.0;

     cublasDdot(handle,nIntCells,d_phi,1,d_phi,1,&tmp1);
     //cublasDdot(handle,(nIntCells+nGhstCells),d_phi,1,d_phi,1,&tmp1);
     communicator_sum(tmp1);
     tmp1 = sqrt(tmp1);
     if(tmp1==zero)
     {
     // tk = A*res
     // applying the preconditioner 
     preconditioner (d_res, d_resm);
     devicemtx->SpMV(resm, d_resm,d_tk);
     cudaMemset(d_pk, 0, sizeof(double) * nIntCells);
     }
     else
     {
     // pk = A*A*phi
        devicemtx->SpMV(sol, d_phi,d_pk);
     // applying the preconditioner 
        preconditioner (d_pk, d_resm);
        devicemtx->SpMV(resm, d_resm,d_pk);
        preconditioner (d_res, d_resm);
     // tk = pk + A*res
     // applying the preconditioner 
        devicemtx->SpMV(resm, d_resm,d_tk);
        cublasDaxpy(handle,nIntCells,&one,d_pk,1,d_tk,1);
     }

     cublasDdot(handle,nIntCells,d_res,1,d_res,1,tmp_v);
     cublasDdot(handle,nIntCells,d_res,1,d_tk,1,tmp_v+1);
     communicator_sum(tmp_v,2);
     fai = tmp_v[0];
     sigma = tmp_v[1];
    for(iiter=0;iiter<maxiter;iiter++)
{
    usediter = iiter;    
     
     rouo = rou;
     rou = fai-omega*mu;
     delta = (rou*alpha)/(rouo+small);
     beta = delta/(omega+small);
     tau = sigma+beta*tau-delta*pi;
     alphao = alpha;
     alpha = rou/(tau+small);
 
     cublasDcopy(handle,nIntCells,d_vk,1,d_vko,1); 
     cublasDcopy(handle,nIntCells,d_tk,1,d_vk,1); 
     minomega = minone*omega;
     cublasDaxpy(handle,nIntCells,&minomega,d_pk,1,d_vk,1);
     cublasDaxpy(handle,nIntCells,&beta,d_vko,1,d_vk,1);
     mindelta = minone*delta;
     cublasDaxpy(handle,nIntCells,&mindelta,d_qk,1,d_vk,1);

     // applying the preconditioner 
     //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_vk, d_vkm);
     preconditioner (d_vk, d_vkm);
     devicemtx->SpMV(vkm, d_vkm,d_qk);
     cublasDcopy(handle,nIntCells,d_res,1,d_sk,1); 
     minalpha = minone*alpha;
     cublasDaxpy(handle,nIntCells,&minalpha,d_vk,1,d_sk,1);

     minomega = minone*omega;
     cublasDaxpy(handle,nIntCells,&minomega,d_pk,1,d_tk,1);
     minalpha = minone*alpha;
     cublasDaxpy(handle,nIntCells,&minalpha,d_qk,1,d_tk,1);
     
     tmp1=(beta*alpha)/(alphao+small);
     tmp2=alpha*delta;
     mintmp2=minone*tmp2;
     cublasDscal(handle,nIntCells,&tmp1,d_zk,1);
     cublasDaxpy(handle,nIntCells,&alpha,d_res,1,d_zk,1);
     cublasDaxpy(handle,nIntCells,&mintmp2,d_vko,1,d_zk,1);

     // applying the preconditioner 
     //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_tk, d_tkm);
     preconditioner (d_tk, d_tkm);
     devicemtx->SpMV(tkm, d_tkm,d_pk);

     cublasDdot(handle,nIntCells,d_reso,1,d_sk,1,tmp_v);
     cublasDdot(handle,nIntCells,d_reso,1,d_qk,1,tmp_v+1);
     cublasDdot(handle,nIntCells,d_sk,1,d_tk,1,tmp_v+2);
     cublasDdot(handle,nIntCells,d_tk,1,d_tk,1,tmp_v+3);
     cublasDdot(handle,nIntCells,d_reso,1,d_tk,1,tmp_v+4);
     cublasDdot(handle,nIntCells,d_reso,1,d_pk,1,tmp_v+5);
     communicator_sum(tmp_v,6);

     fai=tmp_v[0];
     pi=tmp_v[1];
     theta=tmp_v[2];
     kappa=tmp_v[3];
     mu=tmp_v[4];
     nu=tmp_v[5];
     omega = theta/(kappa+small);
     sigma=mu-omega*nu;

     cublasDcopy(handle,nIntCells,d_sk,1,d_res,1); 
     minomega = minone*omega;
     cublasDaxpy(handle,nIntCells,&minomega,d_tk,1,d_res,1);

     cublasDaxpy(handle,nIntCells,&one,d_zk,1,d_phi,1);
     cublasDaxpy(handle,nIntCells,&omega,d_sk,1,d_phi,1);
     
     //check convergence 
     cublasDdot(handle,nIntCells,d_res,1,d_res,1,&resl);    
     communicator_sum(resl);
     resl = sqrt(resl);
     rsm = resl/(res0+small);
     if(resvec!=NULL)
	 resvec[iiter]=rsm;
     if(rsm<tol)
     {
       usediter = iiter;
       // applying the preconditioner 
       //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_phi, d_phi);
       preconditioner (d_phi, d_phi);
       cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
       gpu_bicgstab_free();
       delete []vkm; 
       delete []tkm; 
       delete []resm; 
       delete []res; 
       delete []tk; 
       delete []pk; 
       delete []qk; 
       delete []zk; 
       delete []uk; 
       delete []vk; 
       delete []vko; 
       delete []sk; 
       delete []reso; 
       delete []tmp_v; 
       return;
     }
     
} 
       // applying the preconditioner 
       //jacobiCUDA<<<dim3(d_nblock), dim3(d_nthread), 0, 0>>>( nIntCells, d_diag, d_phi, d_phi);
       preconditioner (d_phi, d_phi);
       cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
       gpu_bicgstab_free();
    delete []vkm; 
    delete []tkm; 
    delete []resm; 
    delete []res; 
    delete []tk; 
    delete []pk; 
    delete []qk; 
    delete []zk; 
    delete []uk; 
    delete []vk; 
    delete []vko; 
    delete []sk; 
    delete []reso; 
    delete []tmp_v; 
}     



extern "C" void gpu_gcr_solve(double *rhs, double *sol, int &restart, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{

//left preconditioned GCR method with Jacobi preconditioner
//here the classical Gram-Schmidt scheme is utilized

     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_gcr_malloc(nIntCells, nGhstCells,restart);
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;
     
     double **uk = new double*[restart];
     double **ck = new double*[restart]; 
     for (int i=0; i<restart; i++)
     {
        uk[i] = new double[nIntCells+nGhstCells];
        ck[i] = new double[nIntCells];
     } 
     
     double *res  = new double[nIntCells];
     double *tmp_v  = new double[2];
     double *alpha = new double[restart];
     
     double error, rnorm, tmp, mintmp;
     int iiter, i, k;
     
     //double small = 0, one = 1.0, minone = -1.0;
     double small = 1e-20, one = 1.0, minone = -1.0;

     cudaMemset(d_ukr, 0, sizeof(double) * restart*(nIntCells+nGhstCells));
     cudaMemset(d_ckr, 0, sizeof(double) * restart*nIntCells);
     memset(alpha, 0, restart*sizeof(double));
     cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
     //cudaMemcpy(d_pk, pk, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     
     //applying the preconditioner
     preconditioner (d_res, d_res);
     cublasDdot(handle,nIntCells,d_res,1,d_res,1,&res0);
     
     communicator_sum(res0);
     res0 = sqrt(res0);
     error = res0;

    iiter = 0;
    while(iiter<maxiter)
{
    k = 0;
    while( k<restart && iiter<maxiter )
    {
       //printf("k = %d\n",k);
       cublasDcopy(handle,nIntCells,d_res,1,d_ukr+k*(nIntCells+nGhstCells),1);
       devicemtx->SpMV(uk[k], d_ukr+k*(nIntCells+nGhstCells),d_ckr+k*nIntCells);
       // applying the preconditioner 
       preconditioner (d_ckr+k*nIntCells, d_ckr+k*nIntCells);
       for(i=0;i<k;i++)
       {
          cublasDdot(handle,nIntCells,d_ckr+i*nIntCells,1,d_ckr+k*nIntCells,1,alpha+i); 
       } 
          communicator_sum(alpha,k);
       for(i=0;i<k;i++)
       {
          tmp = minone*alpha[i];
          cublasDaxpy(handle,nIntCells,&tmp,d_ckr+i*nIntCells,1,d_ckr+k*nIntCells,1);
          cublasDaxpy(handle,nIntCells,&tmp,d_ukr+i*(nIntCells+nGhstCells),1,d_ukr+k*(nIntCells+nGhstCells),1);
          //cublasDaxpy(handle,nIntCells+nGhstCells,&tmp,d_ukr+i*(nIntCells+nGhstCells),1,d_ukr+k*(nIntCells+nGhstCells),1);
       } 
       
       
       cublasDdot(handle,nIntCells,d_ckr+k*nIntCells,1,d_ckr+k*nIntCells,1,tmp_v); 
       cublasDdot(handle,nIntCells,d_ckr+k*nIntCells,1,d_res,1,tmp_v+1); 
       //communicator_sum(tmp_v[1]);
       //communicator_sum(tmp_v[2]);
       communicator_sum(tmp_v,2);
       tmp_v[0] = sqrt(tmp_v[0]);
       tmp = one/(tmp_v[0]+small);
       cublasDscal(handle,nIntCells,&tmp,d_ukr+k*(nIntCells+nGhstCells),1);
       //cublasDscal(handle,nIntCells+nGhstCells,&tmp,d_ukr+k*(nIntCells+nGhstCells),1);
       cublasDscal(handle,nIntCells,&tmp,d_ckr+k*nIntCells,1);
       
       tmp = tmp_v[1]/(tmp_v[0]+small);
       mintmp = minone*tmp;
       //cublasDaxpy(handle,nIntCells+nGhstCells,&tmp,d_ukr+k*(nIntCells+nGhstCells),1,d_phi,1);
       cublasDaxpy(handle,nIntCells,&tmp,d_ukr+k*(nIntCells+nGhstCells),1,d_phi,1);
       cublasDaxpy(handle,nIntCells,&mintmp,d_ckr+k*nIntCells,1,d_res,1);
     

       //check convergence 
       cublasDdot(handle,nIntCells,d_res,1,d_res,1,&tmp);    
       communicator_sum(tmp);
       rnorm = sqrt(tmp);
       error = rnorm/(res0+small);
       if(resvec!=NULL)
	   resvec[iiter]=error;
       //printf("gcr iter=%d, rsm= % 4.5f\n", iiter, error);
       if(error<tol)
       {
         usediter = iiter;
         cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
         gpu_gcr_free();
         delete []res; 
         delete []tmp_v; 
         delete []alpha; 
         for (int i=0; i<restart; i++)
         {
           	delete []uk[i]; 
         	delete []ck[i]; 
         } 
         delete []uk; 
         delete []ck; 
         return;
       }
       k = k + 1;
       iiter = iiter + 1;
    }
}
    usediter = iiter;
    cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    gpu_gcr_free();
    delete []res; 
    delete []tmp_v; 
    delete []alpha; 
    for (int i=0; i<restart; i++)
    {
      	delete []uk[i]; 
    	delete []ck[i]; 
    } 
    delete []uk; 
    delete []ck; 
}



extern "C" void gpu_igcr_solve(double *rhs, double *sol, int &restart, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{

//left preconditioned GCR method with Jacobi preconditioner
//here the modified Gram-Schmidt scheme is utilized

     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_gcr_malloc(nIntCells, nGhstCells,restart);
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;
     
     double **uk = new double*[restart];
     double **ck = new double*[restart]; 
     for (int i=0; i<restart; i++)
     {
        uk[i] = new double[nIntCells+nGhstCells];
        ck[i] = new double[nIntCells];
     } 
     
     double *res  = new double[nIntCells];
     double *tmp_v  = new double[2];
     
     double alpha, minalpha, error, rnorm, tmp, mintmp;
     int iiter, i, k;
     
     double small = 1e-20, one = 1.0, minone = -1.0;
   
     alpha  = 0.0;

     cudaMemset(d_ukr, 0, sizeof(double) * restart*(nIntCells+nGhstCells));
     cudaMemset(d_ckr, 0, sizeof(double) * restart*nIntCells);
     cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
     //cudaMemcpy(d_pk, pk, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     
     //applying the preconditioner
     preconditioner (d_res,d_res);
     cublasDdot(handle,nIntCells,d_res,1,d_res,1,&res0);
     
     communicator_sum(res0);
     res0 = sqrt(res0);
     error = res0;

    iiter = 0;
    while(iiter<maxiter)
{
    k = 0;
    while( k<restart && iiter<maxiter )
    {
       cublasDcopy(handle, nIntCells, d_res, 1, d_ukr+k*(nIntCells+nGhstCells), 1);
       devicemtx->SpMV(uk[k], d_ukr+k*(nIntCells+nGhstCells),d_ckr+k*nIntCells);
       // applying the preconditioner 
       preconditioner( d_ckr+k*nIntCells, d_ckr+k*nIntCells);
       for(i=0;i<k;i++)
       {
          cublasDdot(handle,nIntCells,d_ckr+i*nIntCells,1,d_ckr+k*nIntCells,1,&alpha); 
          communicator_sum(alpha);
          minalpha = minone*alpha;
          cublasDaxpy(handle,nIntCells,&minalpha,d_ckr+i*nIntCells,1,d_ckr+k*nIntCells,1);
          //cublasDaxpy(handle,nIntCells+nGhstCells,&minalpha,d_ukr+i*(nIntCells+nGhstCells),1,d_ukr+k*(nIntCells+nGhstCells),1);
          cublasDaxpy(handle,nIntCells,&minalpha,d_ukr+i*(nIntCells+nGhstCells),1,d_ukr+k*(nIntCells+nGhstCells),1);
       } 
       
       cublasDdot(handle,nIntCells,d_ckr+k*nIntCells,1,d_ckr+k*nIntCells,1,tmp_v);
       cublasDdot(handle,nIntCells,d_ckr+k*nIntCells,1,d_res,1,tmp_v+1);
       //communicator_sum(tmp_v[1]);
       //communicator_sum(tmp_v[2]);
       communicator_sum(tmp_v,2);
       tmp_v[0] = sqrt(tmp_v[0]);
       tmp = one/(tmp_v[0]+small);
       cublasDscal(handle,nIntCells,&tmp,d_ukr+k*(nIntCells+nGhstCells),1);
       //cublasDscal(handle,nIntCells+nGhstCells,&tmp,d_ukr+k*(nIntCells+nGhstCells),1);
       cublasDscal(handle,nIntCells,&tmp,d_ckr+k*nIntCells,1); 
       
       tmp = tmp_v[1]/(tmp_v[0]+small);
       mintmp = minone*tmp;
       cublasDaxpy(handle,nIntCells,&tmp,d_ukr+k*(nIntCells+nGhstCells),1,d_phi,1);
       //cublasDaxpy(handle,nIntCells+nGhstCells,&tmp,d_ukr+k*(nIntCells+nGhstCells),1,d_phi,1);
       cublasDaxpy(handle,nIntCells,&mintmp,d_ckr+k*nIntCells,1,d_res,1);
     

       //check convergence 
       cublasDdot(handle,nIntCells,d_res,1,d_res,1,&tmp);    
       communicator_sum(tmp);
       rnorm = sqrt(tmp);
       error = rnorm/(res0+small);
     if(resvec!=NULL)
	 resvec[iiter]=error;
       if(error<tol)
       {
         usediter = iiter;
         cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
         gpu_gcr_free();
         delete []res; 
         delete []tmp_v; 
         for (int i=0; i<restart; i++)
         {
           	delete []uk[i]; 
         	delete []ck[i]; 
         } 
         delete []uk; 
         delete []ck; 
         return;
       }
       k = k + 1;
       iiter = iiter + 1;
    }
}
    usediter = iiter;
    cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    gpu_gcr_free();
    delete []res; 
    delete []tmp_v; 
    for (int i=0; i<restart; i++)
    {
      	delete []uk[i]; 
    	delete []ck[i]; 
    } 
    delete []uk; 
    delete []ck; 
}


void updatecgpu(double *d_phi, int k, double **H, double *s, double *d_V, int n)
{
     double *y = new double[k];
     int i, j, intone=1;
     double small = 1e-20;
 
     cblas_dcopy(k,s,intone,y,intone);
     for (i=k-1;i>=0;i--)
     {
         y[i] = y[i]/(H[i][i]+small);
         for (j=i-1;j>=0;j--)
         {
             y[j] = y[j] - H[j][i]*y[i];
         }
     }
    
     for (i=0;i<k;i++)
     {
         cublasDaxpy(handle,n,y+i,d_V+i*n,1,d_phi,1);
     }
}

extern "C" void gpu_gmres_solve(double *rhs, double *sol, int &restart, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{

//left preconditioned GMRES method with Jacobi preconditioner
//here the modified Gram-Schmidt scheme is utilized

     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_gmres_malloc(nIntCells, nGhstCells,restart);
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;
     
     double **V = new double*[restart+1];
     //double **Z = new double*[restart+1];
     double **H = new double*[restart+1];
     for (int i=0; i<restart+1; i++)
     {
        V[i] = new double[nIntCells+nGhstCells];
     //   Z[i] = new double[nIntCells+nGhstCells];
        H[i] = new double[restart];
     } 
     
     double *res  = new double[nIntCells];
     double *w = new double[nIntCells];
     double *tmpv = new double[nIntCells];
     double *s = new double[restart+1];
     double *cs = new double[restart+1];
     double *sn = new double[restart+1];
     
     double beta, error, tmp, mintmp;
     int i, j, k, m;
     
     double small = 1e-20, one = 1.0, minone = -1.0;
   
     beta  = 0.0;
     res0  = 0.0;
     m = restart;

     cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
     //cudaMemcpy(d_pk, pk, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemset(d_V, 0, sizeof(double) * (restart+1)*(nIntCells+nGhstCells));
     for(int n=0; n<restart+1; n++){
         memset(H[n], 0, restart*sizeof(double));
     }
     memset(s, 0, (restart+1)*sizeof(double));
     memset(cs, 0, (restart+1)*sizeof(double));
     memset(sn, 0, (restart+1)*sizeof(double));
     //cudaMemset(d_H, 0, sizeof(double) * (restart+1)*restart);
     //cudaMemset(d_s, 0, sizeof(double) * (restart+1));
     //cudaMemset(d_cs, 0, sizeof(double) * (restart+1));
     //cudaMemset(d_sn, 0, sizeof(double) * (restart+1));
     //residuumc(nIntCells, a_p, a_l, NbCell_ptr_c, NbCell_s, q, phi, res);
     
     //applying the preconditioner
     preconditioner(d_res, d_res);
     cublasDdot(handle,nIntCells,d_res,1,d_res,1,&res0);
     
     communicator_sum(res0);
     res0 = sqrt(res0);
     error = res0/(res0+small);
     beta = res0;

    j = 0;
    while(j<maxiter)
    {
     cudaMemset(d_V, 0, sizeof(double) * (restart+1)*(nIntCells+nGhstCells));
     for(int n=0; n<restart+1; n++){
         memset(H[n], 0, restart*sizeof(double));
     }
     memset(s, 0, (restart+1)*sizeof(double));
     memset(cs, 0, (restart+1)*sizeof(double));
     memset(sn, 0, (restart+1)*sizeof(double));
     //cudaMemset(d_H, 0, sizeof(double) * (restart+1)*restart);
     //cudaMemset(d_s, 0, sizeof(double) * (restart+1));
     //cudaMemset(d_cs, 0, sizeof(double) * (restart+1));
     //cudaMemset(d_sn, 0, sizeof(double) * (restart+1));
    
    tmp = one/(beta+small); 
    cublasDscal(handle,nIntCells,&tmp,d_res,1); 
    cublasDcopy(handle,nIntCells,d_res,1,d_V,1);
    s[0] = beta;

    i = 0;
    while( i<restart && j<maxiter )
    {
       devicemtx->SpMV(V[i], d_V+i*(nIntCells+nGhstCells),d_w);
       // applying the preconditioner 
       preconditioner( d_w, d_w);
       for(k=0;k<=i;k++)
       {
          cublasDdot(handle,nIntCells,d_w,1,d_V+k*(nIntCells+nGhstCells),1,&tmp); 
          communicator_sum(tmp);
          H[k][i] = tmp;
          mintmp = minone*tmp;
          cublasDaxpy(handle,nIntCells,&mintmp,d_V+k*(nIntCells+nGhstCells),1,d_w,1);
       } 
       
       cublasDdot(handle,nIntCells,d_w,1,d_w,1,&tmp);
       communicator_sum(tmp);
       tmp = sqrt(tmp);
       H[i+1][i] = tmp;
       tmp = one/(tmp+small);
       cublasDscal(handle,nIntCells,&tmp,d_w,1);
       cublasDcopy(handle,nIntCells,d_w,1,d_V+(i+1)*(nIntCells+nGhstCells),1);
       
       for(k=0;k<i;k++)
       {
          applyplanerotationc(H[k][i],H[k+1][i],cs[k],sn[k]);
       }
       generateplanerotationc(H[i][i],H[i+1][i],cs[i],sn[i]);
       applyplanerotationc(H[i][i],H[i+1][i],cs[i],sn[i]);
       applyplanerotationc(s[i],s[i+1],cs[i],sn[i]);

       //check convergence 
       error = fabs(s[i+1])/(res0+small);
     if(resvec!=NULL)
	 resvec[i]=error;
       if(error<tol)
       {
         updatecgpu(d_phi,i,H,s,d_V,nIntCells+nGhstCells);
         usediter = j;
         cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    	 gpu_gmres_free();
         delete []res; 
         delete []w; 
         delete []tmpv; 
         delete []s; 
         delete []cs; 
         delete []sn; 
         for (int i=0; i<restart+1; i++)
         {
           	delete []V[i]; 
         	delete []H[i]; 
         } 
         delete []V; 
         delete []H; 
         return;
       }
       i = i + 1;
       j = j + 1;
    }
    updatecgpu(d_phi,m,H,s,d_V,nIntCells+nGhstCells);
    //residuumc(nIntCells, a_p, a_l, NbCell_ptr_c, NbCell_s, q, phi, res);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
    //applying the preconditioner
    preconditioner(d_res, d_res);
    beta = fabs(s[m]);
    usediter = j;
    }
    cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    gpu_gmres_free();
    delete []res; 
    delete []w; 
    delete []tmpv; 
    delete []s; 
    delete []cs; 
    delete []sn; 
    for (int i=0; i<restart+1; i++)
    {
      	delete []V[i]; 
    	delete []H[i]; 
    } 
    delete []V; 
    delete []H; 
}



extern "C" void gpu_fgmres_solve(double *rhs, double *sol, int &restart, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec=NULL)
{

//right preconditioned FGMRES method with Jacobi preconditioner
//only right preconditioning is allowed in the FGMRES method
//here the modified Gram-Schmidt scheme is utilized

     int nIntCells=hostmtx.nInterior;
     int nGhstCells=hostmtx.nHalo;
     gpu_gmres_malloc(nIntCells, nGhstCells,restart);
     d_exchange_ptr=devicemtx->exchange_ptr;
     double res0;
     
     double **V = new double*[restart+1];
     double **Z = new double*[restart+1];
     double **H = new double*[restart+1];
     for (int i=0; i<restart+1; i++)
     {
        V[i] = new double[nIntCells+nGhstCells];
        Z[i] = new double[nIntCells+nGhstCells];
        H[i] = new double[restart];
     } 
     
     double *res  = new double[nIntCells];
     double *w = new double[nIntCells];
     double *tmpv = new double[nIntCells];
     double *s = new double[restart+1];
     double *cs = new double[restart+1];
     double *sn = new double[restart+1];
     
     double beta, error, tmp, mintmp;
     int i, j, k, m;
     
     double small = 1e-20, one = 1.0, minone = -1.0;
   
     beta  = 0.0;
     res0  = 0.0;
     m = restart;

     cudaMemcpy(d_phi, sol_init, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     cudaMemcpy(d_q, rhs, sizeof(double) * nIntCells, cudaMemcpyHostToDevice);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
     cudaMemset(d_V, 0, sizeof(double) * (restart+1)*(nIntCells+nGhstCells));
     cudaMemset(d_Z, 0, sizeof(double) * (restart+1)*(nIntCells+nGhstCells));
     for(int n=0; n<restart+1; n++){
         memset(H[n], 0, restart*sizeof(double));
     }
     memset(s, 0, (restart+1)*sizeof(double));
     memset(cs, 0, (restart+1)*sizeof(double));
     memset(sn, 0, (restart+1)*sizeof(double));
     
     cublasDdot(handle,nIntCells,d_res,1,d_res,1,&res0);
     
     communicator_sum(res0);
     res0 = sqrt(res0);
     error = res0/(res0+small);
     beta = res0;

    j = 0;
    while(j<maxiter)
    {
     cudaMemset(d_V, 0, sizeof(double) * (restart+1)*(nIntCells+nGhstCells));
     cudaMemset(d_Z, 0, sizeof(double) * (restart+1)*(nIntCells+nGhstCells));
     for(int n=0; n<restart+1; n++){
         memset(H[n], 0, restart*sizeof(double));
     }
     memset(s, 0, (restart+1)*sizeof(double));
     memset(cs, 0, (restart+1)*sizeof(double));
     memset(sn, 0, (restart+1)*sizeof(double));
    
    tmp = one/(beta+small); 
    cublasDscal(handle,nIntCells,&tmp,d_res,1); 
    cublasDcopy(handle,nIntCells,d_res,1,d_V,1);
    s[0] = beta;

    i = 0;
    while( i<restart && j<maxiter )
    {
       // applying the preconditioner 
       preconditioner( d_V+i*(nIntCells+nGhstCells), d_Z+i*(nIntCells+nGhstCells));
       devicemtx->SpMV(Z[i], d_Z+i*(nIntCells+nGhstCells),d_w);
       for(k=0;k<=i;k++)
       {
          cublasDdot(handle,nIntCells,d_w,1,d_V+k*(nIntCells+nGhstCells),1,&tmp); 
          communicator_sum(tmp);
          H[k][i] = tmp;
          mintmp = minone*tmp;
          cublasDaxpy(handle,nIntCells,&mintmp,d_V+k*(nIntCells+nGhstCells),1,d_w,1);
       } 
       
       cublasDdot(handle,nIntCells,d_w,1,d_w,1,&tmp);
       communicator_sum(tmp);
       tmp = sqrt(tmp);
       H[i+1][i] = tmp;
       tmp = one/(tmp+small);
       cublasDscal(handle,nIntCells,&tmp,d_w,1);
       cublasDcopy(handle,nIntCells,d_w,1,d_V+(i+1)*(nIntCells+nGhstCells),1);
       
       for(k=0;k<i;k++)
       {
          applyplanerotationc(H[k][i],H[k+1][i],cs[k],sn[k]);
       }
       generateplanerotationc(H[i][i],H[i+1][i],cs[i],sn[i]);
       applyplanerotationc(H[i][i],H[i+1][i],cs[i],sn[i]);
       applyplanerotationc(s[i],s[i+1],cs[i],sn[i]);

       //check convergence 
       error = fabs(s[i+1])/(res0+small);
     if(resvec!=NULL)
	 resvec[i]=error;
       if(error<tol)
       {
         updatecgpu(d_phi,i,H,s,d_Z,nIntCells+nGhstCells);
         usediter = j;
         //preconditioner(d_phi, d_phi);
         cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    	 gpu_gmres_free();
         delete []res; 
         delete []w; 
         delete []tmpv; 
         delete []s; 
         delete []cs; 
         delete []sn; 
         for (int i=0; i<restart+1; i++)
         {
           	delete []V[i]; 
           	delete []Z[i]; 
         	delete []H[i]; 
         } 
         delete []V; 
         delete []Z; 
         delete []H; 
         return;
       }
       i = i + 1;
       j = j + 1;
    }
    updatecgpu(d_phi,m,H,s,d_Z,nIntCells+nGhstCells);
     devicemtx->bmAx(d_q, sol, d_phi,d_res);
    beta = fabs(s[m]);
    usediter = j;
    }
    //preconditioner(d_phi, d_phi);
    cudaMemcpy(sol, d_phi, sizeof(double) * nIntCells, cudaMemcpyDeviceToHost);
    gpu_gmres_free();
    delete []res; 
    delete []w; 
    delete []tmpv; 
    delete []s; 
    delete []cs; 
    delete []sn; 
    for (int i=0; i<restart+1; i++)
    {
      	delete []V[i]; 
      	delete []Z[i]; 
    	delete []H[i]; 
    } 
    delete []V; 
    delete []Z; 
    delete []H; 
}



void applyplanerotationc(double &dx, double &dy, double csx, double snx)
{
     double tmp_scalar;

     tmp_scalar = csx*dx + snx*dy;
     dy = -snx*dx + csx*dy;
     dx = tmp_scalar;
}


void generateplanerotationc(double dx, double dy, double &csx, double &snx)
{
     double tmp_scalar, zero=0.0, one=1.0;

     if (dy == zero)
     {
        csx = one;
        snx = zero;
     }
     else if (fabs(dy)>fabs(dx))
     {
       tmp_scalar = dx/dy;
       snx = one/sqrt(one + tmp_scalar*tmp_scalar);
       csx = tmp_scalar*snx;
     }
     else
     {
       tmp_scalar = dy/dx;
       csx = one/sqrt(one + tmp_scalar*tmp_scalar);
       snx = tmp_scalar*csx;
     }

}

//void updatec(double *phi, int k, double **H, double *s, double **V, int n)
//{
//     double *y = new double[k];
//     int i, j, intone=1;
//     double small = 1e-20;
// 
//     cblas_dcopy(k,s,intone,y,intone);
//     for (i=k-1;i>=0;i--)
//     {
//         y[i] = y[i]/(H[i][i]+small);
//         for (j=i-1;j>=0;j--)
//         {
//             y[j] = y[j] - H[j][i]*y[i];
//         }
//     }
//    
//     for (i=0;i<k;i++)
//     {
//         cblas_daxpy(n,y[i],V[i],intone,phi,intone);
//     }
//
//}
