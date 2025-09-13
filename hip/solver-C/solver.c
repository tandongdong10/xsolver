#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libhead.h"
#include <unistd.h>

#ifndef TYPE
#define TYPE double
#endif

void b_ax(int nIntCells,double *val,int *ptr,int *col_idx,double *b0,double *x0,double *b_ax){
     spmv_unroll2_pre(nIntCells, nIntCells, ptr, val, col_idx, x0, b_ax);
     scal(nIntCells,-1.0,b_ax);
     axpy(nIntCells,1.0,b0,b_ax);
}
void my_solvebicgstab_c_(int maxiter, TYPE tol, int nIntCells, TYPE *a_p, TYPE *a_l, int *NbCell_ptr_c, int *NbCell_s, TYPE *b0, TYPE *x0, TYPE *res0, int *usediter)
{

    //right preconditioned classical BICGStab method with Jacobi preconditioner

    TYPE *zk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);

    TYPE *res = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
    TYPE *diag = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
    TYPE *pk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
    TYPE *uk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
    TYPE *vk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
    TYPE *sk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
    TYPE *reso = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
    TYPE *tmp_v = (TYPE *)malloc(sizeof(TYPE) * 2);

    TYPE alpha, minalpha, omega, resl, rsm, beta, beto, gama, mingama, tmp;
    int iiter, icell, intone = 1;
    TYPE small = 0, one = 1.0, minone = -1.0;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (icell = 0; icell < nIntCells; icell++)
    {
        tmp = a_p[icell] + small;
        diag[icell] = one / tmp;
    }
    
    b_ax(nIntCells,a_l, NbCell_ptr_c,NbCell_s, x0,res,res);
    
    *res0 = dot(nIntCells, res, res);
    *res0 = sqrt(*res0);

    copy(nIntCells, res, reso);
    memset(uk, 0, nIntCells * sizeof(TYPE));
    memset(pk, 0, nIntCells * sizeof(TYPE));
    alpha = 1.0;
    beto = 1.0;
    gama = 1.0;
    for (iiter = 0; iiter < maxiter; iiter++)
    {
        *usediter = iiter;

        beta = dot(nIntCells, res, reso);
        omega = (beta * gama) / (alpha * beto + small);
        beto = beta;
        //pk = res + omega*(pk - alpha*uk)
        kernel_1(nIntCells,omega,alpha,res,uk,pk);
        // applying the preconditioner
        jacobi(nIntCells, diag ,pk ,zk);
        spmv_unroll2_pre(nIntCells, nIntCells, NbCell_ptr_c, a_l, NbCell_s, zk, uk);

        tmp = dot(nIntCells, uk, reso);
        gama = beta / (tmp + small);
        //sk = res - gama * uk
        kernel_2(nIntCells,res,gama,uk,sk);
        // applying the preconditioner
        jacobi(nIntCells,diag,sk,zk);
        spmv_unroll2_pre(nIntCells, nIntCells, NbCell_ptr_c, a_l, NbCell_s, zk, vk);
        alpha = kernel_4(nIntCells,vk,sk,small);
        //x0 = x0 + gama * pk + alpha * sk
        kernel_3(nIntCells,gama,pk,alpha,sk,x0);
        //res = sk - alpha * vk
        kernel_2(nIntCells,sk,alpha,vk,res);
        //check convergence
        resl = dot(nIntCells, res, res);
        resl = sqrt(resl);
        rsm = resl / (*res0 + small);
        if (rsm < tol)
        {
            *usediter = iiter;
            // applying the preconditioner
            jacobi(nIntCells,diag,x0,x0);
            free(zk);
            free(res);
            free(diag);
            free(pk);
            free(uk);
            free(vk);
            free(sk);
            free(reso);
            free(tmp_v);
            return;
        }
    }
    // applying the preconditioner
    *usediter = iiter;
    jacobi(nIntCells,diag,x0,x0);
    free(zk);
    free(res);
    free(diag);
    free(pk);
    free(uk);
    free(vk);
    free(sk);
    free(reso);
    free(tmp_v);
}

void my_solvecg_c_(int maxiter, TYPE tol, int nIntCells, TYPE *a_p,\
 TYPE *a_l, int *NbCell_ptr_c, int *NbCell_s, TYPE *b0, TYPE *x0, TYPE *res0, int *usediter)
{

   //left preconditioned cg method with Jacobi preconditioner

   TYPE *pk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
   TYPE *res = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
   TYPE *zk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
   TYPE *diag = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
   TYPE *Apk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);
   TYPE *qk = (TYPE *)malloc(sizeof(TYPE) * nIntCells);

   TYPE sigma, alpha, minalpha, taoo, tao, resl, rsm, beta, tmp;
   int iiter, icell, intone = 1;

   memset(Apk,0,nIntCells*sizeof(TYPE));
   TYPE small = 0, one = 1.0, minone = -1.0;
#ifdef _OPENMP
   #pragma omp parallel for
#endif
   for (icell = 0; icell < nIntCells; icell++)
   {
      tmp = a_p[icell] + small;
      diag[icell] = one / tmp;
   }

    b_ax(nIntCells,a_l, NbCell_ptr_c,NbCell_s, x0,res,res);
   //applying the preconditioner
   jacobi(nIntCells,diag,res,zk);


   *res0 = dot(nIntCells, res, res);
   tao = dot(nIntCells, res, zk);
   copy(nIntCells, zk, pk);
   *res0 = sqrt(*res0);

   for (iiter = 0; iiter < maxiter; iiter++)
   {
      *usediter = iiter;
      taoo = tao;

      spmv_unroll2_pre(nIntCells, nIntCells, NbCell_ptr_c, a_l, NbCell_s, pk, Apk);
      
      sigma = dot(nIntCells, pk, Apk);
      alpha = tao / (sigma + small);
      minalpha = minone * alpha;
      axpy(nIntCells, alpha, pk, x0);
      axpy(nIntCells, minalpha, Apk, res);

      // applying the preconditioner
      jacobi(nIntCells,diag,res,zk);
      tao = dot(nIntCells, res, zk);
      beta = tao / (taoo + small);

      scal(nIntCells, beta, pk);
      axpy(nIntCells, one, zk, pk);
      resl = dot(nIntCells, res, res);
      resl = sqrt(resl);
      rsm = resl / (*res0 + small);
      if (rsm < tol)
      {
         *usediter = iiter;
         free(pk);
         free(res);
         free(diag);
         free(Apk);
         free(qk);
         return;
      }
   }
   *usediter = iiter;
   free(pk);
   free(res);
   free(diag);
   free(Apk);
   free(qk);
}
