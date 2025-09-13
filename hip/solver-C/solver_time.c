#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libhead.h"
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#ifndef TYPE
#define TYPE double
#endif

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
    TYPE small = 1e-20, one = 1.0, minone = -1.0;
    
    struct timeval begin1, end1,st,ed;
    double Time = 0.0,dotTime = 0.0, veccombTime = 0.0, spmv_csrTime = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (icell = 0; icell < nIntCells; icell++)
    {
        tmp = a_p[icell] + small;
        diag[icell] = one / tmp;
    }
    copy(nIntCells,b0,res);//x0 = 0
     //spmv_csr(nIntCells, nIntCells, NbCell_ptr_c, a_l, NbCell_s, x0,res);
    //axpy(nIntCells,minone,b0,res);
    //scal(nIntCells,minone,res);
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
    gettimeofday(&st, NULL);
        *usediter = iiter;

        gettimeofday(&begin1, NULL);
        beta = dot(nIntCells, res, reso);
        omega = (beta * gama) / (alpha * beto + small);
        beto = beta;
        gettimeofday(&end1, NULL);
        dotTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
        
        gettimeofday(&begin1, NULL);
        //pk = res + omega*(pk - alpha*uk)
        kernel_1(nIntCells,omega,alpha,res,uk,pk);
        gettimeofday(&end1, NULL);
        veccombTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
        
        // applying the preconditioner
        jacobi(nIntCells, diag ,pk ,zk);
        
        gettimeofday(&begin1, NULL);
        spmv_csr(nIntCells, nIntCells, NbCell_ptr_c, a_l, NbCell_s, zk, uk);
        gettimeofday(&end1, NULL);
        spmv_csrTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;

        gettimeofday(&begin1, NULL);
        tmp = dot(nIntCells, uk, reso);
        gama = beta / (tmp + small);
        gettimeofday(&end1, NULL);
        dotTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
        //sk = res - gama * uk
        gettimeofday(&begin1, NULL);
        kernel_2(nIntCells,res,gama,uk,sk);
        gettimeofday(&end1, NULL);
        veccombTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
        // applying the preconditioner
        jacobi(nIntCells,diag,sk,zk);
        
        gettimeofday(&begin1, NULL);
        spmv_csr(nIntCells, nIntCells, NbCell_ptr_c, a_l, NbCell_s, zk, vk);
        gettimeofday(&end1, NULL);
        spmv_csrTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
        
        gettimeofday(&begin1, NULL);
        alpha = kernel_4(nIntCells,vk,sk,small);
        //x0 = x0 + gama * pk + alpha * sk
        kernel_3(nIntCells,gama,pk,alpha,sk,x0);
        //res = sk - alpha * vk
        kernel_2(nIntCells,sk,alpha,vk,res);
        gettimeofday(&end1, NULL);
        veccombTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
        //check convergence
        gettimeofday(&begin1, NULL);
        resl = dot(nIntCells, res, res);
        resl = sqrt(resl);
        rsm = resl / (*res0 + small);
        gettimeofday(&end1, NULL);
        dotTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
        
        if (rsm < tol)
        {
            *usediter = iiter;
            printf("KERNEL Time = %lf\n",Time);
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
    gettimeofday(&ed, NULL);
    Time += (ed.tv_sec - st.tv_sec)  + (ed.tv_usec - st.tv_usec)/1000000.0;
    }

    *usediter = iiter;
     printf("[Bicgstab](s) : TOTAL Time = %lf Dot Time = %lf Veccomb Time = %lf Spmv Time = %lf\n",Time, dotTime, veccombTime, spmv_csrTime);
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
   TYPE small = 1e-20, one = 1.0, minone = -1.0;
    
   struct timeval begin1, end1,st,ed;
   double Time = 0.0,dotTime = 0.0, veccombTime = 0.0, spmv_csrTime = 0.0;
#ifdef _OPENMP
   #pragma omp parallel for
#endif
   for (icell = 0; icell < nIntCells; icell++)
   {
      tmp = a_p[icell] + small;
      diag[icell] = one / tmp;
   }

   copy(nIntCells,b0,res);
   //residuumc(nIntCells, a_l, NbCell_ptr_c, NbCell_s, b0, x0, res);
   //applying the preconditioner
   jacobi(nIntCells,diag,res,zk);


   *res0 = dot(nIntCells, res, res);
   tao = dot(nIntCells, res, zk);
   copy(nIntCells, zk, pk);
   *res0 = sqrt(*res0);

   for (iiter = 0; iiter < maxiter; iiter++)
   {
      gettimeofday(&st, NULL);
      *usediter = iiter;
      taoo = tao;

      gettimeofday(&begin1, NULL);
      spmv_csr(nIntCells, nIntCells, NbCell_ptr_c, a_l, NbCell_s, pk, Apk);
      gettimeofday(&end1, NULL);
      spmv_csrTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
      
      gettimeofday(&begin1, NULL);
      sigma = dot(nIntCells, pk, Apk);
      alpha = tao / (sigma + small);
      minalpha = minone * alpha;
      gettimeofday(&end1, NULL);
      dotTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
      
      gettimeofday(&begin1, NULL);
      axpy(nIntCells, alpha, pk, x0);
      axpy(nIntCells, minalpha, Apk, res);
      gettimeofday(&end1, NULL);
      veccombTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;

      // applying the preconditioner
      jacobi(nIntCells,diag,res,zk);
      
      gettimeofday(&begin1, NULL);
      tao = dot(nIntCells, res, zk);
      beta = tao / (taoo + small);
      gettimeofday(&end1, NULL);
      dotTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
      
      gettimeofday(&begin1, NULL);
      scal(nIntCells, beta, pk);
      axpy(nIntCells, one, zk, pk);
      gettimeofday(&end1, NULL);
      veccombTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
      gettimeofday(&begin1, NULL);
      resl = dot(nIntCells, res, res);
      resl = sqrt(resl);
      rsm = resl / (*res0 + small);
      gettimeofday(&end1, NULL);
      dotTime += (end1.tv_sec - begin1.tv_sec)  + (end1.tv_usec - begin1.tv_usec)/1000000.0;
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
    gettimeofday(&ed, NULL);
    Time += (ed.tv_sec - st.tv_sec)  + (ed.tv_usec - st.tv_usec)/1000000.0;
   }
   *usediter = iiter;
    printf("[CG](s) : TOTAL Time = %lf Dot Time = %lf Veccomb Time = %lf Spmv Time = %lf\n",Time, dotTime, veccombTime, spmv_csrTime);
   free(pk);
   free(res);
   free(diag);
   free(Apk);
   free(qk);
}
