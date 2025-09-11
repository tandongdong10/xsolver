#include "mkl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_MPI
#include <mpi.h>
extern struct topology_c topo_c;
#endif

#include "HostMatrix.h"

// extern void communicator_sum(double &value);
// extern void communicator_sum(double *value,int n);
// extern void communicator_sum(float &value);
// extern void communicator_sum(float *value,int n);
// extern void communicator_sum(int &value);
// extern void communicator_sum(int *value,int n);

void applyplanerotationc(double &dx, double &dy, double csx, double snx);
void generateplanerotationc(double dx, double dy, double &csx, double &snx);
void updatec(double *phi, int k, double **H, double *s, double **V, int n);

// extern HostMatrix hostmtx;

extern "C" void cpu_cg_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{
    // left preconditioned cg method with Jacobi preconditioner
    int     nIntCells    = hostmtx.nInterior;
    int     nGhstCells   = hostmtx.nHalo;
    double *a_p          = hostmtx.diag_val;
    int    *exchange_ptr = hostmtx.exchange_ptr;

    double *pk  = new double[nIntCells + nGhstCells];
    double *phi = new double[nIntCells + nGhstCells];

    double *res = new double[nIntCells];
    double *Apk = new double[nIntCells];
    double *qk  = new double[nIntCells];

    double sigma, alpha, minalpha, taoo, tao, resl, rsm, beta, tmp;
    int    iiter;

    double small = 1e-20, one = 1.0, minone = -1.0;
    if (sol_init == NULL) sol_init = sol;
    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);

    hostmtx.bmAx(rhs, phi, res);
    // applying the preconditioner
    preconditionerCPU(res, res);

    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);
    communicator_sum(res0);
    cblas_dcopy(nIntCells, res, 1, pk, 1);
    taoo = res0;
    res0 = sqrt(res0);

    for (iiter = 0; iiter < maxiter; iiter++)
    {
        usediter = iiter;

        hostmtx.SpMV(pk, Apk);
        // applying the preconditioner
        preconditionerCPU(Apk, qk);
        sigma = cblas_ddot(nIntCells, pk, 1, qk, 1);
        communicator_sum(sigma);
        alpha    = taoo / (sigma + small);
        minalpha = minone * alpha;
        cblas_daxpy(nIntCells, alpha, pk, 1, phi, 1);
        cblas_daxpy(nIntCells, minalpha, qk, 1, res, 1);

        // check convergence
        resl = cblas_ddot(nIntCells, res, 1, res, 1);
        communicator_sum(resl);
        tao  = resl;
        resl = sqrt(resl);
        rsm  = resl / (res0 + small);
        if (resvec != NULL) resvec[iiter] = rsm;
        if (rsm < tol)
        {
            usediter = iiter;
            cblas_dcopy(nIntCells, phi, 1, sol, 1);
            delete[] pk;
            delete[] res;
            delete[] Apk;
            delete[] qk;
            return;
        }

        beta = tao / (taoo + small);
        taoo = tao;

        cblas_dscal(nIntCells, beta, pk, 1);
        cblas_daxpy(nIntCells, one, res, 1, pk, 1);
    }
    cblas_dcopy(nIntCells, phi, 1, sol, 1);
    delete[] pk;
    delete[] phi;
    delete[] res;
    delete[] Apk;
    delete[] qk;
}

extern "C" void cpu_icg_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{
    // left preconditioned improved cg method with Jacobi preconditioner
    int     nIntCells    = hostmtx.nInterior;
    int     nGhstCells   = hostmtx.nHalo;
    double *a_p          = hostmtx.diag_val;
    int    *exchange_ptr = hostmtx.exchange_ptr;

    double *pk    = new double[nIntCells + nGhstCells];
    double *phi   = new double[nIntCells + nGhstCells];
    double *res   = new double[nIntCells];
    double *Apk   = new double[nIntCells];
    double *qk    = new double[nIntCells];
    double *tmp_v = new double[3];

    double sigma, alpha, minalpha, taoo, tao, resl, rsm, beta, tmp, rho, theta;
    int    iiter;

    double small = 1e-20, one = 1.0, minone = -1.0;

    if (sol_init == NULL) sol_init = sol;
    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);

    hostmtx.bmAx(rhs, phi, res);
    // applying the preconditioner
    preconditionerCPU(res, res);

    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);

    communicator_sum(res0);
    cblas_dcopy(nIntCells, res, 1, pk, 1);
    taoo = res0;
    res0 = sqrt(res0);

    for (iiter = 0; iiter < maxiter; iiter++)
    {
        usediter = iiter;

        hostmtx.SpMV(pk, Apk);
        // applying the preconditioner
        preconditionerCPU(Apk, qk);
        tmp_v[0] = cblas_ddot(nIntCells, pk, 1, qk, 1);
        tmp_v[1] = cblas_ddot(nIntCells, res, 1, qk, 1);
        tmp_v[2] = cblas_ddot(nIntCells, qk, 1, qk, 1);
        communicator_sum(tmp_v, 3);
        sigma    = tmp_v[0];
        rho      = tmp_v[1];
        theta    = tmp_v[2];
        alpha    = taoo / (sigma + small);
        minalpha = minone * alpha;
        cblas_daxpy(nIntCells, alpha, pk, 1, phi, 1);
        cblas_daxpy(nIntCells, minalpha, qk, 1, res, 1);

        // check convergence
        resl = cblas_ddot(nIntCells, res, 1, res, 1);
        communicator_sum(resl);
        tao  = resl;
        resl = sqrt(resl);
        rsm  = resl / (res0 + small);
        if (resvec != NULL) resvec[iiter] = rsm;
        if (rsm < tol)
        {
            usediter = iiter;
            cblas_dcopy(nIntCells, phi, 1, sol, 1);
            delete[] pk;
            delete[] phi;
            delete[] res;
            delete[] Apk;
            delete[] qk;
            delete[] tmp_v;
            return;
        }

        tao  = taoo - 2.0 * alpha * rho + alpha * alpha * theta;
        beta = tao / (taoo + small);
        taoo = tao;

        cblas_dscal(nIntCells, beta, pk, 1);
        cblas_daxpy(nIntCells, one, res, 1, pk, 1);
    }
    cblas_dcopy(nIntCells, phi, 1, sol, 1);
    delete[] pk;
    delete[] phi;
    delete[] res;
    delete[] Apk;
    delete[] qk;
    delete[] tmp_v;
}

extern "C" void cpu_cgm_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{
    // left preconditioned cg method with Jacobi preconditioner
    int  nIntCells    = hostmtx.nInterior;
    int  nGhstCells   = hostmtx.nHalo;
    int *exchange_ptr = hostmtx.exchange_ptr;

    double *pk  = new double[nIntCells + nGhstCells];
    double *phi = new double[nIntCells + nGhstCells];
    double *res = new double[nIntCells];
    double *zk  = new double[nIntCells];
    double *Apk = new double[nIntCells];
    double *qk  = new double[nIntCells];
    double  sigma, alpha, minalpha, taoo, tao, resl, rsm, beta;
    int     iiter;
    // double small = 0, one = 1.0, minone = -1.0;
    double small = 1e-20, one = 1.0, minone = -1.0;
    if (sol_init == NULL) sol_init = sol;
    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);
    hostmtx.bmAx(rhs, phi, res);
    preconditionerCPU(res, zk);

    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);
    tao         = cblas_ddot(nIntCells, res, 1, zk, 1);

    communicator_sum(res0);
    communicator_sum(tao);
    cblas_dcopy(nIntCells, zk, 1, pk, 1);
    res0 = sqrt(res0);

    for (iiter = 0; iiter < maxiter; iiter++)
    {
        usediter = iiter;

        taoo = tao;

        hostmtx.SpMV(pk, Apk);
        sigma = cblas_ddot(nIntCells, pk, 1, Apk, 1);
        communicator_sum(sigma);
        alpha    = tao / (sigma + small);
        minalpha = minone * alpha;
        cblas_daxpy(nIntCells, alpha, pk, 1, phi, 1);
        cblas_daxpy(nIntCells, minalpha, Apk, 1, res, 1);
        // check convergence
        preconditionerCPU(res, zk);
        tao = cblas_ddot(nIntCells, res, 1, zk, 1);
        communicator_sum(tao);

        beta = tao / (taoo + small);

        cblas_dscal(nIntCells, beta, pk, 1);
        cblas_daxpy(nIntCells, one, zk, 1, pk, 1);
        resl = cblas_ddot(nIntCells, res, 1, res, 1);
        communicator_sum(resl);
        resl = sqrt(resl);
        rsm  = resl / (res0 + small);
        // resvec[iiter]=rsm;
        if (resvec != NULL) resvec[iiter] = rsm;
        // printf("cg iter=%d, rsm= % 4.5f\n", iiter, rsm);
        if (rsm < tol)
        {
            usediter = iiter;
            cblas_dcopy(nIntCells, phi, 1, sol, 1);
            delete[] pk;
            delete[] phi;
            delete[] res;
            delete[] Apk;
            delete[] qk;
            delete[] zk;
            return;
        }
    }
    cblas_dcopy(nIntCells, phi, 1, sol, 1);
    delete[] pk;
    delete[] phi;
    delete[] res;
    delete[] Apk;
    delete[] qk;
    delete[] zk;
}
extern "C" void cpu_icgm_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{

    // left preconditioned improved cg method with Jacobi preconditioner
    int     nIntCells    = hostmtx.nInterior;
    int     nGhstCells   = hostmtx.nHalo;
    int    *exchange_ptr = hostmtx.exchange_ptr;
    double *pk           = new double[nIntCells + nGhstCells];
    double *phi          = new double[nIntCells + nGhstCells];
    double *zk           = new double[nIntCells];

    double *res   = new double[nIntCells];
    double *Apk   = new double[nIntCells];
    double *qk    = new double[nIntCells];
    double *tmp_v = new double[5];

    double sigma, alpha, minalpha, tauo, tau, resl, rsm, beta, rho, theta, fai;
    int    iiter;

    // double small = 0, one = 1.0, minone = -1.0;
    double small = 1e-20, one = 1.0, minone = -1.0;

    if (sol_init == NULL) sol_init = sol;

    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);
    hostmtx.bmAx(rhs, phi, res);
    preconditionerCPU(res, zk);

    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);
    tau         = cblas_ddot(nIntCells, res, 1, zk, 1);

    communicator_sum(res0);
    communicator_sum(tau);
    cblas_dcopy(nIntCells, zk, 1, pk, 1);
    res0 = sqrt(res0);

    for (iiter = 0; iiter < maxiter; iiter++)
    {
        usediter = iiter;
        hostmtx.SpMV(pk, Apk);
        preconditionerCPU(Apk, qk);
        tmp_v[0] = cblas_ddot(nIntCells, pk, 1, Apk, 1);
        tmp_v[1] = cblas_ddot(nIntCells, res, 1, qk, 1);
        tmp_v[2] = cblas_ddot(nIntCells, Apk, 1, zk, 1);
        tmp_v[3] = cblas_ddot(nIntCells, qk, 1, Apk, 1);
        tmp_v[4] = cblas_ddot(nIntCells, res, 1, res, 1);
        communicator_sum(tmp_v, 5);
        sigma = tmp_v[0];
        rho   = tmp_v[1]; // omega
        fai   = tmp_v[2];
        theta = tmp_v[3]; // delta

        alpha    = tau / (sigma + small);
        minalpha = minone * alpha;
        cblas_daxpy(nIntCells, alpha, pk, 1, phi, 1);
        cblas_daxpy(nIntCells, minalpha, Apk, 1, res, 1);
        preconditionerCPU(res, zk);
        tauo = tau;
        tau  = tau - alpha * fai - alpha * rho + alpha * alpha * theta;
        beta = tau / (tauo + small);

        cblas_dscal(nIntCells, beta, pk, 1);
        cblas_daxpy(nIntCells, one, zk, 1, pk, 1);
        // check convergence
        resl = tmp_v[4];
        resl = sqrt(resl);
        rsm  = resl / (res0 + small);
        if (resvec != NULL) resvec[iiter] = rsm;
        if (rsm < tol)
        {
            usediter = iiter;
            cblas_dcopy(nIntCells, phi, 1, sol, 1);
            delete[] pk;
            delete[] phi;
            delete[] zk;
            delete[] res;
            delete[] Apk;
            delete[] qk;
            delete[] tmp_v;
            return;
        }
    }
    cblas_dcopy(nIntCells, phi, 1, sol, 1);
    delete[] pk;
    delete[] phi;
    delete[] zk;
    delete[] res;
    delete[] Apk;
    delete[] qk;
    delete[] tmp_v;
}

extern "C" void cpu_bicgstab_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{
    // right preconditioned classical BICGStab method with Jacobi preconditioner

    int     nIntCells  = hostmtx.nInterior;
    int     nGhstCells = hostmtx.nHalo;
    double *a_p        = hostmtx.diag_val;
    // int *exchange_ptr=hostmtx.exchange_ptr;

    double *zk  = new double[nIntCells + nGhstCells];
    double *phi = new double[nIntCells + nGhstCells];

    double *res   = new double[nIntCells];
    double *pk    = new double[nIntCells];
    double *uk    = new double[nIntCells];
    double *vk    = new double[nIntCells];
    double *sk    = new double[nIntCells];
    double *reso  = new double[nIntCells];
    double *tmp_v = new double[2];

    double alpha, minalpha, omega, resl, rsm, beta, beto, gama, mingama, tmp;
    int    iiter;

    double small = 1e-20, one = 1.0, minone = -1.0;

    if (sol_init == NULL) sol_init = sol;

    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);
    hostmtx.bmAx(rhs, phi, res);
    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);
    communicator_sum(res0);
    res0 = sqrt(res0);
    cblas_dcopy(nIntCells, res, 1, reso, 1);
    memset(uk, 0, nIntCells * sizeof(double));
    memset(vk, 0, nIntCells * sizeof(double));
    memset(zk, 0, (nIntCells + nGhstCells) * sizeof(double));
    memset(pk, 0, nIntCells * sizeof(double));
    memset(sk, 0, nIntCells * sizeof(double));
    alpha = 1.0;
    beto  = 1.0;
    gama  = 1.0;

    for (iiter = 0; iiter < maxiter; iiter++)
    {
        usediter = iiter;

        beta = cblas_ddot(nIntCells, res, 1, reso, 1);
        communicator_sum(beta);
        omega = (beta * gama) / (alpha * beto + small);
        beto  = beta;

        minalpha = minone * alpha;
        cblas_daxpy(nIntCells, minalpha, uk, 1, pk, 1);
        cblas_dscal(nIntCells, omega, pk, 1);
        cblas_daxpy(nIntCells, one, res, 1, pk, 1);

        // applying the preconditioner
        preconditionerCPU(pk, zk);
        hostmtx.SpMV(zk, uk);

        tmp = cblas_ddot(nIntCells, uk, 1, reso, 1);
        communicator_sum(tmp);
        gama    = beta / (tmp + small);
        mingama = minone * gama;
        cblas_dcopy(nIntCells, res, 1, sk, 1);
        cblas_daxpy(nIntCells, mingama, uk, 1, sk, 1);

        // applying the preconditioner
        preconditionerCPU(sk, zk);
        hostmtx.SpMV(zk, vk);

        tmp_v[0] = cblas_ddot(nIntCells, vk, 1, sk, 1);
        tmp_v[1] = cblas_ddot(nIntCells, vk, 1, vk, 1);
        communicator_sum(&tmp_v[0], 2);
        alpha = tmp_v[0] / (tmp_v[1] + small);

        cblas_daxpy(nIntCells, gama, pk, 1, phi, 1);
        cblas_daxpy(nIntCells, alpha, sk, 1, phi, 1);

        minalpha = minone * alpha;
        cblas_dcopy(nIntCells, sk, 1, res, 1);
        cblas_daxpy(nIntCells, minalpha, vk, 1, res, 1);

        // check convergence
        resl = cblas_ddot(nIntCells, res, 1, res, 1);
        communicator_sum(resl);
        resl = sqrt(resl);
        rsm  = resl / (res0 + small);
        if (resvec != NULL) resvec[iiter] = rsm;
        if (rsm < tol)
        {
            usediter = iiter;
            // applying the preconditioner
            preconditionerCPU(phi, sol);
            delete[] zk;
            delete[] phi;
            delete[] res;
            delete[] pk;
            delete[] uk;
            delete[] vk;
            delete[] sk;
            delete[] reso;
            delete[] tmp_v;
            return;
        }
    }
    // applying the preconditioner
    preconditionerCPU(phi, sol);
    delete[] zk;
    delete[] phi;
    delete[] res;
    delete[] pk;
    delete[] uk;
    delete[] vk;
    delete[] sk;
    delete[] reso;
    delete[] tmp_v;
}

extern "C" void cpu_ibicgstab_solve(double *rhs, double *sol, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{
    // right preconditioned improved BICGStab method with Jacobi preconditioner
    int     nIntCells  = hostmtx.nInterior;
    int     nGhstCells = hostmtx.nHalo;
    double *a_p        = hostmtx.diag_val;

    double *vkm  = new double[nIntCells + nGhstCells];
    double *tkm  = new double[nIntCells + nGhstCells];
    double *resm = new double[nIntCells + nGhstCells];
    double *phi  = new double[nIntCells + nGhstCells];

    double *res   = new double[nIntCells];
    double *tk    = new double[nIntCells];
    double *pk    = new double[nIntCells];
    double *qk    = new double[nIntCells];
    double *zk    = new double[nIntCells];
    double *uk    = new double[nIntCells];
    double *vk    = new double[nIntCells];
    double *vko   = new double[nIntCells];
    double *sk    = new double[nIntCells];
    double *reso  = new double[nIntCells];
    double *tmp_v = new double[6];

    double pi, tau, fai, sigma, alpha, minalpha, alphao, rou, rouo, omega, minomega;
    double beta, delta, mindelta, theta, kappa, mu, nu;
    double resl, rsm, tmp, tmp1, tmp2, mintmp2;
    int    iiter;

    double small = 1e-20, one = 1.0, minone = -1.0, zero = 0.0;

    if (sol_init == NULL) sol_init = sol;
    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);

    hostmtx.bmAx(rhs, phi, res);

    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);

    communicator_sum(res0);
    res0 = sqrt(res0);

    cblas_dcopy(nIntCells, res, 1, reso, 1);
    memset(qk, 0, nIntCells * sizeof(double));
    memset(vk, 0, nIntCells * sizeof(double));
    memset(zk, 0, nIntCells * sizeof(double));
    alpha  = 1.0;
    alphao = 1.0;
    rou    = 1.0;
    rouo   = 1.0;
    omega  = 1.0;
    pi     = 0.0;
    tau    = 0.0;
    mu     = 0.0;

    tmp1 = cblas_ddot(nIntCells, phi, 1, phi, 1);
    communicator_sum(tmp1);
    tmp1 = sqrt(tmp1);
    if (tmp1 == zero)
    {
        // tk = A*res
        // applying the preconditioner
        preconditionerCPU(res, resm);
        hostmtx.SpMV(resm, tk);
        memset(pk, 0, nIntCells * sizeof(double));
    }
    else
    {
        // pk = A*A*phi
        hostmtx.SpMV(phi, pk);
        // applying the preconditioner
        preconditionerCPU(pk, resm);
        hostmtx.SpMV(resm, pk);
        // tk = pk + A*res
        // applying the preconditioner
        preconditionerCPU(res, resm);
        hostmtx.SpMV(resm, tk);
        cblas_daxpy(nIntCells, one, pk, 1, tk, 1);
    }

    tmp_v[0] = cblas_ddot(nIntCells, res, 1, res, 1);
    tmp_v[1] = cblas_ddot(nIntCells, res, 1, tk, 1);
    communicator_sum(tmp_v, 2);
    fai   = tmp_v[0];
    sigma = tmp_v[1];

    for (iiter = 0; iiter < maxiter; iiter++)
    {
        usediter = iiter;

        rouo   = rou;
        rou    = fai - omega * mu;
        delta  = (rou * alpha) / (rouo + small);
        beta   = delta / (omega + small);
        tau    = sigma + beta * tau - delta * pi;
        alphao = alpha;
        alpha  = rou / (tau + small);

        cblas_dcopy(nIntCells, vk, 1, vko, 1);
        cblas_dcopy(nIntCells, tk, 1, vk, 1);
        minomega = minone * omega;
        cblas_daxpy(nIntCells, minomega, pk, 1, vk, 1);
        cblas_daxpy(nIntCells, beta, vko, 1, vk, 1);
        mindelta = minone * delta;
        cblas_daxpy(nIntCells, mindelta, qk, 1, vk, 1);

        // applying the preconditioner
        preconditionerCPU(vk, vkm);
        hostmtx.SpMV(vkm, qk);

        cblas_dcopy(nIntCells, res, 1, sk, 1);
        minalpha = minone * alpha;
        cblas_daxpy(nIntCells, minalpha, vk, 1, sk, 1);

        minomega = minone * omega;
        cblas_daxpy(nIntCells, minomega, pk, 1, tk, 1);
        minalpha = minone * alpha;
        cblas_daxpy(nIntCells, minalpha, qk, 1, tk, 1);

        tmp1    = (beta * alpha) / (alphao + small);
        tmp2    = alpha * delta;
        mintmp2 = minone * tmp2;
        cblas_dscal(nIntCells, tmp1, zk, 1);
        cblas_daxpy(nIntCells, alpha, res, 1, zk, 1);
        cblas_daxpy(nIntCells, mintmp2, vko, 1, zk, 1);

        // applying the preconditioner
        preconditionerCPU(tk, tkm);
        hostmtx.SpMV(tkm, pk);

        tmp_v[0] = cblas_ddot(nIntCells, reso, 1, sk, 1);
        tmp_v[1] = cblas_ddot(nIntCells, reso, 1, qk, 1);
        tmp_v[2] = cblas_ddot(nIntCells, sk, 1, tk, 1);
        tmp_v[3] = cblas_ddot(nIntCells, tk, 1, tk, 1);
        tmp_v[4] = cblas_ddot(nIntCells, reso, 1, tk, 1);
        tmp_v[5] = cblas_ddot(nIntCells, reso, 1, pk, 1);
        communicator_sum(tmp_v, 6);

        fai   = tmp_v[0];
        pi    = tmp_v[1];
        theta = tmp_v[2];
        kappa = tmp_v[3];
        mu    = tmp_v[4];
        nu    = tmp_v[5];
        omega = theta / (kappa + small);
        sigma = mu - omega * nu;

        cblas_dcopy(nIntCells, sk, 1, res, 1);
        minomega = minone * omega;
        cblas_daxpy(nIntCells, minomega, tk, 1, res, 1);

        cblas_daxpy(nIntCells, one, zk, 1, phi, 1);
        cblas_daxpy(nIntCells, omega, sk, 1, phi, 1);

        // check convergence
        resl = cblas_ddot(nIntCells, res, 1, res, 1);
        communicator_sum(resl);
        resl = sqrt(resl);
        rsm  = resl / (res0 + small);
        if (resvec != NULL) resvec[iiter] = rsm;
        if (rsm < tol)
        {
            usediter = iiter;
            // applying the preconditioner
            preconditionerCPU(phi, sol);
            delete[] vkm;
            delete[] tkm;
            delete[] resm;
            delete[] phi;
            delete[] res;
            delete[] tk;
            delete[] pk;
            delete[] qk;
            delete[] zk;
            delete[] uk;
            delete[] vk;
            delete[] vko;
            delete[] sk;
            delete[] reso;
            delete[] tmp_v;
            return;
        }
    }
    // applying the preconditioner
    preconditionerCPU(phi, sol);
    delete[] vkm;
    delete[] tkm;
    delete[] resm;
    delete[] phi;
    delete[] res;
    delete[] tk;
    delete[] pk;
    delete[] qk;
    delete[] zk;
    delete[] uk;
    delete[] vk;
    delete[] vko;
    delete[] sk;
    delete[] reso;
    delete[] tmp_v;
}

extern "C" void cpu_gcr_solve(double *rhs, double *sol, int &restart, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{

    // left preconditioned GCR method with Jacobi preconditioner
    // here the classical Gram-Schmidt scheme is utilized

    int     nIntCells  = hostmtx.nInterior;
    int     nGhstCells = hostmtx.nHalo;
    double *a_p        = hostmtx.diag_val;

    double **uk = new double *[restart];
    double **ck = new double *[restart];
    for (int i = 0; i < restart; i++)
    {
        uk[i] = new double[nIntCells + nGhstCells];
        ck[i] = new double[nIntCells];
    }

    double *res   = new double[nIntCells];
    double *phi   = new double[nIntCells + nGhstCells];
    double *tmp_v = new double[2];
    double *alpha = new double[restart];

    double beta, error, rnorm, tmp, mintmp;
    int    iiter, i, j, k, m;

    double small = 1e-20, one = 1.0, minone = -1.0, zero = 0.0;

    beta = 0.0;
    if (sol_init == NULL) sol_init = sol;
    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);

    for (int i = 0; i < restart; i++)
    {
        memset(uk[i], 0, (nIntCells + nGhstCells) * sizeof(double));
        memset(ck[i], 0, nIntCells * sizeof(double));
    }
    memset(alpha, 0, restart * sizeof(double));
    hostmtx.bmAx(rhs, phi, res);
    // applying the preconditioner
    preconditionerCPU(res, res);

    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);
    communicator_sum(res0);
    res0  = sqrt(res0);
    error = res0;

    iiter = 0;
    while (iiter < maxiter)
    {
        k = 0;
        while (k < restart && iiter < maxiter)
        {
            // printf("k = %d\n",k);
            cblas_dcopy(nIntCells, res, 1, uk[k], 1);
            hostmtx.SpMV(uk[k], ck[k]);
            // applying the preconditioner
            preconditionerCPU(ck[k], ck[k]);
            for (i = 0; i < k; i++)
            {
                alpha[i] = cblas_ddot(nIntCells, ck[i], 1, ck[k], 1);
            }
            communicator_sum(alpha, k);
            for (i = 0; i < k; i++)
            {
                tmp = minone * alpha[i];
                cblas_daxpy(nIntCells, tmp, ck[i], 1, ck[k], 1);
                cblas_daxpy(nIntCells + nGhstCells, tmp, uk[i], 1, uk[k], 1);
            }

            tmp_v[0] = cblas_ddot(nIntCells, ck[k], 1, ck[k], 1);
            tmp_v[1] = cblas_ddot(nIntCells, ck[k], 1, res, 1);
            // communicator_sum(tmp_v[1]);
            // communicator_sum(tmp_v[2]);
            communicator_sum(tmp_v, 2);
            tmp_v[0] = sqrt(tmp_v[0]);
            tmp      = one / (tmp_v[0] + small);
            cblas_dscal(nIntCells + nGhstCells, tmp, uk[k], 1);
            cblas_dscal(nIntCells, tmp, ck[k], 1);

            tmp    = tmp_v[1] / (tmp_v[0] + small);
            mintmp = minone * tmp;
            cblas_daxpy(nIntCells + nGhstCells, tmp, uk[k], 1, phi, 1);
            cblas_daxpy(nIntCells, mintmp, ck[k], 1, res, 1);

            // check convergence
            tmp = cblas_ddot(nIntCells, res, 1, res, 1);
            communicator_sum(tmp);
            rnorm = sqrt(tmp);
            error = rnorm / (res0 + small);
            if (resvec != NULL) resvec[iiter] = error;
            if (error < tol)
            {
                usediter = iiter;
                cblas_dcopy(nIntCells, phi, 1, sol, 1);
                delete[] res;
                delete[] phi;
                delete[] tmp_v;
                delete[] alpha;
                for (int i = 0; i < restart; i++)
                {
                    delete[] uk[i];
                    delete[] ck[i];
                }
                delete[] uk;
                delete[] ck;
                return;
            }
            k     = k + 1;
            iiter = iiter + 1;
        }
    }
    usediter = iiter;
    cblas_dcopy(nIntCells, phi, 1, sol, 1);
    delete[] res;
    delete[] phi;
    delete[] tmp_v;
    delete[] alpha;
    for (int i = 0; i < restart; i++)
    {
        delete[] uk[i];
        delete[] ck[i];
    }
    delete[] uk;
    delete[] ck;
}

extern "C" void cpu_igcr_solve(double *rhs, double *sol, int &restart, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{

    // left preconditioned GCR method with Jacobi preconditioner
    // here the modified Gram-Schmidt scheme is utilized

    int     nIntCells  = hostmtx.nInterior;
    int     nGhstCells = hostmtx.nHalo;
    double *a_p        = hostmtx.diag_val;

    double **uk = new double *[restart];
    double **ck = new double *[restart];
    for (int i = 0; i < restart; i++)
    {
        uk[i] = new double[nIntCells + nGhstCells];
        ck[i] = new double[nIntCells];
    }

    double *res   = new double[nIntCells];
    double *phi   = new double[nIntCells + nGhstCells];
    double *tmp_v = new double[2];

    double beta, alpha, minalpha, error, rnorm, tmp, mintmp;
    int    iiter, i, j, k, m;

    double small = 1e-20, one = 1.0, minone = -1.0, zero = 0.0;

    beta  = 0.0;
    alpha = 0.0;

    if (sol_init == NULL) sol_init = sol;
    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);

    for (int i = 0; i < restart; i++)
    {
        memset(uk[i], 0, (nIntCells + nGhstCells) * sizeof(double));
        memset(ck[i], 0, nIntCells * sizeof(double));
    }
    hostmtx.bmAx(rhs, phi, res);
    // applying the preconditioner
    preconditionerCPU(res, res);

    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);
    communicator_sum(res0);
    res0  = sqrt(res0);
    error = res0;

    iiter = 0;
    while (iiter < maxiter)
    {
        k = 0;
        while (k < restart && iiter < maxiter)
        {
            cblas_dcopy(nIntCells, res, 1, uk[k], 1);
            hostmtx.SpMV(uk[k], ck[k]);
            // applying the preconditioner
            preconditionerCPU(ck[k], ck[k]);
            for (i = 0; i < k; i++)
            {
                alpha = cblas_ddot(nIntCells, ck[i], 1, ck[k], 1);
                communicator_sum(alpha);
                minalpha = minone * alpha;
                cblas_daxpy(nIntCells, minalpha, ck[i], 1, ck[k], 1);
                cblas_daxpy(nIntCells + nGhstCells, minalpha, uk[i], 1, uk[k], 1);
            }

            tmp_v[0] = cblas_ddot(nIntCells, ck[k], 1, ck[k], 1);
            tmp_v[1] = cblas_ddot(nIntCells, ck[k], 1, res, 1);
            // communicator_sum(tmp_v[1]);
            // communicator_sum(tmp_v[2]);
            communicator_sum(tmp_v, 2);
            tmp_v[0] = sqrt(tmp_v[0]);
            tmp      = one / (tmp_v[0] + small);
            cblas_dscal(nIntCells + nGhstCells, tmp, uk[k], 1);
            cblas_dscal(nIntCells, tmp, ck[k], 1);

            tmp    = tmp_v[1] / (tmp_v[0] + small);
            mintmp = minone * tmp;
            cblas_daxpy(nIntCells + nGhstCells, tmp, uk[k], 1, phi, 1);
            cblas_daxpy(nIntCells, mintmp, ck[k], 1, res, 1);

            // check convergence
            tmp = cblas_ddot(nIntCells, res, 1, res, 1);
            communicator_sum(tmp);
            rnorm = sqrt(tmp);
            error = rnorm / (res0 + small);
            if (resvec != NULL) resvec[iiter] = error;
            if (error < tol)
            {
                usediter = iiter;
                cblas_dcopy(nIntCells, phi, 1, sol, 1);
                delete[] res;
                delete[] phi;
                delete[] tmp_v;
                for (int i = 0; i < restart; i++)
                {
                    delete[] uk[i];
                    delete[] ck[i];
                }
                delete[] uk;
                delete[] ck;
                return;
            }
            k     = k + 1;
            iiter = iiter + 1;
        }
    }
    usediter = iiter;
    cblas_dcopy(nIntCells, phi, 1, sol, 1);
    delete[] res;
    delete[] phi;
    delete[] tmp_v;
    for (int i = 0; i < restart; i++)
    {
        delete[] uk[i];
        delete[] ck[i];
    }
    delete[] uk;
    delete[] ck;
}

extern "C" void cpu_gmres_solve(double *rhs, double *sol, int &restart, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{

    // left preconditioned GMRES method with Jacobi preconditioner
    // here the modified Gram-Schmidt scheme is utilized

    int     nIntCells  = hostmtx.nInterior;
    int     nGhstCells = hostmtx.nHalo;
    double *a_p        = hostmtx.diag_val;

    double **V = new double *[restart + 1];
    // double **Z = new double*[restart+1];
    double **H = new double *[restart + 1];
    for (int i = 0; i < restart + 1; i++)
    {
        V[i] = new double[nIntCells + nGhstCells];
        //   Z[i] = new double[nIntCells+nGhstCells];
        H[i] = new double[restart];
    }

    double *phi  = new double[nIntCells + nGhstCells];
    double *res  = new double[nIntCells];
    double *w    = new double[nIntCells];
    double *tmpv = new double[nIntCells];
    double *s    = new double[restart + 1];
    double *cs   = new double[restart + 1];
    double *sn   = new double[restart + 1];

    double beta, error, tmp, mintmp, reso;
    int    iiter, i, j, k, m;

    double small = 1e-20, one = 1.0, minone = -1.0, zero = 0.0;

    beta = 0.0;
    m    = restart;

    if (sol_init == NULL) sol_init = sol;
    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);

    for (int i = 0; i < restart + 1; i++)
    {
        memset(V[i], 0, (nIntCells + nGhstCells) * sizeof(double));
        memset(H[i], 0, restart * sizeof(double));
    }
    memset(s, 0, (restart + 1) * sizeof(double));
    memset(cs, 0, (restart + 1) * sizeof(double));
    memset(sn, 0, (restart + 1) * sizeof(double));
    hostmtx.bmAx(rhs, phi, res);
    // applying the preconditioner
    preconditionerCPU(res, res);

    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);
    communicator_sum(res0);
    res0  = sqrt(res0);
    error = res0 / (res0 + small);
    beta  = res0;

    j = 0;
    while (j < maxiter)
    {
        for (int n = 0; n < restart + 1; n++)
        {
            memset(V[n], 0, (nIntCells + nGhstCells) * sizeof(double));
            memset(H[n], 0, restart * sizeof(double));
        }
        memset(s, 0, (restart + 1) * sizeof(double));
        memset(cs, 0, (restart + 1) * sizeof(double));
        memset(sn, 0, (restart + 1) * sizeof(double));

        tmp = one / (beta + small);
        cblas_dscal(nIntCells, tmp, res, 1);
        cblas_dcopy(nIntCells, res, 1, V[0], 1);
        s[0] = beta;

        i = 0;
        while (i < restart && j < maxiter)
        {
            hostmtx.SpMV(V[i], w);
            // applying the preconditioner
            preconditionerCPU(w, w);
            for (k = 0; k <= i; k++)
            {
                tmp = cblas_ddot(nIntCells, w, 1, V[k], 1);
                communicator_sum(tmp);
                H[k][i] = tmp;
                mintmp  = minone * tmp;
                cblas_daxpy(nIntCells, mintmp, V[k], 1, w, 1);
            }

            tmp = cblas_ddot(nIntCells, w, 1, w, 1);
            communicator_sum(tmp);
            tmp         = sqrt(tmp);
            H[i + 1][i] = tmp;
            tmp         = one / (tmp + small);
            cblas_dscal(nIntCells, tmp, w, 1);
            cblas_dcopy(nIntCells, w, 1, V[i + 1], 1);

            for (k = 0; k < i; k++)
            {
                applyplanerotationc(H[k][i], H[k + 1][i], cs[k], sn[k]);
            }
            generateplanerotationc(H[i][i], H[i + 1][i], cs[i], sn[i]);
            applyplanerotationc(H[i][i], H[i + 1][i], cs[i], sn[i]);
            applyplanerotationc(s[i], s[i + 1], cs[i], sn[i]);

            // check convergence
            error = fabs(s[i + 1]) / (res0 + small);
            if (resvec != NULL) resvec[i] = error;
            if (error < tol)
            {
                updatec(phi, i, H, s, V, nIntCells + nGhstCells);
                usediter = j;
                cblas_dcopy(nIntCells, phi, 1, sol, 1);
                delete[] res;
                delete[] phi;
                delete[] w;
                delete[] tmpv;
                delete[] s;
                delete[] cs;
                delete[] sn;
                for (int i = 0; i < restart + 1; i++)
                {
                    delete[] V[i];
                    delete[] H[i];
                }
                delete[] V;
                delete[] H;
                return;
            }
            i = i + 1;
            j = j + 1;
        }
        updatec(phi, m, H, s, V, nIntCells + nGhstCells);
        hostmtx.bmAx(rhs, phi, res);
        // applying the preconditioner
        preconditionerCPU(res, res);
        beta     = fabs(s[m]);
        usediter = j;
    }
    cblas_dcopy(nIntCells, phi, 1, sol, 1);
    delete[] res;
    delete[] phi;
    delete[] w;
    delete[] tmpv;
    delete[] s;
    delete[] cs;
    delete[] sn;
    for (int i = 0; i < restart + 1; i++)
    {
        delete[] V[i];
        delete[] H[i];
    }
    delete[] V;
    delete[] H;
}

extern "C" void cpu_fgmres_solve(double *rhs, double *sol, int &restart, double &tol, int &maxiter, double *sol_init, int &usediter, double *resvec = NULL)
{

    // right preconditioned FGMRES method with Jacobi preconditioner
    // only right preconditioning is allowed in the FGMRES method
    // here the modified Gram-Schmidt scheme is utilized

    int     nIntCells  = hostmtx.nInterior;
    int     nGhstCells = hostmtx.nHalo;
    double *a_p        = hostmtx.diag_val;

    double **V = new double *[restart + 1];
    double **Z = new double *[restart + 1];
    double **H = new double *[restart + 1];
    for (int i = 0; i < restart + 1; i++)
    {
        V[i] = new double[nIntCells + nGhstCells];
        Z[i] = new double[nIntCells + nGhstCells];
        H[i] = new double[restart];
    }

    double *phi  = new double[nIntCells + nGhstCells];
    double *res  = new double[nIntCells];
    double *w    = new double[nIntCells];
    double *tmpv = new double[nIntCells];
    double *s    = new double[restart + 1];
    double *cs   = new double[restart + 1];
    double *sn   = new double[restart + 1];

    double beta, error, tmp, mintmp, reso;
    int    iiter, i, j, k, m;

    double small = 1e-20, one = 1.0, minone = -1.0, zero = 0.0;

    beta = 0.0;
    m    = restart;
    if (sol_init == NULL) sol_init = sol;
    cblas_dcopy(nIntCells, sol_init, 1, phi, 1);

    for (int i = 0; i < restart + 1; i++)
    {
        memset(V[i], 0, (nIntCells + nGhstCells) * sizeof(double));
        memset(H[i], 0, restart * sizeof(double));
        memset(Z[i], 0, (nIntCells + nGhstCells) * sizeof(double));
    }
    memset(s, 0, (restart + 1) * sizeof(double));
    memset(cs, 0, (restart + 1) * sizeof(double));
    memset(sn, 0, (restart + 1) * sizeof(double));

    hostmtx.bmAx(rhs, phi, res);
    double res0 = cblas_ddot(nIntCells, res, 1, res, 1);
    communicator_sum(res0);
    res0  = sqrt(res0);
    error = res0 / (res0 + small);
    beta  = res0;

    j = 0;
    while (j < maxiter)
    {
        for (int n = 0; n < restart + 1; n++)
        {
            memset(V[n], 0, (nIntCells + nGhstCells) * sizeof(double));
            memset(H[n], 0, restart * sizeof(double));
            memset(Z[n], 0, (nIntCells + nGhstCells) * sizeof(double));
        }
        memset(s, 0, (restart + 1) * sizeof(double));
        memset(cs, 0, (restart + 1) * sizeof(double));
        memset(sn, 0, (restart + 1) * sizeof(double));

        tmp = one / (beta + small);
        cblas_dscal(nIntCells, tmp, res, 1);
        cblas_dcopy(nIntCells, res, 1, V[0], 1);
        s[0] = beta;

        i = 0;
        while (i < restart && j < maxiter)
        {
            // applying the preconditioner
            preconditionerCPU(V[i], Z[i]);
            hostmtx.SpMV(Z[i], w);
            for (k = 0; k <= i; k++)
            {
                tmp = cblas_ddot(nIntCells, w, 1, V[k], 1);
                communicator_sum(tmp);
                H[k][i] = tmp;
                mintmp  = minone * tmp;
                cblas_daxpy(nIntCells, mintmp, V[k], 1, w, 1);
            }

            tmp = cblas_ddot(nIntCells, w, 1, w, 1);
            communicator_sum(tmp);
            tmp         = sqrt(tmp);
            H[i + 1][i] = tmp;
            tmp         = one / (tmp + small);
            cblas_dscal(nIntCells, tmp, w, 1);
            cblas_dcopy(nIntCells, w, 1, V[i + 1], 1);

            for (k = 0; k < i; k++)
            {
                applyplanerotationc(H[k][i], H[k + 1][i], cs[k], sn[k]);
            }
            generateplanerotationc(H[i][i], H[i + 1][i], cs[i], sn[i]);
            applyplanerotationc(H[i][i], H[i + 1][i], cs[i], sn[i]);
            applyplanerotationc(s[i], s[i + 1], cs[i], sn[i]);

            // check convergence
            error = fabs(s[i + 1]) / (res0 + small);
            if (resvec != NULL) resvec[i] = error;
            if (error < tol)
            {
                updatec(phi, i, H, s, Z, nIntCells + nGhstCells);
                usediter = j;
                // preconditionerCPU(phi, sol);
                cblas_dcopy(nIntCells, phi, 1, sol, 1);
                delete[] phi;
                delete[] res;
                delete[] w;
                delete[] tmpv;
                delete[] s;
                delete[] cs;
                delete[] sn;
                for (int i = 0; i < restart + 1; i++)
                {
                    delete[] V[i];
                    delete[] Z[i];
                    delete[] H[i];
                }
                delete[] V;
                delete[] Z;
                delete[] H;
                return;
            }
            i = i + 1;
            j = j + 1;
        }
        updatec(phi, m, H, s, Z, nIntCells + nGhstCells);
        hostmtx.bmAx(rhs, phi, res);
        beta     = fabs(s[m]);
        usediter = j;
    }
    // preconditionerCPU(phi, sol);
    cblas_dcopy(nIntCells, phi, 1, sol, 1);
    delete[] phi;
    delete[] res;
    delete[] w;
    delete[] tmpv;
    delete[] s;
    delete[] cs;
    delete[] sn;
    for (int i = 0; i < restart + 1; i++)
    {
        delete[] V[i];
        delete[] Z[i];
        delete[] H[i];
    }
    delete[] V;
    delete[] Z;
    delete[] H;
}

void applyplanerotationc(double &dx, double &dy, double csx, double snx)
{
    double tmp_scalar;

    tmp_scalar = csx * dx + snx * dy;
    dy         = -snx * dx + csx * dy;
    dx         = tmp_scalar;
}

void generateplanerotationc(double dx, double dy, double &csx, double &snx)
{
    double tmp_scalar, zero = 0.0, one = 1.0;

    if (dy == zero)
    {
        csx = one;
        snx = zero;
    }
    else if (fabs(dy) > fabs(dx))
    {
        tmp_scalar = dx / dy;
        snx        = one / sqrt(one + tmp_scalar * tmp_scalar);
        csx        = tmp_scalar * snx;
    }
    else
    {
        tmp_scalar = dy / dx;
        csx        = one / sqrt(one + tmp_scalar * tmp_scalar);
        snx        = tmp_scalar * csx;
    }
}

void updatec(double *phi, int k, double **H, double *s, double **V, int n)
{
    double *y = new double[k];
    int     i, j;
    double  small = 1e-20;

    cblas_dcopy(k, s, 1, y, 1);
    for (i = k - 1; i >= 0; i--)
    {
        y[i] = y[i] / (H[i][i] + small);
        for (j = i - 1; j >= 0; j--)
        {
            y[j] = y[j] - H[j][i] * y[i];
        }
    }

    for (i = 0; i < k; i++)
    {
        cblas_daxpy(n, y[i], V[i], 1, phi, 1);
    }
}
