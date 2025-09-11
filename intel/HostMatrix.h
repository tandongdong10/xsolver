#pragma once
#include "mkl.h"
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include "mpi.h"
#include "mpicpu.h"
#ifndef _HOSTMATRIX_H_
#define _HOSTMATRIX_H_

#ifdef HAVE_MPI
extern struct topology_c topo_c;
#endif

// Preconditioner
enum PRECON
{
    JACOBIGPU = 1,
    JACOBICPU,
    ILU0
} precon;

class HostMatrix
{
  public:
    int     nInterior;
    int     nHalo;
    int     nSizes;
    double *diag_val;
    double *offdiag_val;
    int    *offdiag_row_offset;
    int    *offdiag_col_index;
    int    *exchange_ptr;

    HostMatrix()
    {
        nInterior          = 0;
        nHalo              = 0;
        nSizes             = 0;
        diag_val           = NULL;
        offdiag_val        = NULL;
        offdiag_row_offset = NULL;
        offdiag_col_index  = NULL;
        exchange_ptr       = NULL;
    }

    HostMatrix(int &nInterior, int &nHalo, int &nSizes, double *diag_val, double *offdiag_val, int *offdiag_row_offset, int *offdiag_col_index)
        : nInterior(nInterior), nHalo(nHalo), nSizes(nSizes), diag_val(diag_val), offdiag_val(offdiag_val), offdiag_row_offset(offdiag_row_offset), offdiag_col_index(offdiag_col_index)
    {
    }

    void operator=(const HostMatrix &rhs)
    {
        nInterior          = rhs.nInterior;
        nHalo              = rhs.nHalo;
        nSizes             = rhs.nSizes;
        diag_val           = rhs.diag_val;    // new double[nInterior];
        offdiag_val        = rhs.offdiag_val; // new double[nSizes],
        offdiag_row_offset = rhs.offdiag_row_offset;
        offdiag_col_index  = rhs.offdiag_col_index;
        exchange_ptr       = rhs.exchange_ptr;
    }

    void update(double *diag_val_new, double *offdiag_val_new)
    {
        diag_val    = diag_val_new;
        offdiag_val = offdiag_val_new;
    }

    void SpMV(double *x, double *y); // y=Ax

    void bmAx(double *rhs, double *x, double *y);

    ~HostMatrix()
    {
    }
};

HostMatrix hostmtx;
void       HostMatrix::SpMV(double *x, double *y)
{
    int    icell, iside, colstart, colend, colin;
    double r;
#ifdef HAVE_MPI
    communicator_p2p(x);
    communicator_p2p_waitall();
#endif
    for (icell = 0; icell < nInterior; icell++)
    {
        r        = 0.0;
        colstart = offdiag_row_offset[icell];     // 0-based
        colend   = offdiag_row_offset[icell + 1]; // 0-based
        for (iside = colstart; iside < colend; iside++)
        {
            colin = offdiag_col_index[iside]; // 0-based
            r     = r + offdiag_val[iside] * x[colin];
        }
        y[icell] = r + diag_val[icell] * x[icell]; // diag
    }
}

void HostMatrix::bmAx(double *rhs, double *x, double *y)
{
    int    icell, iside, colstart, colend, colin;
    double r;
#ifdef HAVE_MPI
    communicator_p2p(x);
    communicator_p2p_waitall();
#endif
    for (icell = 0; icell < nInterior; icell++)
    {
        r        = rhs[icell];
        colstart = offdiag_row_offset[icell];     // 0-based
        colend   = offdiag_row_offset[icell + 1]; // 0-based
        for (iside = colstart; iside < colend; iside++)
        {
            colin = offdiag_col_index[iside]; // 0-based
            r     = r - offdiag_val[iside] * x[colin];
        }
        y[icell] = r - diag_val[icell] * x[icell]; // diag
    }
}

void jacobiInit(int nIntCells, const double *a_p, double *diag, const double small)
{
    // double tmp=0;
    // int icell=0;
    // #pragma omp parallel for
    for (int icell = 0; icell < nIntCells; icell++)
    {
        double tmp  = a_p[icell] + small;
        diag[icell] = 1.0 / tmp;
    }
}

void jacobi(int nIntCells, const double *diag, const double *x, double *y)
{
    // int icell=0;
    // #pragma omp parallel for
    for (int icell = 0; icell < nIntCells; icell++)
    {
        y[icell] = diag[icell] * x[icell];
    }
}
double *diag;
void    preconditionerInitCPU()
{
    int nIntCells = hostmtx.nInterior;
    if (precon == 2)
    {
        diag         = new double[nIntCells];
        double small = 1e-20;
        jacobiInit(nIntCells, hostmtx.diag_val, diag, small);
    }
}

void preconditionerUpdateCPU()
{
    int nIntCells = hostmtx.nInterior;
    if (precon == 2)
    {
        double small = 1e-20;
        jacobiInit(nIntCells, hostmtx.diag_val, diag, small);
    }
}

void preconditionerCPU(const double *a, double *b)
{
    if (precon == 2)
    {
        jacobi(hostmtx.nInterior, diag, a, b);
    }
}

void preconditionerFreeCPU()
{
    if (precon == 2)
    {
        delete[] diag;
    }
}

// template <typename T>
extern "C" void cpu_mat_setup(int &nInterior, int &nHalo, int &nSizes, double *diag_val, double *offdiag_val, int *offdiag_row_offset, int *offdiag_col_index)
{
    HostMatrix hmtx(nInterior, nHalo, nSizes, diag_val, offdiag_val, offdiag_row_offset, offdiag_col_index);
#ifdef HAVE_MPI
    hmtx.exchange_ptr = topo_c.exchange_ptr;
#endif
    hostmtx = hmtx;
    return;
}
extern "C" void cpu_mat_update(double *diag_val, double *offdiag_val)
{
    hostmtx.update(diag_val, offdiag_val);
    /// preconditionerUpdateCPU();
}
extern "C" void cpu_preconditioner_update(char *fmt)
{
    if (strcmp(fmt, "Jacobi") == 0)
    {
        if (precon != JACOBICPU) printf("Preconditioner is not matched!\n");
    }
    preconditionerUpdateCPU();
}
extern "C" void cpu_preconditioner_setup(char *fmt)
{
    // char *str_Jacobi="jacobi";
    if (strcmp(fmt, "Jacobi") == 0)
    {
        precon = JACOBICPU;
    }
    preconditionerInitCPU();
}
extern "C" void cpu_preconditioner_free(char *fmt)
{
    preconditionerFreeCPU();
}
#endif
