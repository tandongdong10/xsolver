#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

#include "mmio.h"
#include "common.h"
// read matrix infomation from mtx file
int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, char *filename)
{
    int m_tmp, n_tmp, nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isSymmetric_tmp = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }
    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;
    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }
    int *cscColPtr_counter = (int *)malloc((n_tmp+1) * sizeof(int));
    memset(cscColPtr_counter, 0, (n_tmp+1) * sizeof(int));

    int *cscRowIdx_tmp = (int *)malloc(nnz_mtx_report*2 * sizeof(int));
    int *cscColIdx_tmp = (int *)malloc(nnz_mtx_report*2 * sizeof(int));
    double *cscVal_tmp    = (double *)malloc(nnz_mtx_report*2 * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    int nnzr=0;
    int i=0;
    for (i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        if(!isSymmetric_tmp){
        cscColPtr_counter[idxj]++;
        cscRowIdx_tmp[i] = idxi;
        cscColIdx_tmp[i] = idxj;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cscVal_tmp[i] = fval;
        }
        else{
            cscColPtr_counter[idxj]++;
            cscRowIdx_tmp[nnzr] = idxi;
            cscColIdx_tmp[nnzr] = idxj;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            cscVal_tmp[nnzr] = fval;
            nnzr++;
	    if(idxi!=idxj){
                cscColPtr_counter[idxi]++;
                cscRowIdx_tmp[nnzr] = idxj;
                cscColIdx_tmp[nnzr] = idxi;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                cscVal_tmp[nnzr] = fval;
		nnzr++;
          }
        }
    }

    if (f != stdin)
        fclose(f);

    // exclusive scan for csrRowPtr_counter
    int old_val, new_val;

    old_val = cscColPtr_counter[0];
    cscColPtr_counter[0] = 0;
    for (i = 1; i <= n_tmp; i++)
    {
        new_val = cscColPtr_counter[i];
        cscColPtr_counter[i] = old_val + cscColPtr_counter[i-1];
        old_val = new_val;
    }

    nnz_tmp = cscColPtr_counter[n_tmp];

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    // free tmp space
    free(cscRowIdx_tmp);
    free(cscVal_tmp);
    free(cscColIdx_tmp);
    free(cscColPtr_counter);

    return 0;
}

// read matrix infomation from mtx file
int mmio_data(int *cscColPtr, int *cscRowIdx, double *cscVal, double *a_p, char *filename,double small)
{
    int m_tmp, n_tmp, nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isSymmetric_tmp = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }
    int *cscColPtr_counter = (int *)malloc((n_tmp+1) * sizeof(int));
    memset(cscColPtr_counter, 0, (n_tmp+1) * sizeof(int));

    int *cscColIdx_tmp = (int *)malloc(2*nnz_mtx_report * sizeof(int));
    int *cscRowIdx_tmp = (int *)malloc(2*nnz_mtx_report * sizeof(int));
    double *cscVal_tmp    = (double *)malloc(2*nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    int nnzr=0;
    int i=0;
    for (i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        if(!isSymmetric_tmp){
        cscColPtr_counter[idxj]++;
        cscRowIdx_tmp[i] = idxi;
        cscColIdx_tmp[i] = idxj;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cscVal_tmp[i] = fval;
        }
        else{
            cscColPtr_counter[idxj]++;
            cscRowIdx_tmp[nnzr] = idxi;
            cscColIdx_tmp[nnzr] = idxj;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            cscVal_tmp[nnzr] = fval;
            nnzr++;
	    if(idxi!=idxj){
                cscColPtr_counter[idxi]++;
                cscRowIdx_tmp[nnzr] = idxj;
                cscColIdx_tmp[nnzr] = idxi;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                cscVal_tmp[nnzr] = fval;
		nnzr++;
          }
        }
	if(idxi==idxj)
	    a_p[idxi]=fval;
    }

    if (f != stdin)
        fclose(f);

    // exclusive scan for csrRowPtr_counter
    int old_val, new_val;

    old_val = cscColPtr_counter[0];
    cscColPtr_counter[0] = 0;
    for (i = 1; i <= n_tmp; i++)
    {
        new_val = cscColPtr_counter[i];
        cscColPtr_counter[i] = old_val + cscColPtr_counter[i-1];
        old_val = new_val;
    }
    nnz_tmp = cscColPtr_counter[n_tmp];
    memcpy(cscColPtr, cscColPtr_counter, (n_tmp+1) * sizeof(int));
    memset(cscColPtr_counter, 0, (n_tmp+1) * sizeof(int));

        for (i = 0; i < nnz_tmp; i++)
        {
            int offset = cscColPtr[cscColIdx_tmp[i]] + cscColPtr_counter[cscColIdx_tmp[i]];
            cscRowIdx[offset] = cscRowIdx_tmp[i];
            cscVal[offset] =cscVal_tmp[i];
            cscColPtr_counter[cscColIdx_tmp[i]]++;
        }

    // free tmp space
    free(cscRowIdx_tmp);
    free(cscVal_tmp);
    free(cscColIdx_tmp);
    free(cscColPtr_counter);

    return 0;
}

int mmio_rhs(int mA, double *rhs, char *filename)
{
    FILE *f;
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;
    int i;
    char line[MM_MAX_LINE_LENGTH];
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MM_PREMATURE_EOF;
    //if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
    //    storage_scheme) != 5)
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MM_PREMATURE_EOF;
    for (i = 0; i < mA; i++)
    {
        double fval=0;
        int returnvalue;

        returnvalue = fscanf(f, "%lg ", &fval);
        rhs[i]=fval;
    }
    return 0;
}
#endif
