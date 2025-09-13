#ifndef _CSR_MMIO_HIGHLEVEL_
#define _CSR_MMIO_HIGHLEVEL_

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
        //printf("Could not process Matrix Market banner.\n");
        //return -2;
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

    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report*2 * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report*2 * sizeof(int));
    VALUE_TYPE *csrVal_tmp    = (VALUE_TYPE *)malloc(nnz_mtx_report*2 * sizeof(VALUE_TYPE));
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
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = -fval;
        }
        else{
            csrRowPtr_counter[idxi]++;
            csrRowIdx_tmp[nnzr] = idxi;
            csrColIdx_tmp[nnzr] = idxj;
            csrVal_tmp[nnzr] = fval;
            nnzr++;
	    if(idxi!=idxj){
            	csrRowPtr_counter[idxj]++;
            	csrRowIdx_tmp[nnzr] = idxj;
            	csrColIdx_tmp[nnzr] = idxi;
            	csrVal_tmp[nnzr] = fval;
		nnzr++;
          }
        }
    }

    if (f != stdin)
        fclose(f);


    // exclusive scan for csrRowPtr_counter
    int old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];
        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric=isSymmetric_tmp; 
    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

// read matrix infomation from mtx file
int mmio_data(int *csrRowPtr, int *csrColIdx, VALUE_TYPE *csrVal, VALUE_TYPE *a_p, char *filename,double small)
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
        //printf("Could not process Matrix Market banner.\n");
        //return -2;
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

    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report*2 * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report*2 * sizeof(int));
    VALUE_TYPE *csrVal_tmp    = (VALUE_TYPE *)malloc(nnz_mtx_report*2 * sizeof(VALUE_TYPE));

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
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = -fval;
        }
        else{
            csrRowPtr_counter[idxi]++;
            csrRowIdx_tmp[nnzr] = idxi;
            csrColIdx_tmp[nnzr] = idxj;
            csrVal_tmp[nnzr] = fval;
            nnzr++;
	    if(idxi!=idxj){
            	csrRowPtr_counter[idxj]++;
            	csrRowIdx_tmp[nnzr] = idxj;
            	csrColIdx_tmp[nnzr] = idxi;
            	csrVal_tmp[nnzr] = fval;
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

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];
        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];
    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

        for (int i = 0; i < nnz_tmp; i++)
        {
            int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx[offset] = csrColIdx_tmp[i];
            csrVal[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

#endif
