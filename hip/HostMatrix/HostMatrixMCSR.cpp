#include "HostMatrixMCSR.h"
void HostMatrixMCSR::MCSRTOCSR(HostMatrix *hostmtxcsr){
#ifdef HAVE_MPI
	hostmtxcsr->MallocMatrix(m,nHalo,rowptr[m]+m);
#else
	hostmtxcsr->MallocMatrix(m,rowptr[m]+m);
#endif
	int *csrptr=hostmtxcsr->getptr();
	int *csrcol=hostmtxcsr->getidx();
	double *csrval=hostmtxcsr->getval();
	int col_idx = 0;
	int ptr_idx = 1;
	int i = 1;//i=1,...,n;
	int j = 0;//j=1,...,nnz
	csrptr[0] = 1;
	int NumOfValues;
	int EndOfRows;
	for (i = 1; i <= m; i++) {
		NumOfValues= rowptr[i] - rowptr[i - 1];
		EndOfRows = j + NumOfValues;
		csrptr[i] = csrptr[i - 1] + NumOfValues;
		
		if (diag_val[i - 1] != 0) {
			csrcol[col_idx] = i ;
			csrval[col_idx] = diag_val[i - 1];
			col_idx += 1;
			csrptr[i] += 1;
		}
		for (; j < EndOfRows; j++) {
			csrcol[col_idx] = colidx[j];
			csrval[col_idx] = val[j];
			col_idx += 1;

		}
	}
	hostmtxcsr->nnz=csrptr[n]-csrptr[0];
	permuteMatrix(hostmtxcsr);
}
void HostMatrixMCSR::MCSRTOCSR(double *diag_val_in,HostMatrix *hostmtxcsr){
#ifdef HAVE_MPI
	hostmtxcsr->MallocMatrix(m,nHalo,rowptr[m]+m);
#else
	hostmtxcsr->MallocMatrix(m,rowptr[m]+m);
#endif
	int *csrptr=hostmtxcsr->getptr();
	int *csrcol=hostmtxcsr->getidx();
	double *csrval=hostmtxcsr->getval();
	int col_idx = 0;
	int ptr_idx = 1;
	int i = 1;//i=1,...,n;
	int j = 0;//j=1,...,nnz
	csrptr[0] = 1;
	int NumOfValues;
	int EndOfRows;
	for (i = 1; i <= n; i++) {
		NumOfValues= rowptr[i] - rowptr[i - 1];
		EndOfRows = j + NumOfValues;
		csrptr[i] = csrptr[i - 1] + NumOfValues;
		
		if (diag_val_in[i - 1] != 0) {
			csrcol[col_idx] = i ;
			csrval[col_idx] = diag_val_in[i - 1];
			col_idx += 1;
			csrptr[i] += 1;
		}
		for (; j < EndOfRows; j++) {
			csrcol[col_idx] = colidx[j];
			csrval[col_idx] = val[j];
			col_idx += 1;

		}
	}
	hostmtxcsr->nnz=csrptr[n]-csrptr[0];
	permuteMatrix(hostmtxcsr);
}
void HostMatrixMCSR::SpMV(HostVector *x, HostVector *y)
{
    double one=1,zero=0;
    bool zerobased=false;
    if(rowptr[0]==0)
	zerobased=true;
#ifdef HAVE_MPI
    communicator_p2p(x->val);
    communicator_p2p_waitall();
#endif
    if(zerobased)
    	mkl_dcsrmv ( "N", &m, &n, &one, "G**C" , val , colidx, rowptr, rowptr+1, x->val, &zero, y->val);
    else
    	mkl_dcsrmv ( "N", &m, &n, &one, "G**F" , val , colidx, rowptr, rowptr+1, x->val, &zero, y->val);
    for(int i=0;i<m;i++)
	y->val[i] += diag_val[i]*(x->val[i]);
}

void HostMatrixMCSR::bmAx(HostVector *rhs, HostVector *x, HostVector *y)
{
    double minone=-1,one=1;
    cblas_dcopy(n,rhs->val,1,y->val,1);
    bool zerobased=false;
    if(rowptr[0]==0)
	zerobased=true;
#ifdef HAVE_MPI
    communicator_p2p(x->val);
    communicator_p2p_waitall();
#endif
    if(zerobased)
    	mkl_dcsrmv ( "N", &m, &n, &minone, "G**C" , val , colidx, rowptr, rowptr+1, x->val, &one, y->val);
    else
    	mkl_dcsrmv ( "N", &m, &n, &minone, "G**F" , val , colidx, rowptr, rowptr+1, x->val, &one, y->val);
    for(int i=0;i<m;i++)
	y->val[i] -= diag_val[i]*(x->val[i]);
}
HostMatrix* set_matrix_mcsr(){
    return new HostMatrixMCSR();
}
