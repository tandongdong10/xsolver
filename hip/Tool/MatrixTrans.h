#ifndef _MATRIX_TRANSFORM_
#define _MATRIX_TRANSFORM_
#include "quick_sort.h"
#include "../HostMatrix/HostMatrix.h"
#ifdef HAVE_MPI
extern struct topology_c topo_c;
extern void communicator_allgather(int num_send,int *receivebuf);
#endif
void based0To1Matrix(HostMatrix *hostmtx)
{
	int M=hostmtx->n;
	int *offdiag_ptr=hostmtx->getptr();
	int *offdiag_col=hostmtx->getidx();
	int i = 0;//i=1,...,M
	int j = 0;//j=1,...,nnz

	int nnz = offdiag_ptr[M];
	for (j = 0; j < nnz; j++)
	{
		offdiag_col[j]++;
	}
	for (i = 0; i <= M; i++) {
		offdiag_ptr[i]++;
	}
}
//void ToCSRMatrix(const int*diag_val, const int*offdiag_ptr, const int*offdiag_col, const double*offdiag_val, int*ptr, int* col, double*val,const int M) 
//{
//
//	int col_idx = 0;
//	int nnz = offdiag_ptr[M];
//	int ptr_idx = 1;
//	int i = 1;//i=1,...,M;
//	int j = 0;//j=1,...,nnz
//	ptr[0] = 1;
//	while (j<nnz) {
//		if (j+1== offdiag_ptr[i-1]&&(i-1)<M) {
//			ptr[ptr_idx] = offdiag_ptr[i] -offdiag_ptr[i-1]+ptr[ptr_idx-1]+1;
//			val[col_idx] =diag_val[i-1];
//			if (i == 5) {
//				i = i + 1 - 1;
//			}
//			col[col_idx] = i;
//			i++;
//			ptr_idx++;
//		}
//		else {
//			val[col_idx] = offdiag_val[j];
//			col[col_idx] = offdiag_col[j];
//			j++;
//		}
//		col_idx++;
//	
//	}
//
//}
void ToCSRMatrix(const double*diag_val, HostMatrix *hostmtx, HostMatrix *hostmtx_new)
{
	int M=hostmtx->n;
	int *offdiag_ptr=hostmtx->getptr();
	int *offdiag_col=hostmtx->getidx();
	double *offdiag_val=hostmtx->getval();
	int *ptr=hostmtx_new->getptr();
	int *col=hostmtx_new->getidx();
	double *val=hostmtx_new->getval();
	int col_idx = 0;
	int i = 1;//i=1,...,M;
	int j = 0;//j=1,...,nnz
	ptr[0] = 1;
	int NumOfValues;
	int EndOfRows;
	for (i = 1; i <= M; i++) {
		NumOfValues= offdiag_ptr[i] - offdiag_ptr[i - 1];
		EndOfRows = j + NumOfValues;
		ptr[i] = ptr[i - 1] + NumOfValues;
		
		if (diag_val[i - 1] != 0) {
			col[col_idx] = i ;
			val[col_idx] = diag_val[i - 1];
			col_idx += 1;
			ptr[i] += 1;
		}
		for (; j < EndOfRows; j++) {
			col[col_idx] = offdiag_col[j];
			val[col_idx] = offdiag_val[j];
			col_idx += 1;

		}
	}
}



void permuteMatrix(HostMatrix *hostmtx) 
{
	int *ptr=hostmtx->getptr();
	int *col=hostmtx->getidx();
	double *val=hostmtx->getval();
	int M=hostmtx->n;
	int i = 1;//i=1,...,M;
	int onebase=ptr[0];
    //#pragma omp parallel for
	for(i=1;i<=M;i++)
	{
		int EndOfRows = ptr[i]-onebase;

		//sort on col[j,EndOfRows]-->col,val[j,EndOfRows];

		for (int ii = ptr[i-1]-onebase; ii < EndOfRows; ii++) {
			for (int jj = ii; jj < EndOfRows; jj++) {
				if (col[ii] > col[jj])
				{
					int temp = col[ii];
					col[ii] = col[jj];
					col[jj] = temp;
					double temp2 = val[ii];
					val[ii] = val[jj];
					val[jj] = temp2;
				}
			}
		}

	}

}
void quicksortMatrix(HostMatrix *hostmtx) {
	int *ptr=hostmtx->getptr();
	int *col=hostmtx->getidx();
	double *val=hostmtx->getval();
	int M=hostmtx->n;
	
	int i = 1;//i=1,...,M;
	int NumOfValues;
    //#pragma omp parallel for
	for(i=1;i<=M;i++)
	{
		NumOfValues = ptr[i] - ptr[i - 1];
		quick_sort_key_val_pair(col+ptr[i-1]-1, val+ptr[i-1]-1, NumOfValues);
	}

}
void quicksortMatrix(const int *ptr,int *col, double *val,const int M) {
	int i = 1;//i=1,...,M;
	int NumOfValues;
    //#pragma omp parallel for
	for(i=1;i<=M;i++)
	{
		NumOfValues = ptr[i] - ptr[i - 1];
		quick_sort_key_val_pair(col+ptr[i-1]-1, val+ptr[i-1]-1, NumOfValues);
	}
}
#endif // !trans.h
