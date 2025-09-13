#include "HostMatrixCSC.h"
void HostMatrixCSC::getdiag(double *a_p){
    int onebase=colptr[0];
    for(int i=0;i<n;i++){
    	for(int j=colptr[i]-onebase;j<colptr[i+1]-onebase;j++){
	    if(rowidx[j]-onebase==i)
	    	a_p[i]=val[j];
	}
    }
}
void HostMatrixCSC::CSCTOCSR(HostMatrix *hostmtxcsr){
    if(m!=n){
        printf("Matrix is not square matrix, can not be trans!!!\n");
        exit(0);
    }
#ifdef HAVE_MPI
    hostmtxcsr->MallocMatrix(m,nHalo,colptr[m]);
#else
    hostmtxcsr->MallocMatrix(m,colptr[m]);
#endif
    MKL_INT job[6];
    job[0] = 1;
    job[1] = 0;
    job[2] = colptr[0];
    job[3] = 2;
    job[4] = 8;
    job[5] = 1;
    MKL_INT info=0;
    mkl_dcsrcsc(job,&n,hostmtxcsr->getval(),hostmtxcsr->getidx(),hostmtxcsr->getptr(),val,rowidx,colptr,&info);
}
void HostMatrixCSC::SpMV(HostVector *x, HostVector *y)
{
    double one=1,zero=0;
    bool zerobased=false;
    if(colptr[0]==0)
	zerobased=true;
    if(zerobased)
    	mkl_dcscmv ( "N", &m, &n, &one, "G**C" , val , rowidx, colptr, colptr+1, x->val, &zero, y->val);
	//csc_mult_vec_mannal(colptr,n,rowidx,val,x->val,m,y->val);
    else
    	mkl_dcscmv ( "N", &m, &n, &one, "G**F" , val , rowidx, colptr, colptr+1, x->val, &zero, y->val);
}
void HostMatrixCSC::bmAx(HostVector *rhs, HostVector *x, HostVector *y)
{
    double minone=-1,one=1;
    cblas_dcopy(n,rhs->val,1,y->val,1);
    bool zerobased=false;
    if(colptr[0]==0)
	zerobased=true;
    if(zerobased)
    	mkl_dcscmv ( "N", &m, &n, &minone, "G**C" , val , rowidx, colptr, colptr+1, x->val, &one, y->val);
    else
    	mkl_dcscmv ( "N", &m, &n, &minone, "G**F" , val , rowidx, colptr, colptr+1, x->val, &one, y->val);
}
HostMatrix* set_matrix_csc(){
    return new HostMatrixCSC();
}
