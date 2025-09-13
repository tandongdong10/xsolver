#ifndef _MATRIXCSRSPARSE_H_
#define _MATRIXCSRSPARSE_H_
void sparsematrixcsr(HostMatrixCSR *hstin, HostMatrixCSR *hstout){
    int nnznew=0;
    double t=1e-3;
    double *A=hstin->getval();
    int *ia=hstin->getptr();
    int *ja=hstin->getidx();
    int nIntCells=hstin->n;
    int nnz=ia[nIntCells];
    hstout->MallocMatrix(nIntCells,nIntCells,nnz);
    double *Anew=hstout->getval();
    int *ianew=hstout->getptr();
    int *janew=hstout->getidx();
    double *diag_val=new double[nIntCells];
    ianew[0]=1;
    for(int i=0;i<nIntCells;i++){
	for(int j=ia[i]-1;j<ia[i+1]-1;j++){
  	    if(ja[j]-1==i)
		diag_val[i]=A[j];
	}
    }
    for(int i=0;i<nIntCells;i++){
	for(int j=ia[i]-1;j<ia[i+1]-1;j++){
	    if(fabs(A[j])>t*fabs(diag_val[i])||fabs(A[j])>t*fabs(diag_val[ja[j]-1])){
		Anew[nnznew]=A[j];
		janew[nnznew]=ja[j];
		nnznew++;
	    }
	}
	ianew[i+1]=nnznew+1;
    }	    
    hstout->nnz=nnznew;
    delete []diag_val;
}
#endif
