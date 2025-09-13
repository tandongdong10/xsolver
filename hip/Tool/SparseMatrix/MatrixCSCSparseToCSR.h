#ifndef _MATRIXCSCSPARSETOCSR_H_
#define _MATRIXCSCSPARSETOCSR_H_
void sparsematrixcsc_csr(HostMatrix *hstin, HostMatrix *hstout, HostMatrix *hstcsr){
    int nnznew=0;
    double t=3*1e-3;
    double *A=hstin->getval();
    int *ia=hstin->getptr();
    int *ja=hstin->getidx();
    int nIntCells=hstin->n;
    int nnz=ia[nIntCells];
    hstout->MallocMatrix(nIntCells,nnz);
    hstcsr->MallocMatrix(nIntCells,nnz);
    double *Anew=hstout->getval();
    int *ianew=hstout->getptr();
    int *janew=hstout->getidx();
    double *csrA=hstcsr->getval();
    int *rowptr=hstcsr->getptr();
    int *colidx=hstcsr->getidx();
    memset(rowptr, 0, (nIntCells+1)*sizeof(int));
    double *diag_val=new double[nIntCells];
    ianew[0]=0;
    for(int i=0;i<nIntCells;i++){
	for(int j=ia[i];j<ia[i+1];j++){
  	    if(ja[j]==i)
		diag_val[i]=A[j];
	    int k=ja[j]+1;
	    rowptr[k]+=1;
	}
    }
    for(int i=0;i<nIntCells;i++)
	rowptr[i+1]+=rowptr[i];	
    for(int i=0;i<nIntCells;i++){
	for(int j=ia[i];j<ia[i+1];j++){
            int k = ja[j]; 
            int next = rowptr[k]; 
            csrA[next] = A[j]; 
            colidx[next] = i;
            rowptr[k]++;
	    if(fabs(A[j])>t*fabs(diag_val[i])||fabs(A[j])>t*fabs(diag_val[ja[j]])){
		Anew[nnznew]=A[j];
		janew[nnznew]=ja[j];
		nnznew++;
	    }
	}
	ianew[i+1]=nnznew;
    }	    
    for( int i = nIntCells; i >= 0 ; i-- ) { 
        rowptr[i+1] = rowptr[i];
    }
    rowptr[0]=0;
    hstcsr->nnz=rowptr[nIntCells];
    hstout->nnz=nnznew;
    delete []diag_val;
}
#endif
