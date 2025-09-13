#ifndef _MATRIXCSCSPARSE_H_
#define _MATRIXCSCSPARSE_H_
void sparsematrixcsc_diag(HostMatrix *hstin, HostMatrix *hstout){
    int nnznew=0;
    double t=3*1e-3;
    double *A=hstin->getval();
    int *ia=hstin->getptr();
    int *ja=hstin->getidx();
    int nIntCells=hstin->n;
    int nnz=ia[nIntCells];
    hstout->MallocMatrix(nIntCells,nnz);
    double *Anew=hstout->getval();
    int *ianew=hstout->getptr();
    int *janew=hstout->getidx();
    double *diag_val=new double[nIntCells];
    ianew[0]=0;
    #pragma omp parallel for
    for(int i=0;i<nIntCells;i++){
	for(int j=ia[i];j<ia[i+1];j++){
  	    if(ja[j]==i){
		diag_val[i]=A[j];
		break;
	    }
	}
    }
    for(int i=0;i<nIntCells;i++){
	for(int j=ia[i];j<ia[i+1];j++){
	    if(fabs(A[j])>t*fabs(diag_val[i])||fabs(A[j])>t*fabs(diag_val[ja[j]])){
		Anew[nnznew]=A[j];
		janew[nnznew]=ja[j];
		nnznew++;
	    }
	}
	ianew[i+1]=nnznew;
    }	    
    hstout->nnz=nnznew;
    delete []diag_val;
}
void sparsematrixcsc_diag_parallel(HostMatrix *hstin, HostMatrix *hstout){
    int nnznew=0;
    double t=3*1e-3;
    double *A=hstin->getval();
    int *ia=hstin->getptr();
    int *ja=hstin->getidx();
    int nIntCells=hstin->n;
    int nnz=ia[nIntCells];
    hstout->MallocMatrix(nIntCells,nnz);
    double *Anew=hstout->getval();
    int *ianew=hstout->getptr();
    int *janew=hstout->getidx();
    double *diag_val=new double[nIntCells];
    ianew[0]=0;
    memset(ianew,0,(nIntCells+1)*sizeof(int));
    #pragma omp parallel for
    for(int i=0;i<nIntCells;i++){
	for(int j=ia[i];j<ia[i+1];j++){
  	    if(ja[j]==i){
		diag_val[i]=A[j];
		break;
	    }
	}
    }
    #pragma omp parallel for
    for(int i=0;i<nIntCells;i++){
	for(int j=ia[i];j<ia[i+1];j++){
	    if(fabs(A[j])>t*fabs(diag_val[i])||fabs(A[j])>t*fabs(diag_val[ja[j]])){
		ianew[i+1]++;
	    }	
	}
    }
    for(int i=0;i<nIntCells;i++)
 	ianew[i+1]+=ianew[i];
    #pragma omp parallel for
    for(int i=0;i<nIntCells;i++){
	int nnzi=ianew[i];
	double otherdata=0;
	for(int j=ia[i];j<ia[i+1];j++){
	    if(fabs(A[j])>t*fabs(diag_val[i])||fabs(A[j])>t*fabs(diag_val[ja[j]])){
		Anew[nnzi]=A[j];
		janew[nnzi]=ja[j];
		nnzi++;
	    }
	    else
		otherdata+=A[j];
	}
	//printf("col %d otherdata=%lg\n",i,otherdata);
    }	    
    hstout->nnz=ianew[nIntCells];
    delete []diag_val;
}
#endif
