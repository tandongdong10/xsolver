#include "HostMatrixCSR.h"
void HostMatrixCSR::CSRTOCSC(HostMatrix *hostmtxcsc){
    if(m!=n){
	printf("Matrix is not square matrix, can not be trans!!!\n");
	exit(0);
    }
#ifdef HAVE_MPI
    hostmtxcsc->MallocMatrix(n,nHalo,rowptr[m]);
#else
    hostmtxcsc->MallocMatrix(n,rowptr[m]+m);
#endif
    MKL_INT job[6];
    job[0] = 0;
    job[1] = rowptr[0];
    job[2] = 0;
    job[3] = 2;
    job[4] = 8;
    job[5] = 1;
    MKL_INT info=0;
    mkl_dcsrcsc(job,&m,val,colidx,rowptr,hostmtxcsc->getval(),hostmtxcsc->getidx(),hostmtxcsc->getptr(),&info);
}
void HostMatrixCSR::CSRTOELL(HostMatrix *hostmtxell){
    int num_nz=0;
    int nnz_ell=0;
    for(int i=0;i<m;i++){
        int tmp=rowptr[i+1]-rowptr[i];
        num_nz=(tmp>num_nz)?tmp:num_nz;
    }
    nnz_ell=num_nz*m;
#ifdef HAVE_MPI
    hostmtxell->MallocMatrix(n,nHalo,nnz_ell);
#else
    hostmtxell->MallocMatrix(n,nnz_ell);
#endif
    double *ell_val=hostmtxell->getval();
    int *ell_idx=hostmtxell->getidx();
    for(int i=0;i<m;i++){
	int st=rowptr[i],ed=rowptr[i+1];
	for(int j=i*num_nz;j<(i+1)*num_nz;j++){
		if(st<ed){
		    ell_val[j]=val[st];
		    ell_idx[j]=colidx[st];
		}
		else{
		    ell_val[j]=0;
		    ell_idx[j]=-1;
		}
           	st++;
	    }
	}
}
void HostMatrixCSR::update(double *diag_val_new,double *val_new){
    int num=0;
    int onebase=rowptr[0];
    for(int i=0;i<n;i++){
	for(int j=rowptr[i]-onebase;j<rowptr[i+1]-onebase;j++){
	    if(colidx[j]-onebase==i)
		val[j]=diag_val_new[i];
	    else
		val[j]=val_new[num++];
	}
    }
}
void HostMatrixCSR::getdiag(HostVector *a_p){
    int onebase=rowptr[0];
    for(int i=0;i<m;i++){
	for(int j=rowptr[i]-onebase;j<rowptr[i+1]-onebase;j++){
	    if(colidx[j]-onebase==i)
		a_p->val[i]=val[j];
	}
    }
}
void HostMatrixCSR::getdiag(double *a_p){
    int onebase=rowptr[0];
    for(int i=0;i<m;i++){
	for(int j=rowptr[i]-onebase;j<rowptr[i+1]-onebase;j++){
	    if(colidx[j]-onebase==i)
		a_p[i]=val[j];
	}
    }
}
void HostMatrixCSR::ToDiagMatrix(HostMatrix *hostmtxold)
{
    int M=n;
    int *offdiag_ptr_new=rowptr;
    int *offdiag_col_new=colidx;
    double *offdiag_val_new=val;
    int *offdiag_ptr=hostmtxold->getptr();
    int *offdiag_col=hostmtxold->getidx();
    double *offdiag_val=hostmtxold->getval();
    int i=0 ;//i=1,...,M
    int j=0;//j=1,...,nnz
    int col_new_idx= 0;
    int myid=0;
    int onebase=offdiag_ptr[0];
    offdiag_ptr_new[0] = onebase;
    int startidx=onebase;
    int endidx=M+startidx;
    for (i = 1; i <=M; i++)
    {
    	int EndOfRows = offdiag_ptr[i]-onebase;
    	int counts = 0;
    	for (j=offdiag_ptr[i-1]-onebase; j < EndOfRows;j++)
    	{
    	    if ((offdiag_col[j]) >= startidx && (offdiag_col[j]) < endidx){
    		offdiag_col_new[col_new_idx] = offdiag_col[j];// - startidx + onebase;
    		offdiag_val_new[col_new_idx] = offdiag_val[j];
    		col_new_idx ++;
    		counts ++;
    	    }
    	}
    	offdiag_ptr_new[i] = offdiag_ptr_new[i - 1] + counts;
    }
}
void HostMatrixCSR::SpMV(HostVector *x, HostVector *y)
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
}

void HostMatrixCSR::SpMM(HostMatrixCSR *B,HostMatrixCSR *C)
{	
    	char trA = 'N';
	int request = 0;
	int sort =1;
	int columns = n;
	if(n!=B->m)
	    printf("matrix BBi can't create!!!LBB column!=UBB row!!!\n");
	int nzmax = m*B->n;
	int info = 0;
	mkl_dcsrmultcsr(&trA, &request, &sort, &m, &n, &B->n, val, colidx, rowptr, B->val, B->colidx, B->rowptr, C->val, C->colidx, C->rowptr, &nzmax, &info);
	C->nnz=C->rowptr[m];
	//for(int i=0;i<m;i++){
	//    for(int j=C->rowptr[i]-1;j<C->rowptr[i+1]-1;j++)
	//	printf("%d %d %4.2f\n",i,C->colidx[j]-1,C->val[j]);
	//}
}
void HostMatrixCSR::bmAx(HostVector *rhs, HostVector *x, HostVector *y)
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
}
void HostMatrixCSR::seqilu0(HostMatrix *hstmtxL,HostMatrix *hstmtxU,double *&diag_val){
	double *bilu0= new double[nnz];
	int *u_ptr=new int[n];
  	ilu0_simple(bilu0, val, rowptr, colidx, n, u_ptr);

	int *lptr=hstmtxL->getptr();
	int *uptr=hstmtxU->getptr();
	lptr[0]=0;
	uptr[0]=0;
        for (int i = 0; i < n; i++) {
	   lptr[i+1]=lptr[i]+u_ptr[i]-rowptr[i];
	   uptr[i+1]=uptr[i]+rowptr[i+1]-u_ptr[i]-1;
	}
	diag_val=new double[n];
	double *bil0=hstmtxL->getval();
	double *biu0=hstmtxU->getval();
	int *ilcolidx=hstmtxL->getidx();
	int *iucolidx=hstmtxU->getidx();
	int il=0;
	int iu=0;
        for (int i = 0; i < n; i++) {
                for (int ii = rowptr[i]; ii < u_ptr[i]; ii++) {
                        bil0[il]= bilu0[ii];
			ilcolidx[il++]=colidx[ii];
		}
		diag_val[i]=1/bilu0[u_ptr[i]];
                for (int ii = u_ptr[i]+1; ii < rowptr[i+1]; ii++) {
                        biu0[iu]= bilu0[ii]*diag_val[i];
			iucolidx[iu++]=colidx[ii];
		}
	}
	delete []bilu0;
        delete []u_ptr;
}
void HostMatrixCSR::seqilu0_mkl(HostMatrix *hstmtxLU){
	double *bilu0=hstmtxLU->getval();
  	MKL_INT ipar[128];
  	double dpar[128];
  	ipar[30] = 1;
  	dpar[30] = 1.E-20;
  	dpar[31] = 1.E-16;
  	MKL_INT ierr = 0;
  	dcsrilu0(&n, val, rowptr, colidx, bilu0, ipar, dpar, &ierr);
  	if (ierr != 0)
    	{
      	    printf ("Preconditioner dcsrilu0 has returned the ERROR code %d\n", ierr);
    	}
}
void HostMatrixCSR::seqilut_mkl(HostMatrix *hstmtxLU,int maxfil,double ilut_tol){
	int *ibilut=hstmtxLU->getptr();
	int *jbilut=hstmtxLU->getidx();
	double *bilut=hstmtxLU->getval();
  	MKL_INT ipar[128];
  	double dpar[128];
  	ipar[30] = 1;
  	dpar[30] = ilut_tol;
  	//dpar[31] = 1.E-5;
  	MKL_INT ierr = 0;
  	dcsrilut(&n, val, rowptr, colidx, bilut, ibilut, jbilut, &ilut_tol, &maxfil, ipar, dpar, &ierr);
    	printf("ILUT nnz = %d\n", ibilut[n]);
  	if (ierr != 0)
    	{
      	    printf ("Preconditioner dcsrilut has returned the ERROR code %d\n", ierr);
    	}

}
void HostMatrixCSR::seqilut(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag,int maxfil,double ilut_tol){
        diag=new double[n];
        double *luvalue=new double[(2*maxfil+2)*n];
        int *lucols=new int[(2*maxfil+2)*n];
        int *lurows=new int[n+1];
        int *u_ptr=new int[n+1];
        ilutp_final<double>(luvalue,val,rowptr,colidx,n,
        diag,maxfil,lucols,lurows,u_ptr,\
        0,ilut_tol,NULL,NULL);
    	printf("ILUT nnz = %d\n", lurows[n]);
	int *lptr=hstmtxL->getptr();
	int *uptr=hstmtxU->getptr();
	lptr[0]=0;
	uptr[0]=0;
        for (int i = 0; i < n; i++) {
	   lptr[i+1]=lptr[i]+u_ptr[i]-lurows[i];
	   uptr[i+1]=uptr[i]+lurows[i+1]-u_ptr[i]-1;
	}
	double *bilt=hstmtxL->getval();
	double *biut=hstmtxU->getval();
	int *ilcolidx=hstmtxL->getidx();
	int *iucolidx=hstmtxU->getidx();
	int il=0;
	int iu=0;
        for (int i = 0; i < n; i++) {
                for (int ii = lurows[i]; ii < u_ptr[i]; ii++) {
                        bilt[il]= luvalue[ii];
			ilcolidx[il++]=lucols[ii];
		}
                for (int ii = u_ptr[i]+1; ii < lurows[i+1]; ii++) {
                        biut[iu]= luvalue[ii]*diag[i];
			iucolidx[iu++]=lucols[ii];
		}
	}
        if(u_ptr!=NULL){delete[]u_ptr;u_ptr=NULL;}
        if(luvalue!=NULL){delete[]luvalue;luvalue=NULL;}
        if(lucols!=NULL){delete[]lucols;lucols=NULL;}
        if(lurows!=NULL){delete[]lurows;lurows=NULL;}
}
void HostMatrixCSR::seqic0(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag_val,double smallnum, double droptol){
        diag_val=new double[n];
        double *luvalue=new double[rowptr[n]+n];
        int *l_ptr=new int[n+1];
        int *l_cols=new int[rowptr[n]+n];
        ic0_csr_A<double>(n,rowptr,val,colidx,l_ptr,luvalue,l_cols,smallnum,droptol);

	int *lptr=hstmtxL->getptr();
	int *uptr=hstmtxU->getptr();
	lptr[0]=0;
	uptr[0]=0;
        for (int i = 0; i < n; i++) {
	   lptr[i+1]=lptr[i]+l_ptr[i+1]-l_ptr[i]-1;
	}
	double *bil0=hstmtxL->getval();
	double *biu0=hstmtxU->getval();
	int *ilcolidx=hstmtxL->getidx();
	int *iucolidx=hstmtxU->getidx();
	memset(uptr,0,(n+1)*sizeof(int));
  	int il=0;
	for(int i=0;i<n;i++){
	    for(int ii=l_ptr[i];ii<l_ptr[i+1];ii++){
		int j=l_cols[ii];
		if(j==i){
		    diag_val[i]=luvalue[ii];
		}
		else{
		    uptr[j+1]++;
		}
	    }
	}
	int *ilcolptr=new int[n+1];
	ilcolptr[0]=0;
	for(int i=1;i<=n;i++){
	    uptr[i]+=uptr[i-1];
	    ilcolptr[i]=uptr[i];
	}
	for(int i=0;i<n;i++){
	    for(int ii=l_ptr[i];ii<l_ptr[i+1];ii++){
		int j=l_cols[ii];
		if(j==i){
		}
		else{
		    bil0[il]=luvalue[ii]*diag_val[j];
		    ilcolidx[il++]=j;
		    biu0[ilcolptr[j]]=luvalue[ii]*diag_val[j];
		    iucolidx[ilcolptr[j]++]=i;
		}
	    }
	}
	for(int i=0;i<n;i++)
	    diag_val[i]*=diag_val[i];
	//printf("nnz= %d %d %d\n",lptr[n],uptr[n],il);
	delete []ilcolptr;
        if(luvalue!=NULL){delete[]luvalue;luvalue=NULL;}
        if(l_ptr!=NULL){delete[]l_ptr;l_ptr=NULL;}
        if(l_cols!=NULL){delete[]l_cols;l_cols=NULL;}
}
void HostMatrixCSR::seqict(HostMatrix *hstmtxL,HostMatrix *hstmtxU, double *&diag_val,double smallnum, double droptol, int maxfil){
        double *l_luvalue=new double[maxfil*n+n];
        int *l_ptr_pattern=new int[n+1];
        int *l_cols_pattern=new int[maxfil*n+n];
        ict_csr_A_l<double>(n,rowptr,val,colidx,l_ptr_pattern,l_luvalue,l_cols_pattern,small_num,droptol, maxfil);
	int *lptr=hstmtxL->getptr();
	int *uptr=hstmtxU->getptr();
	lptr[0]=0;
	uptr[0]=0;
        for (int i = 0; i < n; i++) {
	   lptr[i+1]=lptr[i]+l_ptr_pattern[i+1]-l_ptr_pattern[i]-1;
	}
	double *bilt=hstmtxL->getval();
	double *biut=hstmtxU->getval();
	int *ilcolidx=hstmtxL->getidx();
	int *iucolidx=hstmtxU->getidx();
	diag_val=new double[n];
	memset(uptr,0,(n+1)*sizeof(int));
  	int il=0;
	for(int i=0;i<n;i++){
	    for(int ii=l_ptr_pattern[i];ii<l_ptr_pattern[i+1];ii++){
		int j=l_cols_pattern[ii];
		if(j==i){
		    diag_val[i]=l_luvalue[ii];
		}
		else{
		    uptr[j+1]++;
		}
	    }
	}
	int *ilcolptr=new int[n+1];
	ilcolptr[0]=0;
	for(int i=1;i<=n;i++){
	    uptr[i]+=uptr[i-1];
	    ilcolptr[i]=uptr[i];
	}
	for(int i=0;i<n;i++){
	    for(int ii=l_ptr_pattern[i];ii<l_ptr_pattern[i+1];ii++){
		int j=l_cols_pattern[ii];
		if(j==i){
		}
		else{
		    bilt[il]=l_luvalue[ii]*diag_val[j];
		    ilcolidx[il++]=j;
		    biut[ilcolptr[j]]=l_luvalue[ii]*diag_val[j];
		    iucolidx[ilcolptr[j]++]=i;
		}
	    }
	}
	for(int i=0;i<n;i++)
	    diag_val[i]*=diag_val[i];
	delete []ilcolptr;
        if(l_luvalue!=NULL){delete[]l_luvalue;l_luvalue=NULL;}
        if(l_ptr_pattern!=NULL){delete[]l_ptr_pattern;l_ptr_pattern=NULL;}
        if(l_cols_pattern!=NULL){delete[]l_cols_pattern;l_cols_pattern=NULL;}
        //printf("ict maxfil=%d, droptol=%lg, nnz=%d\n",maxfil, droptol, u_ptr[n_p]+1); 

}
void HostMatrixCSR::LUsolve(HostMatrix *mtxU, double *diag, HostVector *x_vec,HostVector *y_vec){
	double *x=x_vec->val;
	double *y=y_vec->val;
	for(int i=0;i<n;i++){
	    double sum=x[i];
	    for(int ii=rowptr[i];ii<rowptr[i+1];ii++){
		int j=colidx[ii];
		sum -=y[j]*val[ii];
	    }
	    y[i]=sum;
	}
	for(int i=0;i<n;i++)
	    y[i] *= diag[i];
	int *uptr=mtxU->getptr();
	int *iucolidx=mtxU->getidx();
	double *uval=mtxU->getval();
	for(int i=n-1;i>=0;i--){
	    double sum=y[i];
	    for(int ii=uptr[i+1]-1;ii>=uptr[i];ii--){
		int j=iucolidx[ii];
		sum -=y[j]*uval[ii];
	    }
	    y[i]=sum;
	}
}
void HostMatrixCSR::Lsolve(HostVector *x,HostVector *y){
      	char cvar1 = 'L';
      	char cvar = 'N';
      	char cvar2 = 'U';
      	mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &n, val, rowptr, colidx, x->val, y->val);
}
void HostMatrixCSR::Usolve(HostVector *x,HostVector *y){
      	char cvar1 = 'U';
      	char cvar = 'N';
      	char cvar2 = 'N';
      	mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &n, val, rowptr, colidx, x->val, y->val);
}
HostMatrix* set_matrix_csr(){
    return new HostMatrixCSR();
}
