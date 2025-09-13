#include "DeviceMatrixCSR.h"

__global__ void getCSRdiag (const int d_nIntCells, const int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double* diag)
{
    int icell = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
    if (icell < d_nIntCells) {
    for (int iside = d_NbCell_ptr_c[icell]-onebase; iside < d_NbCell_ptr_c[icell+1]-onebase; iside++)
    {
	if(d_NbCell_s[iside]-onebase==icell)
	    diag[icell]=d_a_l[iside];
    }
    }
}
__global__ void LMultCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *d_y)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    //diag[icell]=1;
    double r = 0.0;
    for (int iside = d_NbCell_ptr_c[icell]-onebase; iside < d_NbCell_ptr_c[icell+1]-onebase; iside++)
    {
	if(d_NbCell_s[iside]-onebase==icell){
		//diag[icell]=1/d_a_l[iside];
	}
	else
        	r += d_a_l[iside] * d_x[d_NbCell_s[iside]-onebase];
    }
    d_y[icell] = r;
    }
}
__global__ void lsolveiterCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *tmp, double *d_y)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) 
	d_y[icell]=(d_x[icell]-tmp[icell]);	
}
__global__ void UMultCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *d_y)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    //diag[icell]=1;
    double r = 0.0;
    for (int iside = d_NbCell_ptr_c[icell]-onebase; iside < d_NbCell_ptr_c[icell+1]-onebase; iside++)
    {
	if(d_NbCell_s[iside]-onebase==icell){
	}
	else
        	r += d_a_l[iside] * d_x[d_NbCell_s[iside]-onebase];
    }
    d_y[icell] = r;
    }
}
__global__ void usolveiterCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *tmp, double *d_y, double *diagu)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) 
	d_y[icell]=diagu[icell]*(d_x[icell]-tmp[icell]);	
}
__global__ void matMultCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double*d_x, double *d_y)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    double r = 0.0;
    for (int iside = d_NbCell_ptr_c[icell]-onebase; iside < d_NbCell_ptr_c[icell+1]-onebase; iside++)
    {
        r += d_a_l[iside] * d_x[d_NbCell_s[iside]-onebase];
    }
    d_y[icell] = r;
    }
    
}
__global__ void bmAxCSR (const int d_nIntCells, int onebase, const int* d_NbCell_ptr_c,const int* d_NbCell_s, const double*d_a_l, double *d_q,double*d_x, double *d_y)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    double r = d_q[icell];
    for (int iside = d_NbCell_ptr_c[icell]-onebase; iside < d_NbCell_ptr_c[icell+1]-onebase; iside++)
    {
        r -= d_a_l[iside] * d_x[d_NbCell_s[iside]-onebase];
    }
	
    d_y[icell] = r;
  }
}
__global__ void GetDiagCSRptr (const int d_nIntCells, const int onebase, const int* d_ptr,const int* d_idx, const double* val, int *d_ptr_new, int *d_idx_new, double *d_val_new)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    int count = 0;
    if (icell < d_nIntCells) {
        for (int iside = d_ptr[icell]-onebase; iside < d_ptr[icell+1]-onebase; iside++){
		int startidx=onebase;
		int endidx=d_nIntCells+startidx;
		if ((d_idx[iside]) >= startidx && (d_idx[iside]) < endidx)
		{
		    	count++;
		}
	}
	d_ptr_new[icell]=count;

    }
}//!!!!!!!!!!!!!!!!!!!!!!!!how to sum the array d_ptr_new????
__global__ void GetDiagCSR (const int d_nIntCells, const int onebase, const int* d_ptr,const int* d_idx, const double* d_val, int *d_ptr_new, int *d_idx_new, double *d_val_new)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    int count = d_ptr_new[icell];
    if (icell < d_nIntCells) {
        for (int iside = d_ptr[icell]-onebase; iside < d_ptr[icell+1]-onebase; iside++){
		int startidx=onebase;
		int endidx=d_nIntCells+startidx;
		if ((d_idx[iside]) >= startidx && (d_idx[iside]) < endidx)
		{
			d_val_new[count]=d_val[iside];
			d_idx_new[count]=d_idx[iside]-onebase;
		    	count++;
		}
	}

    }
}
__global__ void SetCSR0basedPtr (const int d_nIntCells, int* d_ptr)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
    	--d_ptr[icell];
    }
}
__global__ void SetCSR0based (const int d_nIntCells, int* d_ptr,int* d_idx)
{
    int icell = blockIdx.x*blockDim.x + threadIdx.x;
    if (icell < d_nIntCells) {
        for (int iside = d_ptr[icell]; iside < d_ptr[icell+1]; iside++){
	    --d_idx[iside];
	}
    }
}
HostMatrix* set_matrix_gpu_csr(){
    return new DeviceMatrixCSR();
}
void DeviceMatrixCSR::ToDiagMatrix(HostMatrix *hostmtxold){
	//printf("GPU get diag matrix is not finished!!!\n");
    int *hostptr=new int[m+1];
    int oldonebase=hostmtxold->getonebase();
    int d_nblock = (m+d_nthread-1)/d_nthread;
    hipLaunchKernelGGL(GetDiagCSRptr,dim3(d_nblock),dim3(d_nthread),0,0,m, oldonebase, hostmtxold->getptr(),hostmtxold->getidx(), hostmtxold->getval(), rowptr, colidx, val);
    hipMemcpy(hostptr+1, rowptr, sizeof(int)*(m), hipMemcpyDeviceToHost);
    hostptr[0]=0;
    for(int i=1;i<=m;i++){
	hostptr[i]+=hostptr[i-1];
    }
    nnz=hostptr[m];
    onebase=0;
    hipMemcpy(rowptr, hostptr, sizeof(int)*(m+1), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(GetDiagCSR,dim3(d_nblock),dim3(d_nthread),0,0,m, oldonebase, hostmtxold->getptr(),hostmtxold->getidx(), hostmtxold->getval(), rowptr, colidx, val);

    delete []hostptr;
}
void DeviceMatrixCSR::SpMV(HostVector *x,HostVector *y){
    double *d_x=x->val;
    double *d_y=y->val;
    int d_nblock = (m+d_nthread-1)/d_nthread;
#ifdef HAVE_MPI
    hipLaunchKernelGGL(getsendarrayHIP,dim3(d_nblock),dim3(d_nthread),0,0,nHalo, onebase, d_x, exchange_ptr, d_x+n);
    hipMemcpy(x_nHalo+n, d_x+n, sizeof(double)*(nHalo), hipMemcpyDeviceToHost);
    for(int i=0;i<nHalo;i++)
	x_nHalo[topo_c.exchange_ptr[i]]=x_nHalo[n+i];
    communicator_p2p(x_nHalo);
    communicator_p2p_waitall();
    hipMemcpy(d_x+n, x_nHalo+n, sizeof(double)*nHalo, hipMemcpyHostToDevice);
#endif
    hipLaunchKernelGGL(matMultCSR,dim3(d_nblock), dim3(d_nthread), 0, 0, m, onebase, rowptr, colidx, val, d_x, d_y);
}
//void DeviceMatrixCSR::bmAx(double *d_q, double *d_x,double *d_y)
void DeviceMatrixCSR::bmAx(HostVector *q, HostVector *x, HostVector *y){
    double *d_x=x->val;
    double *d_y=y->val;
    double *d_q=q->val;
    int d_nblock = (m+d_nthread-1)/d_nthread;
#ifdef HAVE_MPI
    hipLaunchKernelGGL(getsendarrayHIP, dim3(d_nblock),dim3(d_nthread), 0,0,nHalo, onebase, d_x, exchange_ptr, d_x+n);
    hipMemcpy(x_nHalo+n, d_x+n, sizeof(double)*(nHalo), hipMemcpyDeviceToHost);
    for(int i=0;i<nHalo;i++)
	x_nHalo[topo_c.exchange_ptr[i]]=x_nHalo[n+i];
    communicator_p2p(x_nHalo);
    communicator_p2p_waitall();
    hipMemcpy(d_x+n, x_nHalo+n, sizeof(double)*nHalo, hipMemcpyHostToDevice);
#endif
//    hipMemcpy(q_cpu, d_q, sizeof(double)*(m), hipMemcpyDeviceToHost);
    hipLaunchKernelGGL(bmAxCSR, dim3(d_nblock), dim3(d_nthread), 0, 0, m, onebase, rowptr, colidx,  val, d_q, d_x, d_y);
}
void DeviceMatrixCSR::parilu(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU,int **row_referenced, int sweep){
	int *lrows;
	int *lcols;
	double *lval;
	int *ucols;
	int *urows;
	double *uval;
	int nnzl,nnzu;
    	parilu_pre_set(val,rowptr,colidx,n,nnz,\
    	lval,lrows,lcols,nnzl,\
    	uval,ucols,urows,nnzu,\
    	row_referenced[0]);
	parilu_fact(val,rowptr,colidx,row_referenced[0],\
    			lval,lrows,lcols,\
                uval,ucols,urows,n,nnz,sweep);
	mtxL->SetMatrix(m,n,nnzl,lrows,lcols,lval);
	mtxU->SetMatrix(m,n,nnzu,ucols,urows,uval);
	hipFree(row_referenced[0]);
}
void DeviceMatrixCSR::parilut(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU, int sweep){
	int *lrows;
	int *lcols;
	double *lval;
	int *ucols;
	int *urows;
	double *uval;
	int nnzl,nnzu;
	int max_nnz_keep_rate=3;
	parilt::parilut_clean(val,colidx,rowptr,n,nnz,\
        lval,lcols,lrows,nnzl,\
        uval,ucols,urows,nnzu,\
        max_nnz_keep_rate,sweep);//nnzLU=nnz*max_nnz_keep_rate
	mtxL->SetMatrix(m,n,nnzl,lrows,lcols,lval);
	mtxU->MallocMatrix(m,nnzu);
    	hipsparseDcsr2csc(handle1,n,n,nnzu,uval,urows,ucols,mtxU->getval(),mtxU->getidx(),mtxU->getptr(),\
    	HIPSPARSE_ACTION_NUMERIC,HIPSPARSE_INDEX_BASE_ZERO);
	hipFree(uval);
	hipFree(ucols);
	hipFree(urows);
	//mtxU->SetMatrix(m,n,nnzu,ucols,urows,uval);
}
void DeviceMatrixCSR::parilu_csr(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU,int **row_referenced, HostVector *diag_U,int sweep){
	int *lrows;
	int *lcols;
	double *lval;
	int *ucols;
	int *urows;
	double *uval;
	int nnzl,nnzu;
    	parilu_pre_set(val,rowptr,colidx,n,nnz,\
    	lval,lrows,lcols,nnzl,\
    	uval,ucols,urows,nnzu,\
    	row_referenced[0]);
	parilu_fact(val,rowptr,colidx,row_referenced[0],\
    			lval,lrows,lcols,\
                uval,ucols,urows,n,nnz,sweep);
	mtxL->SetMatrix(m,n,nnzl,lrows,lcols,lval);
	mtxU->MallocMatrix(m,nnzu);
    	hipsparseDcsr2csc(handle1,n,n,nnzu,uval,ucols,urows,mtxU->getval(),mtxU->getidx(),mtxU->getptr(),\
    	HIPSPARSE_ACTION_NUMERIC,HIPSPARSE_INDEX_BASE_ZERO);
    	mtxU->getdiag(diag_U);
    	diag_U->jacobiInit(diag_U,0);
	hipFree(row_referenced[0]);
	
	hipFree(uval);
	hipFree(ucols);
	hipFree(urows);
}
void DeviceMatrixCSR::parilut_csr(DeviceMatrixCSR *mtxL,DeviceMatrixCSR *mtxU, HostVector *diag_U,int sweep){
	int *lrows;
	int *lcols;
	double *lval;
	int *ucols;
	int *urows;
	double *uval;
	int nnzl,nnzu;
	parilt::parilut_clean(val,colidx,rowptr,n,nnz,\
        lval,lcols,lrows,nnzl,\
        uval,ucols,urows,nnzu,\
        3,sweep);
	mtxL->SetMatrix(m,n,nnzl,lrows,lcols,lval);
	mtxU->SetMatrix(m,n,nnzu,urows,ucols,uval);
    	mtxU->getdiag(diag_U);
    	diag_U->jacobiInit(diag_U,0);
}
void DeviceMatrixCSR::Lsolve(HostVector *x,HostVector *y){
	lcsr_trsv(val,rowptr,colidx,y->val,x->val,n);
}
void DeviceMatrixCSR::Usolve(HostVector *x,HostVector *y){
	ucsc_trsv(val,rowptr,colidx,y->val,x->val,n,nnz);
}
void DeviceMatrixCSR::Lsolve_iter(HostVector *x,HostVector *y,HostVector *tmp,int maxiter){
	hipMemset(y->val,0,sizeof(double) * n);
	if(maxiter==0){
	  	hipMemcpy(y->val,x->val,sizeof(double)*n,hipMemcpyDeviceToDevice);
	}
	else if(maxiter>0){
	    for(int i=0;i<maxiter;i++){
	        hipLaunchKernelGGL(LMultCSR,dim3(d_nblock), dim3(d_nthread), 0, 0, m, onebase, rowptr, colidx, val, y->val,tmp->val);
	        hipLaunchKernelGGL(lsolveiterCSR,dim3(d_nblock), dim3(d_nthread), 0, 0, m, onebase, rowptr, colidx, val, x->val, tmp->val, y->val);
	    }
	}
}
void DeviceMatrixCSR::Usolve_iter(HostVector *x,HostVector *y,HostVector *tmp,HostVector *diag_U,int maxiter){
	hipMemset(y->val,0,sizeof(double) * n);
	if(maxiter==0){
    	    diag_U->jacobiSolve(x,y);
	}
	else if(maxiter>0){
	    for(int i=0;i<maxiter;i++){
	    	hipLaunchKernelGGL(UMultCSR,dim3(d_nblock), dim3(d_nthread), 0, 0, m, onebase, rowptr, colidx, val, y->val, tmp->val);
	    	hipLaunchKernelGGL(usolveiterCSR,dim3(d_nblock), dim3(d_nthread), 0, 0, m, onebase, rowptr, colidx, val, x->val, tmp->val, y->val, diag_U->val);
	    }
	}
}

