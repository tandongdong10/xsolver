#include "HostVector.h"
void HostVector::MallocVector(int n_in){
    n=n_in;
    val=new double[n];
    memset(val,0,n*sizeof(double));
}
void HostVector::MallocVector(int n_in,double *v){
    n=n_in;
    val=new double[n];
    memcpy(val,v,n*sizeof(double));
}
void HostVector::CopyVector(int n_in,double *v){
    memcpy(val,v,n*sizeof(double));
}
void HostVector::SetVector(int n_in,double *tmp){
    n=n_in;
    val=tmp;
}
void HostVector::GetVector(double *tmp){
    memcpy(tmp,val,n*sizeof(double));
}
#ifdef HAVE_MPI
void HostVector::MallocVector(int n_in,int nHalo_in){
    n=n_in;
    nHalo=nHalo_in;
    val=new double[n+nHalo];
    memset(val,0,(n+nHalo)*sizeof(double));
}
void HostVector::MallocVector(int n_in,int nHalo_in,double *v){
    n=n_in;
    nHalo=nHalo_in;
    val=new double[n+nHalo];
    memcpy(val,v,n*sizeof(double));
}
#endif
void HostVector::UpdateVector(int n_in,double *tmp){	
    n=n_in;
    val=tmp;
}
double HostVector::vec_dot(HostVector *y){
    double res=0;
    res += cblas_ddot(n,val,1,y->val,1);
#ifdef HAVE_MPI
    communicator_sum(res);
#endif
    return res;	
}
void HostVector::vec_dot2(HostVector *y,HostVector *q,HostVector *z, double *res){
    res[0]=0;
    res[1]=0;
    res[0] += cblas_ddot(n,val,1,y->val,1);
    res[1] += cblas_ddot(n,q->val,1,z->val,1);
#ifdef HAVE_MPI
    communicator_sum(res,2);
#endif
}
void HostVector::vec_copy(HostVector *x){
    cblas_dcopy(n,x->val,1,val,1);
}
double HostVector::vec_norm1(){
    double res=0;
    for(int i=0;i<n;i++)
	res+=fabs(val[i]);
#ifdef HAVE_MPI
    communicator_sum(res);
#endif
    return res;	
}
void HostVector::vec_axpy(double alpha, HostVector *x){
    cblas_daxpy(n,alpha,x->val,1,val,1);
}
void HostVector::vec_scal(double alpha){
    cblas_dscal(n,alpha,val,1);
}
void HostVector::vec_bicg_kernel1(double omega, double alpha, HostVector *res,HostVector *uk){
    kernel_1(n,omega,alpha,res->val,uk->val,val);
}
void HostVector::vec_bicg_kernel2(double gama, HostVector *res,HostVector *uk){
    kernel_2(n,res->val,gama,uk->val,val);
}
void HostVector::vec_bicg_kernel3(double gama, double alpha, HostVector *pk,HostVector *sk){
    kernel_3(n,gama,pk->val,alpha,sk->val,val);
}
void HostVector::jacobiInit(HostVector *diag_val, double small){
	double tmp=0;
	for(int i=0;i<n;i++){
	    tmp=diag_val->val[i]+small;
	    if(tmp==0){
		printf("The Matrix val[%d][%d]==0, Jacobi can not work!!!\n",i,i);
		printf("Please change preconditioner\n");
		exit(0);
	    }
	    val[i]=1.0/tmp;
	}
}
void HostVector::jacobiSolve(HostVector *x,HostVector *y){
        for(int i=0;i<n;i++){
       	    y->val[i] = val[i]*x->val[i];
        }
}
void HostVector::vec_print(){
    for(int i=0;i<n;i++)
	printf("val[%d] = %lg  \n",i,val[i]);
}
void HostVector::FreeVector(){
    if(val!=NULL)
	delete []val;
}
HostVector* set_vector_cpu(){
    return new HostVector();
}
HostVector* set_vector_cpu(int n){
    return new HostVector(n);
}
#ifdef HAVE_MPI
HostVector* set_vector_cpu(int n,int nHalo){
    return new HostVector(n,nHalo);
}
#endif
extern int pam[3];
extern HostVector* set_vector_gpu();
extern HostVector* set_vector_gpu(int n);
class DeviceVector;
#ifdef HAVE_MPI
extern HostVector* set_vector_gpu(int n,int nHalo);
#endif
HostVector* NewVector(){
    if(pam[2]!=1)
	return set_vector_cpu();
    else
	return set_vector_gpu();
}
HostVector* NewVector(int n){
    if(pam[2]!=1)
	return set_vector_cpu(n);
    else
	return set_vector_gpu(n);
}
#ifdef HAVE_MPI
HostVector* NewVector(int n, int nHalo){
    if(pam[2]!=1)
	return set_vector_cpu(n,nHalo);
    else
	return set_vector_gpu(n,nHalo);
}
#endif

