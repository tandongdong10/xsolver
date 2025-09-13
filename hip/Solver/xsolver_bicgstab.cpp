#include "xsolver_bicgstab.h"
template <typename VectorType>
void Xsolver_bicgstab<VectorType>::xsolver_init(){
#ifdef HAVE_MPI
     zk  = new VectorType(n,nHalo);
     zk1  = new VectorType(n,nHalo);
#else
     zk  = new VectorType(n);
     zk1  = new VectorType(n);
#endif
     pk  = new VectorType(n);   
     uk  = new VectorType(n);
     vk  = new VectorType(n);
     sk  = new VectorType(n);
     reso  = new VectorType(n);
     res  = new VectorType(n);
     tmp_v  = new double[2];
     precond->preconditioner_init();
}
template <typename VectorType>
void Xsolver_bicgstab<VectorType>::xsolver(){
    double error = res0;
    int iiter = 0;
    double one = 1.0, minone = -1.0, zero = 0.0;
    double beta  = 0.0;
    double alpha  = 1.0, minalpha, tmp;
    double beto  = 1.0;
    double gama  = 1.0;
    double omega, resl, mingama;
    
    struct timeval time3, time4, time5;
    double elapsed_time1 = 0;
    double elapsed_time2 = 0;

    hostmtx->bmAx(q, phi, res);
    //hostmtx->SpMV(phi, res);
    // cblas_daxpy(n,one,q,1,res,1);
    res0 = res->vec_dot(res);
    res0 = sqrt(res0);
    //res0 = res->vec_norm1();
    //printf("!!!!!!!!!!!!!!res0=%lg\n",res0);
    reso->vec_copy(res); 
    if(resvec!=NULL)
	resvec[0]=res0;

    if(res0 == 0|| res0<=absolute_tol){return;}

    for(iiter=0;iiter<maxiter;iiter++)
    {
        usediter = iiter;    
     	beta = reso->vec_dot(res);
     	omega = (beta*gama)/(alpha*beto+1e-40);
     	beto = beta;
     	//minalpha = minone*alpha;
     	//pk->vec_axpy(minalpha,uk);
     	//pk->vec_scal(omega);
     	//pk->vec_axpy(one,res);
	pk->vec_bicg_kernel1(omega,alpha,res,uk);
    hipDeviceSynchronize();
	gettimeofday(&time3,NULL);
      	precond->preconditioner(pk,zk);
    hipDeviceSynchronize();
	gettimeofday(&time4,NULL);
	elapsed_time1 += (time4.tv_sec - time3.tv_sec) * 1000. +(time4.tv_usec - time3.tv_usec) / 1000.;
    	hostmtx->SpMV(zk, uk);
    hipDeviceSynchronize();
	gettimeofday(&time5,NULL);
	elapsed_time2 += (time5.tv_sec - time4.tv_sec) * 1000. +(time5.tv_usec - time4.tv_usec) / 1000.;
     	tmp = reso->vec_dot(uk);
     	gama = beta/(tmp+small);
     	//mingama = minone*gama;
     	//sk->vec_copy(res); 
     	//sk->vec_axpy(mingama,uk);
	sk->vec_bicg_kernel2(gama,res,uk);
    hipDeviceSynchronize();
	gettimeofday(&time3,NULL);
      	precond->preconditioner(sk,zk1);
    hipDeviceSynchronize();
	gettimeofday(&time4,NULL);
	elapsed_time1 += (time4.tv_sec - time3.tv_sec) * 1000. +(time4.tv_usec - time3.tv_usec) / 1000.;
    	hostmtx->SpMV(zk1, vk);
    hipDeviceSynchronize();
	gettimeofday(&time5,NULL);
	elapsed_time2 += (time5.tv_sec - time4.tv_sec) * 1000. +(time5.tv_usec - time4.tv_usec) / 1000.;
     	//tmp_v[0] = sk->vec_dot(vk);
     	//tmp_v[1] = vk->vec_dot(vk);
	sk->vec_dot2(vk,vk,vk,tmp_v);
     	alpha = tmp_v[0]/(tmp_v[1]+small);
     	//phi->vec_axpy(gama,zk);
     	//phi->vec_axpy(alpha,zk1);
	phi->vec_bicg_kernel3(gama,alpha,zk,zk1);
     	minalpha = minone*alpha;
     	res->vec_copy(sk); 
     	res->vec_axpy(minalpha,vk);
     	//check convergence 
     	resl = res->vec_dot(res);    
     	resl = sqrt(resl);
     	//resl = res->vec_norm1();    
     	error = resl/(res0+small);
	if(resvec!=NULL)
	    resvec[iiter+1]=resl;
       	//printf("iter = %d , error = %f , rnorm = %f\n",iiter,error,resl);
     	if(error<tol||resl<=absolute_tol)
     	{
     	    usediter = iiter+1;
    	    usediter_pointer[0]=usediter;
      	    //precond->preconditioner(phi,phi);

	    printf("spmv time: %lf(ms)\n", elapsed_time2); 
            printf("precond time: %lf(ms)\n", elapsed_time1);

	    if(phi_old!=NULL)
	    	phi->GetVector(phi_old);
     	    return;
     	}
    }
    usediter = iiter;
    usediter_pointer[0]=usediter;
    printf("spmv time: %lf(ms)\n", elapsed_time2); 
    printf("precond time: %lf(ms)\n", elapsed_time1);
    //precond->preconditioner(phi,phi);
    if(phi_old!=NULL)
    	phi->GetVector(phi_old);
}
template <typename VectorType>
void Xsolver_bicgstab<VectorType>::xsolver_free(){
    res->FreeVector();
    reso->FreeVector();
    zk->FreeVector();
    zk1->FreeVector();
    pk->FreeVector();
    sk->FreeVector();
    uk->FreeVector();
    vk->FreeVector();
    delete res;
    delete reso;
    delete zk;
    delete zk1;
    delete pk;
    delete sk;
    delete uk;
    delete vk;
    delete []tmp_v;
    precond->preconditioner_free();
	phi->FreeVector();
	q->FreeVector();
	//delete q;
	//delete phi;
	//delete precond;
}
template class Xsolver_bicgstab<HostVector>;
template class Xsolver_bicgstab<DeviceVector>;
template <typename VectorType>
Xsolver* solver_set_bicgstab(){
    return new Xsolver_bicgstab<VectorType>();
}
