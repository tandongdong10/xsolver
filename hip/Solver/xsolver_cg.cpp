#include "xsolver_cg.h"
template <typename VectorType>
void Xsolver_cg<VectorType>::xsolver_init(){
     res  = new VectorType(n);
#ifdef HAVE_MPI
     pk  = new VectorType(n,nHalo);   
     zk  = new VectorType(n,nHalo);
#else
     pk  = new VectorType(n);   
     zk  = new VectorType(n);
#endif
     Apk  = new VectorType(n);
     qk  = new VectorType(n);
     precond->preconditioner_init();
}
template <typename VectorType>
void Xsolver_cg<VectorType>::xsolver(){
    double error = res0;
    int iiter = 0;
    double one = 1.0, minone = -1.0;
    double beta  = 0.0;
    double alpha, minalpha;
    double sigma,tao,taoo;
    double resl;
    hostmtx->bmAx(q, phi, res);
    precond->preconditioner(res,zk);
    tao = res->vec_dot(zk);
    res0 = res->vec_dot(res);
    res0 = sqrt(res0);
    //res0 = res->vec_norm1();
    pk->vec_copy(zk); 
    if(resvec!=NULL)
	resvec[0]=res0;
    //printf("!!!!!!!!!!!!!!res0=%lg\n",res0);

    if(res0 == 0||res0<=absolute_tol){return;}

    for(iiter=0;iiter<maxiter;iiter++)
    {
        usediter = iiter;   
	taoo=tao; 
    	hostmtx->SpMV(pk, Apk);
     	sigma = Apk->vec_dot(pk);
     	alpha = tao/(sigma+small);
     	minalpha = minone*alpha;
     	phi->vec_axpy(alpha,pk);
     	res->vec_axpy(minalpha,Apk);
      	precond->preconditioner(res,zk);
     	tao = zk->vec_dot(res);
     	beta = tao/(taoo+small);
     	pk->vec_scal(beta);
	pk->vec_axpy(one,zk);
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
	    if(phi_old!=NULL)
	    	phi->GetVector(phi_old);
     	    return;
     	}
    }
    usediter = iiter;
    usediter_pointer[0]=usediter;
    if(phi_old!=NULL)
    	phi->GetVector(phi_old);
}
template <typename VectorType>
void Xsolver_cg<VectorType>::xsolver_free(){
    res->FreeVector();
    pk->FreeVector();
    zk->FreeVector();
    Apk->FreeVector();
    qk->FreeVector();
    phi->FreeVector();
    delete res;
    delete pk;
    delete zk;
    delete Apk;
    delete qk;
    //delete q;
    //delete phi;
    precond->preconditioner_free();
    //delete precond;
}
template class Xsolver_cg<HostVector>;
template class Xsolver_cg<DeviceVector>;
template <typename VectorType>
Xsolver* solver_set_cg(){
    return new Xsolver_cg<VectorType>();
}
