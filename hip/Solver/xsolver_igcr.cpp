#include "xsolver_igcr.h"
template <typename VectorType>
void Xsolver_igcr<VectorType>::xsolver_init(){
    //uk = NewVector2(restart);
    uk = new VectorType*[restart];//NewVector2(restart);
    for (int i=0; i<restart; i++)
    {
#ifdef HAVE_MPI
       uk[i] = new VectorType(n,nHalo);
#else
       uk[i] = new VectorType(n);
#endif
    } 
    ck = new VectorType*[restart];//NewVector2(restart);
    for (int i=0; i<restart; i++)
    {
       ck[i] = new VectorType(n);
    } 
    res = new VectorType(n);
#ifdef HAVE_MPI
    tmp_vec = new VectorType(n,nHalo);
#else
    tmp_vec = new VectorType(n);
#endif
    tmp_v  = new double[2];
    precond->preconditioner_init();
}
template <typename VectorType>
void Xsolver_igcr<VectorType>::xsolver(){
    double error = res0;
    int iiter = 0, k=0;
    double one = 1.0, minone = -1.0, zero = 0.0;
    double beta  = 0.0;
    double alpha  = 0.0, minalpha, tmp, mintmp, tmp1, rnorm;
    hostmtx->bmAx(q, phi, res);
    //hostmtx->SpMV(phi, res);
    // cblas_daxpy(n,one,q,1,res,1);
    res0 = res->vec_dot(res);
    res0 = sqrt(res0);
    //res0 = res->vec_norm1();
    if(resvec!=NULL)
	resvec[0]=res0;
    //printf("!!!!!!!!!!!!!!res0=%lg\n",res0);
    if(res0 == 0|| res0<=absolute_tol){return;}
    while(iiter<maxiter)
    {
    	k = 0;
    	while( k<restart && iiter<maxiter )
    	{
      	    precond->preconditioner(res,uk[k]);
    	    hostmtx->SpMV(uk[k], ck[k]);
       	    for(int i=0;i<k;i++)
       	    {
          	alpha = ck[i]->vec_dot(ck[k]); 
          	minalpha = minone*alpha;
          	ck[k]->vec_axpy(minalpha,ck[i]);
          	uk[k]->vec_axpy(minalpha,uk[i]);
       	    } 
       	    tmp_v[0] = ck[k]->vec_dot(ck[k]);
       	    tmp_v[1] = ck[k]->vec_dot(res);
       	    tmp_v[0] = sqrt(tmp_v[0]);
       	    tmp1 = one/(tmp_v[0]+small);
      	    uk[k]->vec_scal(tmp1);
      	    ck[k]->vec_scal(tmp1);
	    //if(k==0)
	    //	uk[k]->vec_print();
       	    
       	    tmp = tmp_v[1]/(tmp_v[0]+small);
       	    mintmp = minone*tmp;
	    //if(k==0)
	    //	uk[k]->vec_print();
  	    //hipDeviceSynchronize();//????????
	    //DeviceVector *cast_x=dynamic_cast<DeviceVector*>(phi);
	    //DeviceVector *cast_uk=dynamic_cast<DeviceVector*>(uk[k]);
       	    //cast_x->vec_axpy(tmp,cast_uk);
       	    phi->vec_axpy(tmp,uk[k]);
	    //if(k==0)
	    //	phi->vec_print();
       	    res->vec_axpy(mintmp,ck[k]);
       	    //check convergence 
       	    tmp = res->vec_dot(res);    
       	    rnorm = sqrt(tmp);
     	    //rnorm = res->vec_norm1();    
       	    error = rnorm/(res0+small);
	    if(resvec!=NULL)
		resvec[iiter+1]=rnorm;
       	    //printf("iter = %d , error = %f , rnorm = %f\n",iiter+1,error,rnorm);
     	    if(error<tol||rnorm<=absolute_tol)
       	    {
         	usediter = iiter+1;
		if(usediter_pointer!=NULL)
    		    usediter_pointer[0]=usediter;
		if(phi_old!=NULL)
		    phi->GetVector(phi_old);
         	return;
       	    }
       	    k = k + 1;
       	    iiter = iiter + 1;
    	}
    }
    usediter = iiter;
    if(usediter_pointer!=NULL)
    	usediter_pointer[0]=usediter;
    //res0=rnorm; 
    ///precond->preconditioner(phi,phi);
    if(phi_old!=NULL)
    	phi->GetVector(phi_old);
}
template <typename VectorType>
void Xsolver_igcr<VectorType>::xsolver_free(){
    res->FreeVector();
    tmp_vec->FreeVector();
    for (int i=0; i<restart; i++)
    {
    	uk[i]->FreeVector();
    	ck[i]->FreeVector();
    }
    delete []tmp_v;
    delete []uk;
    delete []ck;
    precond->preconditioner_free();
}
template class Xsolver_igcr<HostVector>;
template class Xsolver_igcr<DeviceVector>;
template <typename VectorType>
Xsolver* solver_set_igcr(){
    return new Xsolver_igcr<VectorType>();
}
