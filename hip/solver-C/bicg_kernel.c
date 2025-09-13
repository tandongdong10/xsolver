

//pk = res + omega*(pk-alpha*uk)
template<typename TYPE>
void kernel_1(const int len,const TYPE omega,const TYPE alpha,const TYPE *res,TYPE *uk,TYPE *pk){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 0;i < len;i++){
        pk[i] = res[i] + omega*(pk[i] - alpha*uk[i]);
    }
}
//sk = res - gama * uk
template<typename TYPE>
void kernel_2(const int len,const TYPE *res,const TYPE gama,const TYPE *uk,TYPE *sk){
#ifdef _OPENMP
#pragma omp parallel for
#endif 
   for(int i = 0;i < len;i++) {
       sk[i] = res[i] - gama*uk[i];
   } 
}
//xk=xk+gama*pk+alpha*sk
template<typename TYPE>
void kernel_3(const int len,const TYPE gama,const TYPE *pk,const TYPE alpha,const TYPE *sk, TYPE *xk){
#ifdef _OPENMP
#pragma omp parallel for
#endif 
    for(int i = 0;i < len;i++){
        xk[i] = xk[i] + gama * pk[i] + alpha * sk[i];
    }
}
//res = (vk.sk)/(vk.vk)
template<typename TYPE>
TYPE kernel_4(const int len,const TYPE *vk,const TYPE *sk,const TYPE small){
    TYPE tmp1,tmp2,res;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:tmp1) reduction(+:tmp2)
#endif     
    for(int i = 0;i < len;i++){
        tmp1 += vk[i] * sk[i];
        tmp2 += vk[i] * vk[i];
    }
    res = tmp1/(tmp2 + small);
    return res;
}

