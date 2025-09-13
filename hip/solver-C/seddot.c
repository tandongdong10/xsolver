#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#ifndef TYPE
#define TYPE double
#endif

TYPE dot(const int N,   
     const TYPE* X,
     const TYPE* Y)
{
	TYPE res = 0.0;
        int i;
    	int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {     
        num_threads = omp_get_num_threads();
    }
	TYPE* tmp= (TYPE* )malloc(num_threads*sizeof(TYPE));
        memset(tmp,0,num_threads*sizeof(TYPE));
#ifdef _OPENMP
#pragma omp parallel for 
//#pragma omp parallel for reduction(+:res)
#endif	
        for(i = 0;i<N;i++){   
	    int id=omp_get_thread_num();
	    tmp[id] += X[i]*Y[i];
        }
	for(i=0;i<num_threads;i++)
	    res+=tmp[i];
	return res;

}
