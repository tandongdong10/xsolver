#include <stdio.h>
#ifdef _OPENMP
//#include <omp.h>
#include "/usr/lib/gcc/x86_64-redhat-linux/4.8.2/include/omp.h"
#endif


//#ifndef TYPE
//#define TYPE double
//#endif

template<typename TYPE>
TYPE dot(const int N,   
     const TYPE* X,
     const TYPE* Y)
{
	TYPE res = 0.0;
	TYPE tmp = 0.0;
        int i;
	int num_threads=1;
	int idx=0;
	
#ifdef _OPENMP
#pragma omp parallel 
	num_threads=omp_get_num_threads();
	TYPE *res_array=new TYPE[num_threads]();
#endif	
#ifdef _OPENMP
#pragma omp parallel for
#endif	
        for(i = 0;i<N;i++){   
#ifdef _OPENMP
	    idx=omp_get_thread_num();
	    res_array[idx] +=X[i]*Y[i];
#else	
	    res +=X[i]*Y[i];
#endif	
        }
#ifdef _OPENMP
	for(i=0;i<num_threads;i++)
	    res+=res_array[i];
	delete []res_array;
#endif	
	return res;

}
