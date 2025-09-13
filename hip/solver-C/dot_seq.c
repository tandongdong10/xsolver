#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
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
	
        for(i = 0;i<N;i++){   
	    res +=X[i]*Y[i];
        }
	return res;

}
