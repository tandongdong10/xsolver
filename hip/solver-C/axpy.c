#ifdef _OPENMP
#include "/usr/lib/gcc/x86_64-redhat-linux/4.8.2/include/omp.h" 
//#include <omp.h>
#endif

//#ifndef TYPE
//#define TYPE double
//#endif

template<typename TYPE>
void axpy(const int nz,
	   const TYPE a,
	   const TYPE* x,
	   TYPE* y)

{
       int i = 0;
#ifdef _OPENMP
	#pragma omp parallel for
#endif
     for(i = 0;i < nz; i++){
       	y[i] += a*x[i];
     }
}
