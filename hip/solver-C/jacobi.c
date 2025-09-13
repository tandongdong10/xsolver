#ifdef _OPENMP
#include <omp.h>
#endif

#define TYPE double
void jacobi(const int nz,
           const TYPE* X, 
           const TYPE* Y,
           TYPE* Z)
{

#ifdef _OPENMP
    #pragma omp parallel for
#endif
      for (int icell = 0; icell < nz; icell++)
      {
         Z[icell] = X[icell] * Y[icell];

      }
}
