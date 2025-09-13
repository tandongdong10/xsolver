#include "HostMatrix.h"
extern HostMatrix* set_matrix_csc();
extern HostMatrix* set_matrix_csr();
extern HostMatrix* set_matrix_mcsr();
extern HostMatrix* set_matrix_ell();
extern HostMatrix* set_matrix_gpu_csr();
extern HostMatrix* set_matrix_gpu_ell();
HostMatrix* matrixform_set(const char* fmt){
    if(strcmp(fmt,"CSC")==0)
	return set_matrix_csc();
    else if(strcmp(fmt,"CSR")==0)
	return set_matrix_csr();
    else if(strcmp(fmt,"MCSR")==0)
	return set_matrix_mcsr();
    else if(strcmp(fmt,"GPUCSR")==0)
	return set_matrix_gpu_csr();
    else if(strcmp(fmt,"ELL")==0)
	return set_matrix_ell();
    else if(strcmp(fmt,"GPUELL")==0)
	return set_matrix_gpu_ell();
    else{
	printf("Matrix Format Set Wrong !!!\n");
	return NULL;
    }
}
