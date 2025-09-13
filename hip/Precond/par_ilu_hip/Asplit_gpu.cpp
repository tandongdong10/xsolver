#include "Asplit_gpu.h"
extern hipsparseHandle_t handle1;
#define T double
__global__ void split_csr2LU_L1d_Ud_gpu_pre_count(T*val,int*rows,int*cols,int*lrows,int*ucols,int n,int*uptr){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int i=gid;
    if(i>=n)return;
    lrows+=1;
    ucols+=1;
    int flag=0;
    for(int j=rows[i];j<rows[i+1];j++){
        if(cols[j]<i){
            lrows[i]+=1;
        }
        else if(cols[j]==i){
            uptr[i]=j;
            //lrows[i]+=1;
            flag=1;
        }
        else if(cols[j]>i){
            ucols[i]+=1;
        }
    }
    lrows[i]+=1;
    ucols[i]+=flag;
}

__global__ void split_csr2LU_L1d_Ud_gpu_split_lu(T*val,int*rows,int*cols,int n,\
T*lval,int*lrows,int*lcols,T*uval,int*ucols,int*urows,int*uptr){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.x;
    if(i>=n)return;
    int loffset=-rows[i]+lrows[i];
    int uoffset=-uptr[i]+ucols[i];
    int j;
    for(j=rows[i]+threadIdx.x;j<rows[i+1];j+=blockDim.x){
        int k=cols[j];
        if(k<i){
            lval[j+loffset]=val[j];
            lcols[j+loffset]=cols[j];
        }
        else{
            uval[j+uoffset]=val[j];
            urows[j+uoffset]=cols[j];
        }
    }
    lval[lrows[i+1]-1]=1;
    lcols[lrows[i+1]-1]=i;
}
__global__ void upre_count(T*uval,int*ucols,int*urows,int*ucols2,int nnz){
    ucols2+=1;
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid>=nnz)return;
    int col=urows[gid];//ucols
    atomicAdd(ucols2+urows[gid],1);
}

__global__ void Ucsr2Ucsc(T*uval,int*ucols,int*urows,T*uval2,int*ucols2,int*urows2,int n){
}


void parilu_pre(T*d_val,int*d_rows,int*d_cols,int n,int nnz,
T*&d_lval,int*&d_lrows,int*&d_lcols,int&gnnzl,
T*&d_uval,int*&d_ucols,int*&d_urows,int&gnnzu)
{
    hipMalloc(&d_lrows,sizeof(int)*(n+1));
    hipMalloc(&d_ucols,sizeof(int)*(n+1));

    hipMemset(d_lrows,0,sizeof(int)*(n+1));
    hipMemset(d_ucols,0,sizeof(int)*(n+1));

    int*uptr;
    hipMalloc(&uptr,sizeof(int)*(n));
    hipMemset(uptr,-1,sizeof(int)*n);
    int blocksize=64;
    int gridsize=n/blocksize+1;
    //split_csr2LU_L1d_Ud_gpu_pre_count(T*val,int*rows,int*cols,int*lrows,int*ucols,int n)
    hipLaunchKernelGGL(split_csr2LU_L1d_Ud_gpu_pre_count,dim3(gridsize),dim3(blocksize),0,0,\
    d_val,d_rows,d_cols,d_lrows,d_ucols,n,uptr);
    
    int*hlrows=new int[n+1];
    int*hucols=new int[n+1];
    hipMemcpyDtoH(hlrows,d_lrows,sizeof(int)*(n+1));

    hipMemcpyDtoH(hucols,d_ucols,sizeof(int)*(n+1));

    hlrows[0]=0;
    hucols[0]=0;
    for(int i=0;i<n;i++){
        hlrows[i+1]+=hlrows[i];
        hucols[i+1]+=hucols[i];
    }

    int nnzl=hlrows[n];
    int nnzu=hucols[n];
	
    gnnzl=nnzl;
    gnnzu=nnzu;

    hipMalloc(&d_lval,sizeof(T)*nnzl);
    hipMalloc(&d_lcols,sizeof(int)*nnzl);

    hipMalloc(&d_uval,sizeof(T)*nnzu);
    hipMalloc(&d_urows,sizeof(int)*nnzu);

    hipMemcpyHtoD(d_lrows,hlrows,sizeof(int)*(n+1));
    hipMemcpyHtoD(d_ucols,hucols,sizeof(int)*(n+1));

    //split_csr2LU_L1d_Ud_gpu_split_lu(T*val,int*rows,int*cols,int n,\
T*lval,int*lrows,int*lcols,T*uval,int*ucols,int*urows,int*uptr)
    blocksize=64;
    gridsize=n;
    hipLaunchKernelGGL(split_csr2LU_L1d_Ud_gpu_split_lu,dim3(gridsize),dim3(blocksize),0,0,\
    d_val,d_rows,d_cols,n,d_lval,d_lrows,d_lcols,\
    d_uval,d_ucols,d_urows,uptr);


    //transpose U
    T*d_uval2;
    int*d_ucols2;
    int*d_urows2;

    hipMalloc(&d_uval2,sizeof(T)*nnzu);
    hipMalloc(&d_ucols2,sizeof(int)*(n+1));
    hipMalloc(&d_urows2,sizeof(int)*nnzu);
    
    /**
    Use hipsparse
    blocksize=64;
    gridsize=nnzu/blocksize+1;


    hipMemset(ucols2,0,sizeof(int)*(n+1));
    //__global__ void upre_count(T*uval,int*ucols,int*urows,int*ucols2,int nnz)
    hipLaunchKernelGGL(upre_count,dim3(gridsize),dim3(blocksize),0,0,\
    d_uval,d_ucols,d_urows,d_ucols2,nnzu);

    hipMemcpyDtoH(hucols,ucols2,sizeof(int)*(n+1));
    hucols[0]=0;
    for(int i=1;i<=n;i++){
        hucols[i]+=hucols[i-1];
    }
    hipMemcpyHtoD(ucols2,hucols,sizeof(int)*(n+1));
    **/

    hipDeviceSynchronize();
    hipsparseDcsr2csc(handle1,n,n,nnzu,d_uval,d_ucols,d_urows,d_uval2,d_urows2,d_ucols2,\
    HIPSPARSE_ACTION_NUMERIC,HIPSPARSE_INDEX_BASE_ZERO);

    T*temp_uval=d_uval;
    int*temp_ucols=d_ucols;
    int*temp_urows=d_urows;
    d_uval=d_uval2;
    d_urows=d_urows2;
    d_ucols=d_ucols2;
    hipFree(temp_uval);
    hipFree(temp_ucols);
    hipFree(temp_urows);
    delete[]hlrows;
    delete[]hucols;
}
#undef T
