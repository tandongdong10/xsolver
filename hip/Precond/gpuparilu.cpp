#ifndef _GPUPARILU_H_
#define _GPUPARILU_H_
#include "par_ilu_hip/Asplit_gpu.h"
#define T double
__global__ void empty_launch(int n)
{
int i=blockIdx.x*blockDim.x+threadIdx.x;
}

void empty_launch_gpu(int n)
{
    hipLaunchKernelGGL(empty_launch, dim3(1),dim3(1),0,0,1);

};

__global__ void set_reference(T*d_val,int*d_rows,int*d_cols,int n,int*d_row_reference){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.x;
    if(i<n){
    for(int j=d_rows[i]+threadIdx.x;j<d_rows[i+1];j+=blockDim.x){
        d_row_reference[j]=i;
    }
    }
}

void parilu_pre_set(T*d_val,int*d_rows,int*d_cols,int n,int nnz,
T*&d_lval,int*&d_lrows,int*&d_lcols,int&gnnzl,
T*&d_uval,int*&d_ucols,int*&d_urows,int&gnnzu,
int*&d_row_reference){

parilu_pre(d_val,d_rows,d_cols,n,nnz,
d_lval,d_lrows,d_lcols,gnnzl,
d_uval,d_ucols,d_urows,gnnzu);

int blocksize=64;
int gridsize=n;

hipMalloc(&d_row_reference,sizeof(int)*nnz);

hipLaunchKernelGGL(set_reference,dim3(gridsize),dim3(blocksize),0,0,\
    d_val,d_rows,d_cols,n,d_row_reference);

hipDeviceSynchronize();
}

__global__ void ilu0_ij_aw(T*lval,int*lrows,int*lcols,T*uval,int*urows,int*ucols,int n,\
    int*rows,int*cols,int*row_reference,T*val,int nnz)
{
    
        int i, j;
        int k = blockDim.x * blockIdx.x + threadIdx.x;
    
        int il, iu, jl, ju;
        int*rowidxA=row_reference;
        int*colidxA=cols;
        //int nnz=rows[n];
        T*A=val;
        int*rowptrL=lrows;
        int*colidxL=lcols;
        T*valL=lval;
        
        int*rowptrU=ucols;
        int*colidxU=urows;
        T*valU=uval;

        T zero = 0.0;
        T s, sp;
        if (k < nnz) {
           i = rowidxA[k];
            j = colidxA[k];
            
            s =  A[k];
            
            il = rowptrL[i];
            iu = rowptrU[j];
    
            while (il < rowptrL[i+1] && iu < rowptrU[j+1]) {
                sp = zero;
                jl = colidxL[il];
                ju = colidxU[iu];
                sp = (jl == ju) ? valL[il] * valU[iu] : sp;
                s = (jl == ju) ? s-sp : s;
                il = (jl <= ju) ? il+1 : il;
                iu = (jl >= ju) ? iu+1 : iu;
            }
            s += sp;  // undo the last operation (it must be the last)        
            if (i > j){      // modify l entry
		T tmp=valU[rowptrU[j+1]-1];
		if(fabs(tmp)<1e-10)
		    tmp=(tmp>=0?1e-10:-1e-10);
		tmp=s/tmp;
		if (isfinite(tmp)) 
                    valL[il-1] =  tmp;
	    }
            else{            // modify u entry
		if(i==j&&fabs(s)<1e-10)
		    s=(s>=0?1e-10:-1e-10);
		if (isfinite(s)) 
                    valU[iu-1] = s;
	    }
        }
}
__global__ void ilu0_aij_Ldiag1_Udiag_try_faster2(T*lval,int*lrows,int*lcols,\
    T*uval,int*urows,int*ucols,int n,\
    int*rows,int*cols,int*row_reference,T*val,int nnz)
{
    //claim 8
    //input 12
    int k=blockDim.x*blockIdx.x+threadIdx.x;
    if(k<nnz){
        int i=row_reference[k];
        int j=cols[k];
        T tempv=val[k];
    
        int index_u=ucols[j];
        int index_l=lrows[i];
    
        int lj,ui;
        ui=urows[index_u];
        lj=lcols[index_l];

        int check_last=(i<=j)?i:j;
        int&check_id=(i<=j)?ui:lj;

        while(check_id<check_last){

            tempv=(lj==ui)?(tempv-lval[index_l]*uval[index_u]):tempv;
    
            index_l=(lj<=ui)?index_l+1:index_l;
            index_u=(lj>=ui)?index_u+1:index_u;
            
            lj=lcols[index_l];
            ui=urows[index_u];
        }
    
    
        if(j<i){
            lval[index_l]=tempv/uval[ucols[j+1]-1];
        }
        else{
	    if(i==j&&fabs(tempv)<1e-10)
		tempv=1e-10;
            uval[index_u]=tempv;
        }
    
    
    }
}

void parilu_fact(T*&vald,int*&rowsd,int*&colsd,int*&row_referenced,\
                T*&lvald,int*&lrowsd,int*&lcolsd,\
                T*&uvald,int*&ucolsd,int*&urowsd,int n,int nnz,int sweep)
{
    int blocksize=1024;
    int gridsize=nnz/blocksize+1;
	for(int sweepi=0;sweepi<sweep;sweepi++){
           //hipLaunchKernelGGL(ilu0_aij_Ldiag1_Udiag_try_faster2, dim3(gridsize),dim3(blocksize),0,0,
           hipLaunchKernelGGL(ilu0_ij_aw, dim3(gridsize),dim3(blocksize),0,0,\
            lvald,lrowsd,lcolsd,\
            uvald,urowsd,ucolsd,\
            n,rowsd,colsd,row_referenced,vald,nnz);
            hipDeviceSynchronize();
        }

}

__global__ void ilu0_aij_Ldiag1_Udiag_try_faster2_shared(T*lval,int*lrows,int*lcols,T*uval,int*urows,int*ucols,int n,\
    int*rows,int*cols,int*row_reference,T*val,int nnz)
{
    //claim 8
    //input 12
    __shared__ T slval[5000];

    __shared__ int slcols[5000];

int offset;
{
    
    int rs1=blockDim.x*blockIdx.x;
    int rs2=blockDim.x*blockIdx.x+blockDim.x-1;
    //rs1=rs1<(nnz-1)?rs1:(nnz-1);
    rs2=rs2<(nnz-1)?rs2:(nnz-1);
    rs1=row_reference[rs1];
    rs2=row_reference[rs2];
    
    offset=lrows[rs1];
    int mov_size=lrows[rs2+1]-lrows[rs1];

	for(int ti=threadIdx.x;ti<mov_size;ti+=blockDim.x){
	slval[ti]=lval[offset+ti];
    slcols[ti]=lcols[offset+ti];
	}
}
__syncthreads();
	
	//malkl[k]=mov_size;
   int k=blockDim.x*blockIdx.x+threadIdx.x;
    if(k<nnz){
	//malkl[k]=lrows[start_row];
        int i=row_reference[k];
        int j=cols[k];
        T tempv=val[k];
    
        int index_u=ucols[j];
        //int index_l=lrows[i];

        int index_l=lrows[i]-offset;
        //index_u=not changed;
    
        int lj,ui;
        ui=urows[index_u];
        lj=slcols[index_l];

        int check_last=(i<=j)?i:j;
        int&check_id=(i<=j)?ui:lj;

        while(check_id<check_last){

            tempv=(lj==ui)?(tempv-slval[index_l]*uval[index_u]):tempv;

            index_l=(lj<=ui)?index_l+1:index_l;
            index_u=(lj>=ui)?index_u+1:index_u;
            
            lj=slcols[index_l];
            ui=urows[index_u];
        }
    
    
        if(j<i){
            lval[index_l+offset]=tempv/uval[ucols[j+1]-1];
        }
        else{
            uval[index_u]=tempv;
        }
    
    
    }
}

void parilu_fact_shared(T*&vald,int*&rowsd,int*&colsd,int*&row_referenced,\
                T*&lvald,int*&lrowsd,int*&lcolsd,\
                T*&uvald,int*&ucolsd,int*&urowsd,int n,int nnz,int sweep)
{
    int blocksize=1024;
    int gridsize=nnz/blocksize+1;

	for(int sweepi=0;sweepi<sweep;sweepi++){
           hipLaunchKernelGGL(ilu0_aij_Ldiag1_Udiag_try_faster2_shared, dim3(gridsize),dim3(blocksize),0,0,\
            lvald,lrowsd,lcolsd,\
            uvald,urowsd,ucolsd,\
            n,rowsd,colsd,row_referenced,vald,nnz);
            hipDeviceSynchronize();
        }

}

__global__ void Lcsr_trsv_sync_free(T*lval,int*lrows,int*lcols,int n,T*x,T*y,volatile int*set){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int i=gid/warpSize;
    int lane_id=threadIdx.x%warpSize;
    int warp_id=threadIdx.x/warpSize;
    //__shared__ T tsum[10];
    //tsum[0]=0;
    if(i>=n)return;
    T asum=0;
    for(int j=lrows[i]+lane_id;j<lrows[i+1]-1;j+=warpSize){
        int col=lcols[j];
        while(set[col]!=1){
        __threadfence();
        }
        asum+=(x[col]*lval[j]);
    }
    
    for(int s=warpSize/2;s>0;s/=2){
        //if shfl()accept double?
        
        asum+=__shfl_down(asum,s);
        //asum=temp;    
    }
    
    //atomicAdd(tsum,asum);

    if(lane_id==0){
	//asum=tsum[0];
        x[i]=y[i]-asum;
        //x[i]=(y[i]-asum)/lval[lrows[i+1]-1];
        __threadfence();
        set[i]=1;
    }
}


void lcsr_trsv(T*&lvald,int*&lrowsd,int*&lcolsd,T*&xd,T*&yd,int n)
{
    int*setd;
    hipMalloc(&setd,sizeof(int)*n);
    hipMemset(setd,0,sizeof(int)*n);
    //hipMemset(xd,0,sizeof(T)*n);
    //printf("warpSize=%d\n",warpSize);
    int blocksize=warpSize*16;
    int gridsize=(n+(blocksize/warpSize)-1)/(blocksize/warpSize);
    //record_time("spmv");
//Lcsr_trsv_sync_free(T*lval,int*lrows,int*lcols,int n,T*x,T*y,int*set)
    hipLaunchKernelGGL(Lcsr_trsv_sync_free,dim3(gridsize),dim3(blocksize),0,0,\
    lvald,lrowsd,lcolsd,n,xd,yd,setd
    );
    hipDeviceSynchronize();
    hipFree(setd);
}

void lcsr_trsv_setd(T*&lvald,int*&lrowsd,int*&lcolsd,T*&xd,T*&yd,int n,int*setd)
{
    //int*setd;
    //hipMalloc(&setd,sizeof(int)*n);
    hipMemset(setd,0,sizeof(int)*n);
    //hipMemset(xd,0,sizeof(T)*n);
    //printf("warpSize=%d\n",warpSize);
    int blocksize=warpSize*16;
    //int gridsize=n/(blocksize/warpSize)+1;
    int gridsize=(n+(blocksize/warpSize)-1)/(blocksize/warpSize);
    //record_time("spmv");
//Lcsr_trsv_sync_free(T*lval,int*lrows,int*lcols,int n,T*x,T*y,int*set)
    hipLaunchKernelGGL(Lcsr_trsv_sync_free,dim3(gridsize),dim3(blocksize),0,0,\
    lvald,lrowsd,lcolsd,n,xd,yd,setd
    );
    hipDeviceSynchronize();
    //hipFree(setd);
}

__global__ void Ucsc_trsv_sync_free_pre(int*ucols,int*urows,int n,int*d_indegree){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    //if(gid<n)d_indegree[gid]=0;
    //__threadfence();
    if(gid>=ucols[n])return;
    atomicAdd(d_indegree+urows[gid],1);
}

__global__ void Ucsc_trsv_sync_free(T*uval,int*ucols,int*urows,int n,T*y,T*x,T*d_left_sum,volatile int*d_indegree){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int i=n-1-gid/warpSize;
    int wp_block=blockDim.x/warpSize;
    int this_start=blockIdx.x*wp_block;

    //if(i<0)return;
    if(i>=0){
    int lane_id=(warpSize-1)&threadIdx.x;
    //asm("prefetch.global.L2 [%0];"::"r"(urows[ucols[i] + lane_id]));
    while(1!=d_indegree[i])
    {
        __threadfence();
    }
    
    T tx=(y[i]-d_left_sum[i])/uval[ucols[i+1]-1];

    for(int j=ucols[i+1]-1-lane_id;j>=ucols[i];j-=warpSize){
        int row=urows[j];

        atomicAdd((T*)(d_left_sum+row),uval[j]*tx);
        atomicSub((int*)(d_indegree+row),1);

    }
    if(lane_id==0)x[i]=tx;
    }
}

void ucsc_trsv(T*&uvald,int*&ucolsd,int*&urowsd,T*&x,T*&y,int n,int nnzu){
//void Ucsc_trsv_v1(T*uval,int*ucols,int*urows,int n,T*x,T*y)
    T*d_left_sum;
    int*d_indegree;
    hipMalloc(&d_left_sum,sizeof(T)*n);
    hipMalloc(&d_indegree,sizeof(int)*n);

    hipMemset(d_left_sum,0,sizeof(T)*n);
    hipMemset(d_indegree,0,sizeof(int)*n);
    int blocksize=warpSize*16;
    int gridsize=(nnzu+blocksize-1)/blocksize;
    //Ucsc_trsv_sync_free_pre(int*ucols,int*urows,int n,int*d_indegree)
    //record_time("Umalloc_and_set");
    hipLaunchKernelGGL(Ucsc_trsv_sync_free_pre,dim3(gridsize),dim3(blocksize),0,0,\
    ucolsd,urowsd,n,d_indegree);
    hipDeviceSynchronize();
    //Ucsc_trsv_sync_free(T*uval,int*ucols,int*urows,int n,T*y,T*x,T*d_left_sum,int*d_indegree)
    blocksize=warpSize*16;
    gridsize=(n+(blocksize/warpSize)-1)/(blocksize/warpSize);
    hipLaunchKernelGGL(Ucsc_trsv_sync_free,dim3(gridsize),dim3(blocksize),0,0,\
    uvald,ucolsd,urowsd,n,y,x,d_left_sum,d_indegree);

    //hipDeviceSynchronize();
    hipFree(d_left_sum);
    hipFree(d_indegree);
}

void ucsc_trsv_leftsum_indegree(T*&uvald,int*&ucolsd,int*&urowsd,T*&x,T*&y,int n,int nnzu,\
            T*d_left_sum,int*d_indegree){
//void Ucsc_trsv_v1(T*uval,int*ucols,int*urows,int n,T*x,T*y)
    //T*d_left_sum;
    //int*d_indegree;
    //hipMalloc(&d_left_sum,sizeof(T)*n);
    //hipMalloc(&d_indegree,sizeof(int)*n);

    hipMemset(d_left_sum,0,sizeof(T)*n);
    //hipMemset(d_indegree,0,sizeof(int)*n);
    //int blocksize=warpSize*16;
    //int gridsize=nnzu/blocksize+1;
    //Ucsc_trsv_sync_free_pre(int*ucols,int*urows,int n,int*d_indegree)
    //record_time("Umalloc_and_set");
    //hipLaunchKernelGGL(Ucsc_trsv_sync_free_pre,dim3(gridsize),dim3(blocksize),0,0,\
    ucolsd,urowsd,n,d_indegree);
    //hipDeviceSynchronize();
    
    //Ucsc_trsv_sync_free(T*uval,int*ucols,int*urows,int n,T*y,T*x,T*d_left_sum,int*d_indegree)
    int blocksize=warpSize*16;
    int gridsize=n/(blocksize/warpSize)+1;
    hipLaunchKernelGGL(Ucsc_trsv_sync_free,dim3(gridsize),dim3(blocksize),0,0,\
    uvald,ucolsd,urowsd,n,y,x,d_left_sum,d_indegree);

    hipDeviceSynchronize();
    //hipFree(d_left_sum);
    //hipFree(d_indegree);
}



void free_on_gpu(T*&lvald,int*&lrowsd,int*&lcolsd,\
                T*&uvald,int*&ucolsd,int*&urowsd,\
                int*&row_referenced){
                    hipFree(lvald);
                    hipFree(lrowsd);
                    hipFree(lcolsd);
                    hipFree(uvald);
                    hipFree(ucolsd);
                    hipFree(urowsd);
                    hipFree(row_referenced);
                }

void do_factorization(T*val,int*rows,int*cols,int n,int nnz,\
										T*&lvald,int*&lrowsd,int*&lcolsd,int&nnzl,\
                                        T*&uvald,int*&ucolsd,int*&urowsd,int&nnzu,\
                                        int*&row_referenced,int sweep)
{
	T*vald;
    int*rowsd;
    int*colsd;
	hipMalloc(&vald,sizeof(T)*nnz);
    hipMalloc(&rowsd,sizeof(int)*(n+1));
    hipMalloc(&colsd,sizeof(int)*nnz);
    
    hipMemcpyHtoD(vald,val,sizeof(T)*nnz);
    hipMemcpyHtoD(rowsd,rows,sizeof(int)*(n+1));
    hipMemcpyHtoD(colsd,cols,sizeof(int)*nnz);
    
    //parilu_pre_set(T*d_val,int*d_rows,int*d_cols,int n,int nnz,\
T*&d_lval,int*&d_lrows,int*&d_lcols,int&gnnzl,\
T*&d_uval,int*&d_ucols,int*&d_urows,int&gnnzu,\
int*&d_row_reference);
	
    parilu_pre_set(vald,rowsd,colsd,n,nnz,\
    lvald,lrowsd,lcolsd,nnzl,\
    uvald,ucolsd,urowsd,nnzu,\
    row_referenced);
    
    //void parilu_fact(T*&vald,int*&rowsd,int*&colsd,int*&row_referenced,\
                T*&lvald,int*&lrowsd,int*&lcolsd,\
                T*&uvald,int*&ucolsd,int*&urowsd,int n,int nnz,int sweep)
                
	parilu_fact(vald,rowsd,colsd,row_referenced,\
    			lvald,lrowsd,lcolsd,\
                uvald,ucolsd,urowsd,n,nnz,sweep);
    
    hipFree(vald);
    hipFree(rowsd);
    hipFree(colsd);
                                        

}

void do_solve(T*x,T*y,int n,\
						T*lvald,int*lrowsd,int*lcolsd,\
                        T*uvald,int*ucolsd,int*urowsd,int nnzu){

T*xd;
T*yd;
T*xd_inter;
hipMalloc(&xd,sizeof(T)*n);
hipMalloc(&yd,sizeof(T)*n);
hipMalloc(&xd_inter,sizeof(T)*n);

hipMemcpyHtoD(yd,y,sizeof(T)*n);

lcsr_trsv(lvald,lrowsd,lcolsd,xd_inter,yd,n);


//ucsc_trsv(T*&uvald,int*&ucolsd,int*&urowsd,T*&x,T*&y,int n,int nnz)

ucsc_trsv(uvald,ucolsd,urowsd,xd,xd_inter,n,nnzu);


hipMemcpyDtoH(x,xd,sizeof(T)*n);

hipFree(xd);
hipFree(yd);
hipFree(xd_inter);

}

#undef T
#endif
