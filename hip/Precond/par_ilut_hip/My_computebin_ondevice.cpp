#include "hip/hip_runtime.h"
#include "My_computebin_ondevice.h"


__global__ void getrowLenth(int*rowAptr,int n,int*ans){
	int volatile tid=threadIdx.x+blockIdx.x*blockDim.x;
	for(int i=tid;i<n;i+=gridDim.x*blockDim.x){
		int a=rowAptr[i+1];
		int b=rowAptr[i];
		ans[i]=a-b;
		//__syncthreads();
	}
}


__global__ void prefix_sumof_gridprefix(int*queue_one,int*grid_prefix){
	
	int volatile blkidxx=blockIdx.x;
	if(blkidxx<14){
		int idx=threadIdx.x;
		for(int i=1;i<blockDim.x;i<<=1){
			int tmp=0;
			if(idx-i>=0){
				tmp=grid_prefix[getidx(idx,blkidxx)]+grid_prefix[getidx(idx-i,blkidxx)];
			}
			__threadfence();
			__syncthreads();
			if(idx-i>=0){
				grid_prefix[getidx(idx,blkidxx)]=tmp;
			}
			__threadfence();
			__syncthreads();
		}
	}/*
	__syncthreads();
	__threadfence();
	if(blockIdx.x==0){	
	if(threadIdx.x==0){
		int atmp=0;
		__threadfence();
		for(int i=0;i<14;i++){
		queue_one[i]=atmp;
		atmp+=(grid_prefix[14*(blockDim.x-1)+i]);
		}
	}	
	}
	*/
}

__global__ void add_gridprefix(int*queue_one,int*grid_prefix,int lasgblkdim){
	if(threadIdx.x==0){
		int atmp=0;
		for(int i=0;i<14;i++){
			queue_one[i]=atmp;
			atmp+=grid_prefix[14*(lasgblkdim-1)+i];
		}
	}
}

__global__ void set_queue(int*queue,int*queue_one,int*grid_prefix,int*rowLenth,int n,int*queue_idx,int*toshow){
	
	int workpergrid=(n+gridDim.x-1)/gridDim.x;
	rowLenth=workpergrid*blockIdx.x+rowLenth;
	queue_idx=workpergrid*blockIdx.x+queue_idx;
	int group_per_grid=(workpergrid+blockDim.x-1)/blockDim.x;

//	block_groupprefix=block_groupprefix+group_per_grid*14;

	int workthisgroup=workpergrid<(n-workpergrid*(int)(blockIdx.x))?workpergrid:(n-workpergrid*(int)(blockIdx.x));
	int localq1[14];
	int localgridprefix[14];
	grid_prefix=grid_prefix+((int)(blockIdx.x)-1)*14;
	//localgridprefix[0]=0;
	for(int i=0;i<14;i++){
		localq1[i]=queue_one[i];
		localgridprefix[i]=((int)(blockIdx.x))>0?grid_prefix[i]:0;
	}	
	int volatile rowLenthoffset=workpergrid*blockIdx.x;
	//queue_idx=queue_idx+rowLenthoffset;
	__syncthreads();

	for(int i=threadIdx.x;i<workthisgroup;i+=blockDim.x){
		int tmp=rowLenth[i];
		//int tag=0;
		//if(tmp>0){
		int tag=gettag_includeadd(tmp);
		//}
		int offset=queue_idx[i]+localq1[tag]+localgridprefix[tag];
		queue[offset]=i+rowLenthoffset;
	//	toshow[i+rowLenthoffset]=100000*(blockIdx.x+1)+localgridprefix[tag];
		//__syncthreads();
	}
}