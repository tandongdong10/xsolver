inline __device__ int gettag_includeadd(int tmp){
	
	if(tmp<=1)return 0;
	if(tmp<=3)return 1;
	if(tmp<=7)return 2;
	if(tmp<=15)return 3;
	if(tmp<=31)return 4;
	if(tmp<=63)return 5;
	if(tmp<=127)return 6;
	if(tmp<=255)return 7;
	if(tmp<=511)return 8;
	if(tmp<=1023)return 9;
	if(tmp<=2047)return 10;
	if(tmp<=4095)return 11;
	return 12;
}

__global__ void getrowLenth(int*rowAptr,int n,int*ans){
	int volatile tid=threadIdx.x+blockIdx.x*blockDim.x;
	for(int i=tid;i<n;i+=gridDim.x*blockDim.x){
		int a=rowAptr[i+1];
		int b=rowAptr[i];
		ans[i]=a-b;
		//__syncthreads();
	}
}

template<int blkdim>
__global__ void calculate_queue_one2(int*queue_one,int n,int*rowLenth,int*queue,int*queue_idx,\
int*grid_prefix,int*block_groupprefix){
	
	int volatile lid=threadIdx.x&(warpSize-1);
	int volatile wid=threadIdx.x/warpSize;
	int volatile workpergrid=(n+gridDim.x-1)/gridDim.x;
	
	//int local_q1[14];
	__shared__ int presum[14];

	if(threadIdx.x<14)presum[threadIdx.x]=0;

	//int volatile  tid=threadIdx.x;
	rowLenth=workpergrid*blockIdx.x+rowLenth;
	queue_idx=workpergrid*blockIdx.x+queue_idx;
	int group_per_grid=(workpergrid+blockDim.x-1)/blockDim.x;

//	block_groupprefix=block_groupprefix+group_per_grid*14;

	int workthisgroup=workpergrid<(n-workpergrid*(int)(blockIdx.x))?workpergrid:(n-workpergrid*(int)(blockIdx.x));
	__syncthreads();
	for(int i=threadIdx.x;i<workthisgroup;i+=blockDim.x){
		//__syncthreads();
		int tmp=rowLenth[i];
		int tag=0;
		//queue_idx[i]=tmp;
		//if(tmp==0)local_q1[0]+=1;
	//	if(tmp>0){
		tag=gettag_includeadd(tmp);
	//	}
		//queue_idx[i]=tag;
		int old=atomicAdd(presum+tag,1);
		//__threadfence_block();
		queue_idx[i]=old;
		//__syncthreads();	

		//careful
		//__threadfence_block();
		/*
		if(threadIdx.x<14){
			block_groupprefix[threadIdx.x]=presum[threadIdx.x];
		}
		block_groupprefix+=14;
		*/
		__syncthreads();
	}

	//return;
	/*
	int wait=1;
	int tc;	
	while(wait<100*blockDim.x){
		wait+=1;
		tc=wait<workpergrid?queue_idx[wait]:0;	
	}
	*/
	//__syncthreads();
	grid_prefix=blockIdx.x*14+grid_prefix;
	__threadfence();
	__syncthreads();
	int volatile thidx=threadIdx.x;
	if(thidx<14){
		//__threadfence();
		//presum[threadIdx.x]=100;
		//int temp=workpergrid<(n-workpergrid*(int)(blockIdx.x))?workpergrid:(n-workpergrid*(int)(blockIdx.x));
		grid_prefix[thidx]=presum[thidx];
	}
	//__syncthreads();
	
}

inline __device__ int getidx(int i,int gridIdxx)
{
	return i*14+gridIdxx;
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
constexpr int mgriddim=256;
constexpr int mblockdim=512;

/*
template<typename T>
void ComputeBin_Device(SparseHostMatrixCSR<T> A, HostVector<uint> &Queue, HostVector<uint> &Queue_one){

	ExMI::WallTime t;
	double v0=t.Seconds();	
	int row_num = A.Height();
	Queue=HostVector<uint>(row_num);
	Queue_one=HostVector<uint>(14);
	HostVector<uint> h_queue(row_num);
	HostVector<uint> h_queue_one(14);

	int *hqueue =(int*) h_queue.Data();
	int *hqueue_one = (int*)h_queue_one.Data();
	int*queue;
	int*queue_one;
	int*drowptr;
	int*rowA;
	int*grid_prefix;
	int*block_groupprefix;
	int*queue_idx;
	//int gridDim=256;
	//int blockDim=512;
	hipMalloc(&drowptr,sizeof(int)*(row_num+1));
	hipMalloc(&rowA,sizeof(int)*(row_num));
	hipMalloc(&grid_prefix,sizeof(int)*14*mgriddim);
	//hipMalloc(&block_groupprefix,sizeof(int)*14*(row_num+mblockdim-1)/mblockdim);
	hipMalloc(&queue_idx,sizeof(int)*(row_num));

	hipMalloc(&queue_one,sizeof(int)*14);
	hipMalloc(&queue,sizeof(int)*row_num);

	cudaMemcpy(drowptr,A.RowStarts().Data(),sizeof(int)*(row_num+1),cudaMemcpyHostToDevice);

	cudaMemset(queue_one,0,sizeof(uint)*14);
	cudaMemset(grid_prefix,0,sizeof(int)*14*mgriddim);

	double v00=t.Seconds();
	getrowLenth<<<dim3(mgriddim),dim3(mblockdim)>>>(drowptr,row_num,rowA);
	hipDeviceSynchronize();
	double v1=t.Seconds();
	//(uint*queue_one,int n,int*rowLenth,uint*queue,int*queue_idx,int*grid_prefix,int*block_groupprefix)
	calculate_queue_one2<mblockdim><<<dim3(mgriddim),dim3(mblockdim)>>>(queue_one,row_num,rowA,queue,queue_idx,\
	grid_prefix,block_groupprefix);
	hipDeviceSynchronize();

	//std::cout<<"prefix_sumof";
	//getchar();
	double v2=t.Seconds();
	prefix_sumof_gridprefix<<<dim3(14),dim3(mgriddim)>>>(queue_one,grid_prefix);
	hipDeviceSynchronize();
    	double v3=t.Seconds();


	int*toshow;
	//hipMalloc(&toshow,sizeof(int)*row_num);
	//set_queue(int*queue,int*queue_one,int*grid_prefix,int*rowLenth,int n,int*queue_idx)
	set_queue<<<dim3(mgriddim),dim3(mblockdim)>>>(queue,queue_one,grid_prefix,rowA,row_num,queue_idx,toshow);

	double v4=t.Seconds();

	
	std::cout<<"v00-v0="<<1e6*(v00-v0)<<" v1-v00="<<1e6*(v1-v00)<<" v2-v1="<<1e6*(v2-v1)<<" v3-v2="<<1e6*(v3-v2)<<" v4-v3="<<1e6*(v4-v3)<<"\n";
	cudaMemcpy(hqueue_one,queue_one,sizeof(int)*14,cudaMemcpyDeviceToHost);
	cudaMemcpy(hqueue,queue,sizeof(int)*row_num,cudaMemcpyDeviceToHost);



	Queue=h_queue;
	//Queue=h_queue;
	Queue_one=h_queue_one;

	return;
}
*/
#include<iostream>

template<typename T>
void myComputeBinAdd_Device(\
T*avals,int*arows,int*acols,int n, \
int*&Queue, int*&Queue_one){

	int row_num = n;

	int*queue;
	int*queue_one;
	int*drowptr;
	int*rowA;
	int*grid_prefix;
	int*block_groupprefix;
	int*queue_idx;
	//int gridDim=256;
	//int blockDim=512;

	hipMalloc(&queue,sizeof(int)*n);
	hipMalloc(&queue_one,sizeof(int)*14);
	//hipMalloc(&drowptr,sizeof(int)*(row_num+1));
	hipMalloc(&rowA,sizeof(int)*(row_num));
	hipMalloc(&grid_prefix,sizeof(int)*14*mgriddim);
	//hipMalloc(&block_groupprefix,sizeof(int)*14*(row_num+mblockdim-1)/mblockdim);
	hipMalloc(&queue_idx,sizeof(int)*(row_num));

	//hipMalloc(&queue_one,sizeof(int)*14);
	//hipMalloc(&queue,sizeof(int)*row_num);

	//hipMemset(queue_one,0,sizeof(int)*14);
	//hipMemset(grid_prefix,0,sizeof(int)*14*mgriddim);

	//getrowLenth<<<dim3(mgriddim),dim3(mblockdim)>>>(arows,row_num,rowA);
	hipLaunchKernelGGL(getrowLenth,dim3(mgriddim),dim3(mblockdim),0,0,\
	arows,row_num,rowA);
	hipDeviceSynchronize();


	/*{
	int*hra=new int[row_num];
	int*hdrptr=new int[row_num+1];

	cudaMemcpy(hra,rowA,sizeof(int)*row_num,cudaMemcpyDeviceToHost);
	cudaMemcpy(hdrptr,arows,sizeof(int)*row_num,cudaMemcpyDeviceToHost);

	for(int i=0;i<row_num;i++){
	std::cout<<hra[i]<<" "<<hdrptr[i];getchar();

	}
	}*/

	//(uint*queue_one,int n,int*rowLenth,uint*queue,int*queue_idx,int*grid_prefix,int*block_groupprefix)
	//calculate_queue_one2<mblockdim><<<dim3(mgriddim),dim3(mblockdim)>>>(queue_one,row_num,rowA,queue,queue_idx,\
	grid_prefix,block_groupprefix);
	hipLaunchKernelGGL(calculate_queue_one2<mblockdim>,dim3(mgriddim),dim3(mblockdim),0,0,\
	queue_one,row_num,rowA,queue,queue_idx,\
	grid_prefix,block_groupprefix);
	hipDeviceSynchronize();
/*
{	std::cout<<"wrtie0:"<<row_num;
	int*hqueue_onea=new int[14*mgriddim];
	hipMemcpy(hqueue_onea,grid_prefix,sizeof(int)*14*mgriddim,hipMemcpyDeviceToHost);
	int tma[14]={0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	
	int atotal=0;
	std::cout<<"hq1:";//getchar();
	for(int i=0;i<14*mgriddim;i++){
		tma[i%14]+=hqueue_onea[i];
		atotal+=hqueue_onea[i];
	}
	for(int i=0;i<14;i++){
		std::cout<<tma[i];
	//	getchar();
	}
	std::cout<<"total="<<atotal;
	//getchar();
	
}
*/


	//prefix_sumof_gridprefix<<<dim3(14),dim3(mgriddim)>>>(queue_one,grid_prefix);
	//getchar();
		
	hipLaunchKernelGGL(prefix_sumof_gridprefix,dim3(14),dim3(mgriddim),0,0,\
	queue_one,grid_prefix);
	hipDeviceSynchronize();
	hipLaunchKernelGGL(add_gridprefix,dim3(1),dim3(64),0,0,\
	queue_one,grid_prefix,mgriddim);
	hipDeviceSynchronize();

/*{
	//getchar();
	int*q1=new int[14];
	std::cout<<"pfxofgrid\n";
	hipMemcpyDtoH(q1,queue_one,sizeof(int)*14);
	for(int i=0;i<14;i++)std::cout<<q1[i]<<"\n";
}*/

	//getchar();
	int*toshow;
	//hipMalloc(&toshow,sizeof(int)*row_num);
	//set_queue(int*queue,int*queue_one,int*grid_prefix,int*rowLenth,int n,int*queue_idx)
	//set_queue<<<dim3(mgriddim),dim3(mblockdim)>>>(queue,queue_one,grid_prefix,rowA,row_num,queue_idx,toshow);
	hipLaunchKernelGGL(set_queue,dim3(mgriddim),dim3(mblockdim),0,0,\
	queue,queue_one,grid_prefix,rowA,row_num,queue_idx,toshow);
	hipDeviceSynchronize();
/*{
	std::cout<<"write1";
	int*hqueue_one=new int[14];
	int*hqueue=new int[row_num];
	hipMemcpy(hqueue_one,queue_one,sizeof(int)*14,hipMemcpyDeviceToHost);
	hipMemcpy(hqueue,queue,sizeof(int)*row_num,hipMemcpyDeviceToHost);
	{
	std::cout<<"hq1:",getchar();
	for(int i=0;i<14;i++){
		std::cout<<hqueue_one[i];getchar();
	}


	for(int i=0;i<row_num;i++){
		std::cout<<hqueue[i];getchar();
	}
	}

}*/
	Queue=queue;
	//Queue=h_queue;
	Queue_one=queue_one;


	hipFree(grid_prefix);
	hipFree(queue_idx);
	hipFree(rowA);
	return;
}
