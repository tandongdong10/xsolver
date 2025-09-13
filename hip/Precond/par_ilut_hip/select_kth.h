#include"timetest.h"

constexpr int tree_height=8;
constexpr int bucket_size=(1<<tree_height);
constexpr int numperthread=16;
/*void call_radius_sort(double*dval,int nnzD){
	void*d_temp_storage=NULL;
	unsigned long int temp_storage_bytes=0;
	hipcub::DeviceRadixSort::SortKeysDescending(d_temp_storage,temp_storage_bytes,dval,dval,nnzD);
	hipMalloc(&d_temp_storage,temp_storage_bytes);
	hipcub::DeviceRadixSort::SortKeysDescending(d_temp_storage,temp_storage_bytes,dval,dval,nnzD);
	hipFree(d_temp_storage);
}*/
#define warpnum 16
inline __device__ void subwarp_prefix_sum(int ts,int& __restrict__ prefix,int& __restrict__ total,int volatile lid){
    total=ts;
    prefix=0;
    for(int i=1;i<warpSize;i*=2){
        int o=__shfl_xor(total,i);
        total+=o;
        prefix+=((i&lid)?o:0);
    }
}

inline __device__ int warp_sum(int val,int lid){
	for(int i=1;i<warpSize;i<<=1){
	val+=__shfl_xor(val,i);
	}
	return val;
}


__device__ inline void sort2(double* in, int i, int j, bool odd) {
    auto ei = in[i];
    auto ej = in[j];
    if (odd != (ej < ei)) {
        in[i] = ej;
        in[j] = ei;
    }
}
constexpr int sample_height=10;
constexpr int samplesize=1<<sample_height;

__device__ void bitonic_sort2(double*val){
    int volatile tid=threadIdx.x;
    for(int i=0;i<sample_height;i++){

        int halfsize=1<<i;
        int odd=tid&(halfsize);
        for(int j=halfsize;j!=0;j>>=1){
            
            int low=(j-1)&tid;
            int up=tid^low;

            int tsidx=low|(up<<1);

            __syncthreads();
            if(tid<(1<<(sample_height-1))){
/*
		int volatile ia=tsidx;
		int volatile ja=tsidx|j;
                //int volatile atsidx=tsidx|j;
                auto volatile ei=val[ia];
                auto volatile ej=val[ja];
                if(odd!=(ej<ei)){
                    val[ia]=ej;
                    val[ja]=ei;
                }
		__syncthreads();
*/
		sort2(val,tsidx,tsidx|j,odd);
            }   
        }
    }
}

constexpr int bitonic_cutoff_log2=10;



__device__ void bitonic_sort(double* in) {
    int volatile idx = threadIdx.x;
    // idx has the form | high | low | where /low/ has /round/ bits
    for (int round = 0; round < bitonic_cutoff_log2; ++round) {
        // the lowest bit of /high/ decides the sort order
        bool odd = idx & (1 << round);
        for (int bit = 1 << round; bit != 0; bit >>= 1) {
            // idx has the form | upper | lower | where /lower/ initially
            // has /round/ bits and gradually shrink
            int lower = idx & (bit - 1);
            int upper = idx ^ lower;
            // we then sort the elements | upper | 0/1 | lower |
            int sort_idx = lower | (upper << 1);

            __syncthreads();

            if (idx < 1 << (bitonic_cutoff_log2 - 1)) {
                sort2(in, sort_idx, sort_idx | bit, odd);
            }
        }
    }
}


__global__ void not_random_select(double*val,double*tree,int nnz){
int volatile tid=threadIdx.x;
__shared__ double sample[samplesize];
__shared__ double acsamp[bucket_size];
//int offset=nnz/blockDim.x
unsigned long long todive=nnz*tid;
unsigned int todivesamplesize=todive/samplesize;
if(todivesamplesize>=nnz)todivesamplesize=todivesamplesize%nnz;
sample[tid]=val[todivesamplesize];
__syncthreads();


bitonic_sort2(sample);
__syncthreads();

if(tid<bucket_size)tree[tid]=sample[tid];
//return;
if(tid<bucket_size){
	acsamp[tid]=sample[tid*samplesize/bucket_size];
	tree[tid-1+bucket_size]=acsamp[tid];
}
__syncthreads();
//return ;
//if(tid<bucket_size)acsamp[tid]=tid;
//__syncthreads();

if(tid<bucket_size&&tid>0){
	//tid+=1;
	int level=tree_height-__ffs(tid);
	int idxpl=tid>>__ffs(tid);
	int pl=(1<<level)-1;	
	tree[pl+idxpl]=acsamp[tid];
}
}

void build_tree(double*val,double*tree,int nnz){
//todo
	hipLaunchKernelGGL(not_random_select,dim3(1),dim3(samplesize),0,0,val,tree,nnz);
	hipDeviceSynchronize();
/*std::cout<<"build_tree\n";
char c;
std::cin>>c;
if(c=='a'){
	for(int i=0;i<bucket_size;i++){
		std::cout<<i<<" "<<tree[i];
		getchar();
	}
}
std::cout<<"treefi\n";
*/
}

inline __device__ int getbkid(double val,double*tree,int i){
	int idx=0;
	//int eval=val[i];
	#pragma unroll
	for(int l=0;l<tree_height;l++){
	int child=val>tree[idx];
	int equal=(val==tree[idx])&(((i>>l)&1));
	idx=2*idx+1+child+equal;
	}
	return idx;
}

constexpr long long int onemask=0xff;
__global__ void count_bucket(double*val,int nnz,int* __restrict__ bucket,int move,int*marker,double*tree){
   
    int volatile id=threadIdx.x;
    __shared__ int shbk[bucket_size];

    constexpr int treesize=bucket_size-1;
    __shared__ double shtree[treesize];

    if(id<treesize)shtree[id]=tree[id];

    if(id<bucket_size)shbk[id]=0;
    __syncthreads();
    int volatile gdm=gridDim.x;

    int start=threadIdx.x+blockDim.x*blockIdx.x*numperthread;
    int end=blockDim.x*(blockIdx.x+1)*numperthread;
    end=min(end,nnz);
    for(int i=start;i<end;i+=blockDim.x){
	//val[i]=i;
	//double fabsvali=val[i]>0?val[i]:-val[i];
	//long long int*todo=(long long int*)(val+i);
        //int bkid=(((*todo)>>move)&onemask);
        int bkid=getbkid(val[i],tree,i)-treesize;
//        int temp_test=val[i];
//	int bkid=temp_test%256;
        atomicAdd(shbk+bkid,1);
        marker[i]=bkid;
//	val[i]=bkid+10000*val[i];
    }
	__syncthreads();
//return;
	int volatile tag=0;
//	if(id<256)tag=1;

    if(id<bucket_size)
    {
	bucket[blockIdx.x+(id*gdm)]=shbk[id];
    }
}

__global__ void sum_bucket_and_prefix_to_edit(int*bucket,int*totalcount,int numperblk){
    int blkid=blockIdx.x;
    int lid=threadIdx.x&(warpSize-1);
    int wid=threadIdx.x/warpSize;
    //int warpnum=blockDim.x/warpSize;

    int bucketid=blockIdx.x;

    __shared__ int warp_sums[warpnum];
    int*thisblock=bucket+bucketid*numperblk;

    int sum=0;
/*
    for(int i=threadIdx.x;i<numperblk;i+=blockDim.x){
	__syncthreads();
        int tocal=i<numperblk?thisblock[i]:0;
        int warptotal=0;
        int warpprefix=0;
	int ws=warp_sum(tocal,lid);
        subwarp_prefix_sum(tocal,warpprefix,warptotal,lid);
        thisblock[i]=warpprefix+sum;
        sum+=warptotal;
	//thisblock[i]=bucketid*numperblk;
    }
*/
auto wthisblock=thisblock;
    for(int i=lid;i<numperblk;i+=warpSize){
	int warptotal=0;
	int warpprefix=0;
	int tocal=i<numperblk?wthisblock[i]:0;
	subwarp_prefix_sum(tocal,warpprefix,warptotal,lid);
	wthisblock[i]=warpprefix+sum;
	sum+=warptotal;
	wthisblock+=warpSize;
	}
//return;
/*
    if(lid==0)warp_sums[wid]=sum;

    __syncthreads();

    if(wid==0){
        for(int i=lid;i<warpnum;i+=warpSize){
	    //__syncthreads();
            int prefix=0;
            int total=0;
	    int vali=i<warpnum?warp_sums[i]:0;
            subwarp_prefix_sum(vali,prefix,total,lid);

            warp_sums[i]=prefix;

            if(lid==0)totalcount[bucketid]=vali;
        }
    }
    __syncthreads();

    int toadd=warp_sums[wid];
    for(int i=threadIdx.x;i<numperblk;i+=blockDim.x){
        thisblock[i]+=toadd;
    }
*/
}
__global__ void testforthread(int* __restrict__ toprint){
int tid=threadIdx.x;
//int volatile  wid=threadIdx.x/warpSize;
//int volatile  lid=threadIdx.x%warpSize;
//toprint[tid]=tid;
//__syncthreads();
//int  prefix=0;
//int  sum=0;
//int val=toprint[tid];
//__syncthreads();
//subwarp_prefix_sum(val,prefix,sum,lid);

__syncthreads();
if(tid<256)toprint[blockIdx.x+tid*256]=tid;

}
__global__ void sum_bucket_and_prefix2(int* __restrict__ bucket,int* __restrict__ totalcount,int numperblk,int*toprint){
    //int blkid=blockIdx.x;
    int pid=threadIdx.x+blockIdx.x*blockDim.x;
    int volatile lid=threadIdx.x&(warpSize-1);
    int volatile wid=(threadIdx.x)/warpSize;
    //int warpnum=blockDim.x/warpSize;

    int volatile bucketid=blockIdx.x;
    int volatile workperwarp=(numperblk+warpSize-1)/warpSize;
    //maybe workperwarp=numperblk/warpnum;?

    __shared__ int volatile warp_sums[warpnum];
    int*thisblock=bucket+bucketid*numperblk;

    int volatile suma=0;
    int volatile baseidx=wid*workperwarp*warpSize;
	
    for(int step=0;step<workperwarp;step++){
        int idx=baseidx+lid+step*warpSize;
        int tocal=idx<numperblk?thisblock[idx]:0;

        int  warptotal=0;
        int  warpprefix=0;
	//int ws=warp_sum(tocal,lid);
	//__syncthreads();
        subwarp_prefix_sum(tocal,warpprefix,warptotal,lid);
        if(idx<numperblk){
		thisblock[idx]=warpprefix+suma;
	}
        suma=suma+warptotal;

	//toprint[bucketid*warpnum+wid]=suma;
    }
	//__syncthreads();
     if(lid==0){
	warp_sums[wid]=suma;
	}

    __syncthreads();
    if(wid==0){
        //int volatile tocal=0;
        for(int i=lid;i<warpnum;i+=warpSize){
            int prefix=0;
            int total=0;
            int tocal=i<warpnum?warp_sums[i]:0;
	  //  __syncthreads();
            subwarp_prefix_sum(tocal,prefix,total,lid);

            warp_sums[i]=prefix;
	//totalcount[blockIdx.x*64+i]=tocal;
            if(lid==0)totalcount[bucketid]=total;
        }
    }
    __syncthreads();
//    return;
    int volatile toadd=warp_sums[wid];
    for(int step=0;step<workperwarp;step++){
        int idx=lid+step*warpSize+baseidx;
        int val=idx<numperblk?thisblock[idx]:0;
        if(idx<numperblk){
            thisblock[idx]+=toadd;
        }
    }
}




void build_bucket(double*val,int nnz,int*bucket,int move,int*marker,int*totalcount,int*totalprefixsum,double*tree){
/*
int*toprint;
hipMalloc(&toprint,sizeof(int)*256*203);
	hipLaunchKernelGGL(testforthread,dim3(203),dim3(1024),0,0,\
    toprint);
    hipDeviceSynchronize();
	
return;
*/
    //build_tree
    int blkDim=1024;
    int totalthreads=(nnz+numperthread-1)/numperthread;
    int zushu=(totalthreads+blkDim-1)/blkDim;
    int gridDim=zushu;
//std::cout<<"build_tree";
//getchar();
	build_tree(val,tree,nnz);
//getchar();
	//std::cout<<"g::"<<gridDim<<" "<<numperthread<<" "<<zushu;
    //count_bucket(double*val,int nnz,int*bucket,int move,int*marker)
    hipLaunchKernelGGL(count_bucket,dim3(gridDim),dim3(blkDim),0,0,\
    val,nnz,bucket,move,marker,tree);
     
	hipDeviceSynchronize();
//	std::cout<<"after_count_bucket"<<"\n";
//getchar();
/*
std::cout<<"val and marker  ";
char c;
std::cin>>c;
if(c=='a'){
	std::cout<<bucket_size*zushu<<" "<<bucket_size*zushu*10<<"kkkkkaaakkddaa\n";
	for(int i=0;i<(bucket_size*zushu);i++){
		if(bucket[i]!=0||1){
		std::cout<<i<<" "<<val[i]<<" "<<marker[i]<<"\n";
		getchar();
		}
	}
}
*/
//return;
	/*int sum=0;
	zushu=8;
	for(int i=0;i<bucket_size*zushu;i++){
		bucket[i]=i;
		sum+=i;
	}*/
/*
	for(int i=0;i<bucket_size;i++){
		double itod=i;
		long long *ma=(long long*)&itod;
		std::cout<<val[i]<<" "<<(((*ma)>>move)&0xff)<<"\n";
	}
     getchar();
*/
    //sum_bucket_and_prefix(int*bucket,int*totalcount,int numperblk)
//   getchar(); 
/*	int*toprint;
	hipMalloc(&toprint,sizeof(int)*9999999);
	hipMemset(toprint,0,sizeof(int)*9999999);
	std::cout<<"zhushu===="<<zushu<<"\n";
*/	
	int *tempi=NULL;
	hipLaunchKernelGGL(sum_bucket_and_prefix2,dim3(bucket_size),dim3(blkDim),0,0,\
    bucket,totalcount,zushu,tempi);
    hipDeviceSynchronize();	
//	std::cout<<"aftersumandprefix"<<"\n";
/*	for(int i=0;i<bucket_size;i++){
		
		std::cout<<i<<" "<<totalcount[i];
		getchar();
		
	}
*/
  //prefix_sum
    int nnza;
//	std::cout<<"totalcount63,64:"<<totalcount[63]<<" "<<totalcount[64]<<"\n",getchar();
    prefix_sum(bucket_size+1,totalcount,nnza);

//	std::cout<<"prefix_sum"<<"\n";
}

__global__ void getKth_bucket(int*bucket,int k,int*ans){
    if(threadIdx.x!=0)return;
    int l=0;
    int r=bucket_size;
    while(l<r){
        int mid=r-((r-l)>>1);
        if(k<bucket[mid]){
            r=mid-1;
        }
        else{
            l=mid;
        }
    }
    (*ans)=l;
    (*(ans+1))=bucket[l];
    (*(ans+2))=bucket[l+1]-bucket[l];
}

void call_getKth_bucket(int*bucket,int k,int*ans,int*tans){
	//int*tans;
	//int*knb;
	//hipMalloc(&tans,3*sizeof(int));
	//hipMalloc(&knb,sizeof(int));
	hipLaunchKernelGGL(getKth_bucket,dim3(1),dim3(64),0,0,\
	bucket,k,tans);
	
	hipMemcpyDtoH(ans,tans,3*sizeof(int));
	//hipMemcpyDtoH(&newbknnz,knb,sizeof(int));
	//newbknnz=bucket[ans+1]-bucket[ans];

	//hipFree(tans);
	//hipFree(knb);
}

__global__ void extract_kth_bucket(int* __restrict__ bucket,int k,int*marker,double* __restrict__ newbucket,double*val,int nnz){
    __shared__ int offset;
    if(threadIdx.x==0){
        offset=bucket[blockIdx.x+k*gridDim.x];
    }
    int start=threadIdx.x+blockDim.x*blockIdx.x*numperthread;
    int end=blockDim.x*(blockIdx.x+1)*numperthread;
    end=min(end,nnz);
    for(int i=start;i<end;i+=blockDim.x){
	__syncthreads();
        if(marker[i]==k){
            int of=atomicAdd(&offset,1);
            newbucket[of]=val[i];
        }
    }
}

void call_extract_kth_bucket(int*bucket,int k,int*marker,double*newbucket,double*val,int nnz){
    int blkDim=1024;	
    int totalthreads=(nnz+numperthread-1)/numperthread;
    int zushu=(totalthreads+blkDim-1)/blkDim;
    int gridDim=zushu;
	hipLaunchKernelGGL(extract_kth_bucket,dim3(gridDim),dim3(blkDim),0,0,\
	bucket,k,marker,newbucket,val,nnz);
	hipDeviceSynchronize();
}


void find_kth_element(double*val,int nnz,int k,double&thresold,int*extrasize,int*extramarker){
//void build_bucket(double*val,int nnz,int*bucket,int move,int*marker,int*totalcount,int*totalprefixsum)
    int blkDim=1024;
	//k=1000;
    int totalthreads=(nnz+numperthread-1)/numperthread;
    int zushu=(totalthreads+blkDim-1)/blkDim;
 //std::cout<<"zushu"<<zushu<<"\n";
int*bucket;
//k=121;
double*tree;
int move=56;
int*marker;
int*totalcount;
double*newbucket;
double*oldbucket;
int kans[3];
int nah;
int oldbknnz=nnz;
int newbknnz=nnz;
int kkkk=k;
double kth=k-1;
long long int*mathk=(long long*)(&kth);
int sumk=0;
int*tans;
//std::cout<<bucket_size*zushu<<"\n";
//getchar();
hipMalloc(&tree,sizeof(double)*bucket_size*2);
hipMalloc(&tans,sizeof(int)*3);
hipMalloc(&bucket,sizeof(int)*((bucket_size)*(zushu)));
hipMalloc(&marker,sizeof(int)*nnz);
hipMalloc(&totalcount,sizeof(int)*(bucket_size+1));
hipMalloc(&newbucket,sizeof(double)*nnz);
hipMalloc(&oldbucket,sizeof(double)*nnz);
//hipMemcpyDtoD(oldbucket,val,sizeof(double)*nnz);
//int*totalprefixsum;
//hipMalloc(&totalprefixsum,sizeof(int)*);
//record_time("loop start");


build_bucket(val,nnz,bucket,move,marker,totalcount,totalcount,tree);
call_getKth_bucket(totalcount,k,kans,tans);
nah=kans[0];
newbknnz=kans[2];
k-=kans[1];
sumk+=kans[1];

call_extract_kth_bucket(bucket,nah,marker,newbucket,val,nnz);

double*tmp=oldbucket;
oldbucket=newbucket;
newbucket=tmp;
move-=8;
oldbknnz=newbknnz;
//record_time("iter_before");
{/*
char a;
std::cin>>a;

if(a=='a'){
	for(int i=0;i<oldbknnz;i++){
		int tp=oldbucket[i];
		std::cout<<tp<<"\n";
		char cca=getchar();
		if(cca=='e')break;
	}
}
*/
}

while(newbknnz>1){
//std::cout<<"nebknnz="<<newbknnz<<"\n";
//getchar();
//std::cout<<"new iter___________________________"<<newbknnz<<" "<<oldbknnz<<" "<<newbucket[0]<<"\n";
//getchar();
//std::cout<<"move:"<<move<<" "<<newbknnz<<" " <<k<<"\n";
//std::cout<<"build_bk"<<"\n";

//getchar();

build_bucket(oldbucket,oldbknnz,bucket,move,marker,totalcount,totalcount,tree);
//record_time("buildbk");
//getchar();
//std::cout<<"getkthbk"<<"\n";
//for(int i=0;i<bucket_size+1;i++)std::cout<<i<<" "<<totalcount[i]<<"\n";
//getchar();
call_getKth_bucket(totalcount,k,kans,tans);
nah=kans[0];
newbknnz=kans[2];
k-=kans[1];
sumk+=kans[1];

//record_time("getKth");
//getchar();
//int words=((*mathk)>>move)&0xff;
//std::cout<<"extractkth"<<"  nah="<<nah<<" sumk="<<sumk<<" kkkk="<<kkkk<<"\n";
//if(newbknnz==oldbknnz)break;
call_extract_kth_bucket(bucket,nah,marker,newbucket,oldbucket,oldbknnz);
//record_time("extract");
//std::cout<<"afterextract newbknnz="<<newbknnz<<"\n";
{/*
char a;
std::cin>>a;

if(a=='a'){
	for(int i=0;i<oldbknnz;i++){
		int tp=oldbucket[i];
		std::cout<<tp<<"\n";
		char cca=getchar();
		if(cca=='e')break;
	}
}*/
}
double*tmp=oldbucket;
oldbucket=newbucket;
newbucket=tmp;
move-=8;
oldbknnz=newbknnz;

//getchar();
//record_time("eachLoop");
}

hipMemcpyDtoH(&thresold,oldbucket,sizeof(double));
hipFree(tree);
hipFree(tans);
hipFree(oldbucket);
hipFree(newbucket);
hipFree(bucket);
hipFree(marker);
hipFree(totalcount);
//int mamt=thresold;
//record_time("togetthresold");
//std::cout<<thresold<<" "<<" "<<kkkk<<"\n";	
}
