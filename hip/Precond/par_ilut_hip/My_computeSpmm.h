#include"My_computebin_ondevice.h"
#include"My_Mulwarp_predictsize.h"
#include"My_Mulwarp.h"

inline  int hostgettag(int tmp){
	
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


//#include <thrust/detail/static_assert.h>
#include <thrust/scan.h>
template<typename T>
static void ScanExclusive(T*crows,int n){
    thrust::exclusive_scan(thrust::device,crows,crows+n+1,crows);
}

template<typename T>
static void DifMulAddRMerge(\
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
T*&cvals,int*&crows,int*&ccols,\
int m,int k,int n,int&nnz,\
int*DQueue,int dq_len,\
int*DQueue_one,int dq1_len)
{

	hipMalloc(&crows,sizeof(int)*(m+1));

    int*h_queue_one=new int[dq1_len];
    hipMemcpy(h_queue_one,DQueue_one,sizeof(int)*dq1_len,hipMemcpyDeviceToHost);
   
	//for(int i=0;i<14;i++)
	//std::cout<<h_queue_one[i]<<"\n";
//record_time("predict_csize_start");
	PredictCSize(crows,\
    avals,arows,acols,\
    bvals,brows,bcols,\
    dvals,drows,dcols,\
    alpha,\
    m,k,n,\
    DQueue,DQueue_one, h_queue_one);
//	std::cout<<"afterpredictCsize\n";
//record_time("predict_csize_end");
	ScanExclusive(crows,m);
//record_time("exclusive_scan");
	//for(int i=m-10;i<=m;i++)std::cout<<crows[i];
//	std::cout<<"afterscan\n";
/*{

cudaDeviceSynchronize();
std::cout<<"PredictCsize"<<"\n";getchar();

int*hcrows=new int[m];
cudaMemcpy(hcrows,crows,sizeof(int)*(m+1),cudaMemcpyDeviceToHost);
for(int i=0;i<m+1;i++){
	std::cout<<hcrows[i],getchar();
}

}*/
    //int nnz;
    hipMemcpy(&nnz,crows+m,sizeof(int),hipMemcpyDeviceToHost);
	//int nnz = crows[m];
//	std::cout << "Fast C nonzerod:"<<nonZeros<<std::endl;
    hipMalloc(&cvals,sizeof(T)*nnz);
    hipMalloc(&ccols,sizeof(int)*nnz);
	//SparseDeviceMatrixCSR<T> C(B.Width(), A.Height(), DeviceVector<T>(nonZeros), DeviceVector<uint>(nonZeros), CrowStarts);

//record_time("malloc");
	DifSpmmWarp(\
    cvals,crows,ccols,\
    avals,arows,acols,\
    bvals,brows,bcols,\
    dvals,drows,dcols,\
    alpha,\
    m,k,n,\
    DQueue,DQueue_one, h_queue_one);
//record_time("cal");
    delete[]h_queue_one;
	return;
}

template<typename T>
void call_multiply_add(T*avals,int*arows,int*acols,int nnza,\
T*bvals,int*brows,int*bcols,int nnzb,\
T*dvals,int*drows,int*dcols,int nnzd,\
T alpha,\
int m,int k,int n,\
T*&cvals,int*&crows,int*&ccols,int&nnzc)
{
    int*Queue;
    int*Queue_one;
//record_time("start");
    myComputeBinAdd_Device(\
    avals,arows,acols,m, \
    Queue, Queue_one);

   /* 
   {
	//getchar();
	hipDeviceSynchronize();
	int*hq=new int[m];
	int*harows=new int[m+1];
	hipMemcpyDtoH(harows,arows,sizeof(int)*(m+1));
	int ptr=0;
	for(int ii=0;ii<14;ii++){
		for(int i=0;i<m&&ptr<nnza;i++){
			if(hostgettag(harows[i+1]-harows[i])==ii){
			hq[ptr]=i;
			ptr++;
			}	
		}
	}
	hipMemcpyHtoD(Queue,hq,sizeof(int)*m);
	//std::cout<<"end",getchar();
    }
	*/
//record_time("compute_bin");
/*
	int*hq=new int[n];
	cudaMemcpy(hq,Queue,sizeof(int)*n,cudaMemcpyDeviceToHost);
	for(int i=0;i<n;i++){
		if(hq[i]==22204){std::cout<<i;getchar();}
	}
	int*hq1=new int[14];
	
	cudaMemcpy(hq1,Queue_one,sizeof(int)*14,cudaMemcpyDeviceToHost);

	for(int i=0;i<14;i++)std::cout<<hq1[i],getchar();
*/
//	std::cout<<"after computebin, before dif\n";
    DifMulAddRMerge(\
    avals,arows,acols, \
    bvals,brows,bcols, \
    dvals,drows,dcols,\
    alpha,\
    cvals,crows,ccols,\
    m,k,n,nnzc,\
    Queue,m,\
    Queue_one,14);

	hipFree(Queue);
	hipFree(Queue_one);

}
