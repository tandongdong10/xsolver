#include "parilut.h"
namespace parilt{
void prefix_sum(int n,int*rows,int&nnz){
    ScanExclusive(rows,n);
    hipMemcpyDtoH(&nnz,rows+n,sizeof(int));
}

__global__ void temp_reverse(double*aval,int nnz){
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    for(;k<nnz;k+=blockDim.x*gridDim.x){
        aval[k]=-aval[k];
    }
}
__global__ void temp_tri_lu_nnz(double*val,int*row,int*col,int n,int nnz,\
double*lval,int*lrow,int*lcol,int nnzl,\
double*uval,int*urow,int*ucol,int nnzu){
    int k=blockDim.x*blockIdx.x+threadIdx.x;
    if(k>=n)return;
    for(int i=k;i<n;i+=blockDim.x*gridDim.x){
        int nnzli=0;
        int nnzui=0;
        int tag=1;
        for(int ii=row[k];ii<row[k+1];ii++){
            if(col[ii]<=i)nnzli+=1;
            if(col[ii]==i)tag=0;
            if(col[ii]>=i)nnzui+=1;
        }
        nnzli+=tag;
        nnzui+=tag;
        lrow[i]=nnzli;
        urow[i]=nnzui;
    }
}

__global__ void temp_tri_lu_nnz2(double*val,int*row,int*col,int n,int nnz,\
double*lval,int*lrow,int*lcol,int nnzl,\
double*uval,int*urow,int*ucol,int nnzu){
    int lid=threadIdx.x&(warpSize-1);
    int wid=threadIdx.x/warpSize;
    int rowi = hipBlockIdx_x * blockDim.x / warpSize + wid;
for(rowi;rowi<n;rowi+=blockDim.x*gridDim.x/warpSize){
    int rowstart=row[rowi];
    int rowend=row[rowi+1];
    int nnzli=0;
    int nnzui=0;
    int tag=1;
    for(int i=rowstart+lid;i<rowend;i+=warpSize){
        if(col[i]<rowi)nnzli+=1;
        if(col[i]>rowi)nnzui+=1;
        //if(col[i]==tag)tag=1;
    }


    warp_reduce_sum(&nnzli);
    warp_reduce_sum(&nnzui);

    nnzli+=tag;
    nnzui+=tag;

    lrow[rowi]=nnzli;
    urow[rowi]=nnzui;

}

}

__global__ void temp_tri_lu(double*val,int*row,int*col,int n,int nnz,\
double*lval,int*lrow,int*lcol,int&nnzl,\
double*uval,int*urow,int*ucol,int&nnzu){
    int k=blockDim.x*blockIdx.x+threadIdx.x;
    if(k>=n)return ;
    for(int i=k;i<n;i+=blockDim.x*gridDim.x){
        int da=row[i];
        int ii;
        for(ii=lrow[i];ii<lrow[i+1]-1;ii++){
            lcol[ii]=col[da];
            lval[ii]=val[da];
            da+=1;
        }
        lcol[ii]=i;
        lval[ii]=1;

        for(ii=urow[i];ii<urow[i+1];ii++){
            ucol[ii]=col[da];
            uval[ii]=val[da];
            da+=1;
        }
    }
}

__global__ void temp_tri_lu_v2(double*val,int*row,int*col,int n,int nnz,\
double*lval,int*lrow,int*lcol,int nnzl,\
double*uval,int*urow,int*ucol,int nnzu){

int lid = hipThreadIdx_x & (warpSize - 1);
int wid = hipThreadIdx_x / warpSize;

int rowi = hipBlockIdx_x * blockDim.x / warpSize + wid;
if(rowi>=n)return;
int rowstart=row[rowi];
int rowend=row[rowi+1]-rowstart;
int i=lid;
col+=rowstart;
val+=rowstart;
lval+=lrow[rowi];
lcol+=lrow[rowi];
uval+=urow[rowi];
ucol+=urow[rowi];
int vcol=i<rowend?col[i]:rowi;
for(;vcol<rowi;){
//vcol=col[i];
lval[i]=val[i];
lcol[i]=col[i];
i+=warpSize;
vcol=i<rowend?col[i]:rowi;
}
warp_reduce_min(&i);

if(lid==0)
{
    lval[i]=1;
    lcol[i]=rowi;
}

int ui=lid;
val+=i;
col+=i;
rowend-=i;
while(ui<rowend){
uval[ui]=val[ui];
ucol[ui]=col[ui];
ui+=warpSize;
}
}

__global__ void embedA_intoB_L(double*aval,int*acol,int*arow,int n,\
double*bval,int*bcol,int*brow,double*uval,int*ucol,int*urow){
    int k=blockDim.x*blockIdx.x+threadIdx.x;
    if(k>=n)return;
    for(int i=k;i<n;i+=blockDim.x*gridDim.x){
        int astart=arow[i];
        int bstart=brow[i];
        int aend=arow[i+1];
        int bend=brow[i+1];
	//double dive=aval[astart];
        for(int ii=bstart;ii<bend&&astart<aend;ii++){
            int bcolii=bcol[ii];
            if(acol[astart]==bcol[ii]){
                bval[ii]=aval[astart];
                astart+=1;
            }
            else{
                bval[ii]=bval[ii]/(uval[urow[bcolii]]);
            }
        }
    }
}

__global__ void embedA_intoB_U(double*aval,int*acol,int*arow,int n,\
double*bval,int*bcol,int*brow,double*uval,int*ucol,int*urow){
    int k=blockDim.x*blockIdx.x+threadIdx.x;
    if(k>=n)return;
    for(int i=k;i<n;i+=blockDim.x*gridDim.x){
        int astart=arow[i];
        int bstart=brow[i];
        int aend=arow[i+1];
        int bend=brow[i+1];
	double dive=aval[astart];
        for(int ii=bstart;ii<bend&&astart<aend;ii++){
            
            if(acol[astart]==bcol[ii]){
                bval[ii]=aval[astart];
                astart+=1;
            }
        }
    }
}

__global__ void embedA_intoB_L2(double*aval,int*acol,int*arow,int n,\
double*bval,int*bcol,int*brow,double*uval,int*ucol,int*urow){
    int lid = hipThreadIdx_x & (warpSize - 1);
    int wid = hipThreadIdx_x / warpSize;
    int rowi = hipBlockIdx_x * blockDim.x / warpSize + wid;
 if(rowi>=n)return;
    //double dive=uval[urow[rowi]];
    for(int i=brow[rowi]+lid;i<brow[rowi+1]&&bcol[i]<rowi;i+=warpSize){	
	bval[i]/=uval[urow[bcol[i]]];
    }
    //__threadfence();
    __syncthreads();

    for(int i=arow[rowi]+lid;i<arow[rowi+1];i+=warpSize){
	int l=brow[rowi];
	int r=brow[rowi+1];
	int fcol=acol[i];
	while(l<r){
		int mid=l+((r-l)>>1);
		if(fcol<=bcol[mid]){
		r=mid;
		}
		else l=mid+1;
	}
	//bcol[l]=acol[i];
	if(bcol[l]==fcol)bval[l]=aval[i];
    }
      
}


__global__ void embedA_intoB_U2(double*aval,int*acol,int*arow,int n,\
double*bval,int*bcol,int*brow){
    int lid = hipThreadIdx_x & (warpSize - 1);
    int wid = hipThreadIdx_x / warpSize;
    int rowi = hipBlockIdx_x * blockDim.x / warpSize + wid;
 if(rowi>=n)return;
    //double dive=uval[urow[rowi]];

    for(int i=arow[rowi]+lid;i<arow[rowi+1];i+=warpSize){
	int l=brow[rowi];
	int r=brow[rowi+1];
	int fcol=acol[i];
	while(l<r){
		int mid=l+((r-l)>>1);
		if(fcol<=bcol[mid]){
		r=mid;
		}
		else l=mid+1;
	}
	//bcol[l]=acol[i];
	if(bcol[l]==fcol)bval[l]=aval[i];
    }
      
}

__global__ void all_abs(double*val,int nnz){
    int k=blockIdx.x*blockDim.x+threadIdx.x;
	for(int i=k;i<nnz;i+=blockDim.x*gridDim.x){
        double fvj=val[i];
        fvj=fvj>0?fvj:-fvj;
        val[i]=fvj;
	}
}

__global__ void set_reference(int*d_rows,int n,int*d_row_reference){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    //int i=blockIdx.x;
    if(i<n){
    for(int j=d_rows[i];j<d_rows[i+1];j+=1){
        d_row_reference[j]=i;
    }
    }
}

__global__ void set_reference_v2(int*d_rows,int n,int nnz,int*d_row_reference){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
   if(i>=nnz)return;
   int l=0;
   int r=n;
   while(l<r){
	int mid=l+((r-l)>>1);
        if(i<=d_rows[mid]){
	    r=mid;	
	}
	else{
	    l=mid+1;
	}
   } 

   d_row_reference[i]=l;
    
}
__global__ void set_reference_v3(int*d_rows,int n,int*d_row_reference){
 int lid = hipThreadIdx_x & (warpSize - 1);
int wid = hipThreadIdx_x / warpSize;
int rowi = hipBlockIdx_x * blockDim.x / warpSize + wid;
  if(rowi>=n)return;
for(int i=d_rows[rowi]+lid;i<d_rows[rowi+1];i+=warpSize){
	d_row_reference[i]=rowi;
}
  
}

__global__ void thresold_remove_nnz(double*val,int*row,int*col,double thresold,int nnz,int n,\
int*out_row){
	int k=blockIdx.x*blockDim.x+threadIdx.x;
	for(int i=k;i<n;i+=blockDim.x*gridDim.x){
        int count=0;
        for(int j=row[i];j<row[i+1]-1;j++){
            double fvj=val[j];
            fvj=fvj>0?fvj:-fvj;
            if(fvj<thresold){
                col[j]=-1;
            }
            else{
                count+=1;
            }
        }
        count+=1;
        out_row[i]=count;
	}
}

__global__ void thresold_remove_nnz2(double*val,int*row,int*col,double thresold,int nnz,int n,\
int*out_row){
    int lid = hipThreadIdx_x & (warpSize - 1);
    int wid = hipThreadIdx_x / warpSize;
    int rowi = hipBlockIdx_x * blockDim.x / warpSize + wid;
    int totalwarp=blockDim.x*gridDim.x/warpSize;
    for(int i=rowi;i<n;i+=totalwarp){
        int count=0;
        for(int j=row[i]+lid;j<row[i+1]-1;j+=warpSize){
            double fvj=val[j];
            fvj=fvj>0?fvj:-fvj;
            col[j]=fvj<thresold?-1:col[j];
            count+=(fvj>=thresold);
        }
        warp_reduce_sum(&count);
        count+=1;
        out_row[i]=count;
    }
}

__global__ void thresold_remove(double*val,int*row,int*col,double thresold,int nnz,int n,\
double*out_val,int*out_row,int*out_col){
	int k=blockIdx.x*blockDim.x+threadIdx.x;
	for(int i=k;i<n;i+=blockDim.x*gridDim.x){
        int loc=out_row[i];
	int j;
        for(j=row[i];j<row[i+1]-1;j++){
            int pcol=col[j];
            if(pcol!=-1){
                out_val[loc]=val[j];
                out_col[loc]=pcol;
                loc+=1;
            }
        }
	out_val[loc]=val[j];
	out_col[loc]=col[j];
	}
}


__global__ void thresold_remove2(double*val,int*row,int*col,double thresold,int nnz,int n,\
double*out_val,int*out_row,int*out_col){
    int lid = hipThreadIdx_x & (warpSize - 1);
    int wid = hipThreadIdx_x / warpSize;
    int rowi = hipBlockIdx_x * blockDim.x / warpSize + wid;
    if(rowi>=n)return;
/*
    __shared__ char sharedperwarp[64*1024];
    int*mysharestart=(int*)(sharedperwarp+64*1024/(blockDim.x / warpSize)*wid);
    int*mysharesend=(int*)(sharedperwarp+64*1024/(blockDim.x / warpSize)*(wid+1));
    int chunksize=(mysharesend-mysharestart)>>1;
    int*prefixsum=mysharestart+chunksize;
*/
    int istart=row[rowi];
    int iend=row[rowi+1];
    int nthis_row=iend-istart;
    val+=istart;
    col+=istart;
    out_col+=out_row[rowi];
    out_val+=out_row[rowi];
    int base0=0;
//    for(int loadstart=0;loadstart<nthis_row;loadstart+=chunksize){
//        int endi=min(chunksize,nthis_row-loadstart);
	int endi=nthis_row;
	int loadstart=0;
        int*ccol=col+loadstart;
        int*cout_col=out_col+loadstart;
        double*cval=val+loadstart;
        double*cout_val=out_val+loadstart;
        int nextoffset=0;
        for(int i=lid;i<endi;i+=warpSize){
            bool tag=(ccol[i]!=-1);
            unsigned long long mask = __ballot(tag);
            int offset=__popcll(__lanemask_lt()&mask)+nextoffset;
            if(tag){
                cout_col[offset]=ccol[i];
                cout_val[offset]=cval[i];
            }
            nextoffset+= __popcll(mask);
        }
//    }
}



#include"par_ilut_hip/select_kth.h"

#define T  double

__global__ void L_sweep(T*lval,int*lrows,int*lcols,int n,int lnnz,int*lrow_refer,\
    T*uval,int*urows,int*ucols,\
    T*val,int*rows,int*cols){
int k = blockDim.x * blockIdx.x + threadIdx.x;
if(k>=lnnz)return;

int i=lrow_refer[k];
int j=lcols[k];
T tempv=0;
//maybe binsearch here
/*
for(int kk=rows[i];kk<rows[i+1];kk+=1){
    if(cols[kk]==j){
        tempv=val[kk];
        break;
    }
}
*/
{
int l=rows[i];
int r=rows[i+1];
while(l<r){
	int mid=l+((r-l)>>1);
	if(cols[mid]<j){
		l=mid+1;
	}else{
		r=mid;
	}
}
tempv=cols[l]==j?val[l]:0;
}
__syncthreads();
        int index_u=ucols[j];
        int index_l=lrows[i];
    
        int lj,ui;
        ui=urows[index_u];
        lj=lcols[index_l];

        int check_last=(i<=j)?i:j;
        int&check_id=(i<=j)?ui:lj;

        while(check_id<check_last){

            tempv=(lj==ui)?(tempv-lval[index_l]*uval[index_u]):tempv;
    
            //index_l=(lj<=ui)?index_l+1:index_l;
            index_l+=(lj<=ui);
            //index_u=(lj>=ui)?index_u+1:index_u;
            index_u+=(ui<=lj);
            
            lj=lcols[index_l];
            ui=urows[index_u];
        }
	if(i>j)lval[index_l]=tempv/uval[ucols[j+1]-1];
}


__global__ void U_sweep(T*lval,int*lrows,int*lcols,\
    T*uval,int*urows,int*ucols,int n,int unnz,int*ucol_refer,\
    T*val,int*rows,int*cols){
int k = blockDim.x * blockIdx.x + threadIdx.x;
if(k>=unnz)return;

int i=urows[k];
int j=ucol_refer[k];
T tempv=0;
//maybe binsearch here
/*
for(int kk=rows[i];kk<rows[i+1];kk+=1){
    if(cols[kk]==j){
        tempv=val[kk];
        break;
    }
}
*/
{
int l=rows[i];
int r=rows[i+1];
while(l<r){
	int mid=l+((r-l)>>1);
	if(cols[mid]<j){
		l=mid+1;
	}else{
		r=mid;
	}
}
tempv=cols[l]==j?val[l]:0;
}
__syncthreads();
        int index_u=ucols[j];
        int index_l=lrows[i];
    
        int lj,ui;
        ui=urows[index_u];
        lj=lcols[index_l];

        int check_last=(i<=j)?i:j;
        int&check_id=(i<=j)?ui:lj;

        while(check_id<check_last){

            tempv=(lj==ui)?(tempv-lval[index_l]*uval[index_u]):tempv;
    
            //index_l=(lj<=ui)?index_l+1:index_l;
            index_l+=(lj<=ui);
            //index_u=(lj>=ui)?index_u+1:index_u;
            index_u+=(ui<=lj);
            
            lj=lcols[index_l];
            ui=urows[index_u];
        }
        uval[index_u]=tempv;

}

__global__ void LU_sweep(T*lval,int*lrows,int*lcols,int lnnz,int*lrow_refer,\
    T*uval,int*urows,int*ucols,int unnz,int*ucol_refer,int n,\
    T*val,int*rows,int*cols){
int kk = blockDim.x * blockIdx.x + threadIdx.x;
if(kk>=unnz+lnnz)return;

int k=kk<lnnz?kk:kk-lnnz;
int i=kk<lnnz?lrow_refer[k]:urows[k];
int j=kk<lnnz?lcols[k]:ucol_refer[k];
T tempv=0;
//maybe binsearch here
/*
for(int kk=rows[i];kk<rows[i+1];kk+=1){
    if(cols[kk]==j){
        tempv=val[kk];
        break;
    }
}
*/
{
int l=rows[i];
int r=rows[i+1];
while(l<r){
	int mid=l+((r-l)>>1);
	if(cols[mid]<j){
		l=mid+1;
	}else{
		r=mid;
	}
}
tempv=cols[l]==j?val[l]:0;
}
__syncthreads();
        int index_u=ucols[j];
        int index_l=lrows[i];
    
        int lj,ui;
        ui=urows[index_u];
        lj=lcols[index_l];

        int check_last=(i<=j)?i:j;
        int&check_id=(i<=j)?ui:lj;

        while(check_id<check_last){

            tempv=(lj==ui)?(tempv-lval[index_l]*uval[index_u]):tempv;
    
            //index_l=(lj<=ui)?index_l+1:index_l;
            index_l+=(lj<=ui);
            //index_u=(lj>=ui)?index_u+1:index_u;
            index_u+=(ui<=lj);
            
            lj=lcols[index_l];
            ui=urows[index_u];
        }

if(i>j)lval[index_l]=tempv/uval[ucols[j+1]-1];
else   { 
	uval[index_u]=tempv;
	}
}

template<typename nt>
void check(nt*tocheck,nt*orgin,int nnz,char*tprint){
nt*tch=new nt[nnz];
nt*och=new nt[nnz];
hipMemcpyDtoH(tch,tocheck,sizeof(nt)*nnz);
hipMemcpyDtoH(och,orgin,sizeof(nt)*nnz);
for(int i=0;i<nnz;i++){
	if(tch[i]!=och[i]){
		printf("wrong on ");
		printf("%s",tprint);
		getchar();	
	}
}
delete[]tch;
delete[]och;
}
void cold_init()
{
	int tempint;
	double *tempd=NULL;
	int *tempi=NULL;	
	hipLaunchKernelGGL(temp_tri_lu_nnz2, dim3(1),dim3(warpSize),0,0,\
	tempd,tempi,tempi,0,0,\
	tempd,tempi,tempi,tempint,\
	tempd,tempi,tempi,tempint);
	int n=3;
	int nnzu=3;
	int*intarray;
	hipMalloc(&intarray,sizeof(int)*14);
	double*doublearray;
	hipMalloc(&doublearray,sizeof(double)*6);
	int*urow=intarray;
	int*ucol=urow+4;
	int*ucsc_row=ucol+3;
	int*ucsc_col=ucsc_row+4;
	double*uval=doublearray;
	double*ucsc_val=doublearray+3;
	int*ia=intarray;
	double*da=doublearray;
	ia[0]=0;ia[1]=1;ia[2]=2;ia[3]=3;ia[4]=0;ia[5]=1;ia[6]=2;
	da[0]=1;da[1]=1;da[2]=1;
        hipsparseDcsr2csc(handle1,n,n,nnzu,uval,urow,ucol,ucsc_val,ucsc_row,ucsc_col,\
        HIPSPARSE_ACTION_NUMERIC,HIPSPARSE_INDEX_BASE_ZERO);
	int tag;
	size_t sia=0;
	int*temp=intarray;
    	ScanExclusive(temp,0);
	hipFree(doublearray);
	hipFree(intarray);
}

void parilut_clean(T*dval,int*dcol,int*drow,int n,int nnz,\
            T*&lval,int*&lcol,int*&lrow,int&nnzl,\
            T*&uval,int*&ucol,int*&urow,int&nnzu,\
	    double max_nnz_keep_rate,int sweep)
{
//return 0;
int blockDim=1024;
int gridDim=n/(blockDim)+1;
//int nnzl=0;
//int nnzu=0;
int doublesize=sizeof(double);
int intsize=sizeof(int);
hipMalloc(&lrow,intsize*(n+1));
hipMalloc(&urow,intsize*(n+1));
//getchar();
//
//record_time("tril_start\0");
hipLaunchKernelGGL(temp_tri_lu_nnz2, dim3(n/(blockDim/warpSize)+1),dim3(blockDim),0,0,\
    dval,drow,dcol,n,nnz,\
    lval,lrow,lcol,nnzl,\
    uval,urow,ucol,nnzu);
hipDeviceSynchronize();
prefix_sum(n,lrow,nnzl);
prefix_sum(n,urow,nnzu);

hipMalloc(&lval,doublesize*nnzl);
hipMalloc(&uval,doublesize*nnzu);
hipMalloc(&lcol,intsize*nnzl);
hipMalloc(&ucol,intsize*nnzu);

//std::cout<<nnzl<<" "<<nnzu<<"\n";
//getchar();
hipLaunchKernelGGL(temp_tri_lu_v2, dim3(n/(blockDim/warpSize)+1),dim3(blockDim),0,0,\
    dval,drow,dcol,n,nnz,\
    lval,lrow,lcol,nnzl,\
    uval,urow,ucol,nnzu);
hipDeviceSynchronize();
//record_time("after trilv2\0");
//getchar();
hipsparseHandle_t&handle=handle1;
hipsparseMatDescr_t descrA;
//hipsparseCreate(&handle);
hipsparseCreateMatDescr(&descrA);

hipsparseMatDescr_t descrB;
hipsparseMatDescr_t descrC;
hipsparseCreateMatDescr(&descrB);
hipsparseCreateMatDescr(&descrC);
int max_nnzC=0;

double*luval;
int*lurow,*lucol;
luval=0;
lucol=0;
//sweep=5;
hipMalloc(&lurow,intsize*(n+1));
//nnz_keep_rate=3; //set matrix nnz

//printf("sweep!!!=%d nnz_keep_rate=%f\n",sweep,nnz_keep_rate);

int l0nnz=nnzl;
int u0nnz=nnzu;

int*dnnzC;
hipMalloc(&dnnzC,sizeof(int));
///////////////////////////////////////////////////////////////////////////////////////////////
record_time("before sweep_time\0");
for(int sweepi=0;sweepi<sweep;sweepi++){
//	record_time("sweep_start\0");
    //if(sweepi==1)break;
        //ExpendA=IterLcsr*IterUcsc;//hipsparsegemm
    double nnz_keep_rate=(max_nnz_keep_rate-1)/sweep*(sweepi+1)+1;
    int nnzC;

//    printf("nnzl=%d nnzu=%d n=%d\n",nnzl,nnzu,n);
/*
    hipsparseXcsrgemmNnz(handle,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,\
        n,n,n,\
        descrA,nnzl,lrow,lcol,\
        descrA,nnzu,urow,ucol,\
        descrA,lurow,dnnzC);


    hipDeviceSynchronize();
    hipMemcpyDtoH(&nnzC,dnnzC,sizeof(int));


    hipMalloc(&luval,nnzC*doublesize);
    hipMalloc(&lucol,nnzC*intsize);


    hipsparseDcsrgemm(handle,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,\
        n,n,n,\
        descrA,nnzl,lval,lrow,lcol,\
        descrA,nnzu,uval,urow,ucol,\
        descrA,luval,lurow,lucol);
    hipDeviceSynchronize();
    
record_time("gemm\0");

    int*add_row;
    int nnzD;
    hipMalloc(&add_row,intsize*(n+1));

    hipsparseXcsrgeamNnz(handle,n,n,\
        descrA,nnz,drow,dcol,\
        descrA,nnzC,lurow,lucol,\
        descrA,add_row,dnnzC);

    hipMemcpyDtoH(&nnzD,dnnzC,sizeof(int));


    double alpha=1;
    double beta=-1;

    double*add_val;
    int*add_col;

    hipMalloc(&add_val,doublesize*nnzD);
    hipMalloc(&add_col,intsize*nnzD);

    hipsparseDcsrgeam(handle,n,n,&alpha,\
        descrA,nnz,dval,drow,dcol,&beta,\
        descrA,nnzC,luval,lurow,lucol,\
        descrA,add_val,add_row,add_col);

    hipDeviceSynchronize();

record_time("geam\0");
    hipFree(luval);
    hipFree(lucol);

*/

double alpha=-1;
double*add_val;
int*add_row;
int*add_col;
int nnzD;
call_multiply_add(lval,lrow,lcol,nnzl,\
uval,urow,ucol,nnzu,\
dval,drow,dcol,nnz,\
alpha,\
n,n,n,\
add_val,add_row,add_col,nnzD);
hipDeviceSynchronize();
hipLaunchKernelGGL(temp_reverse, dim3(nnzD/blockDim+1),dim3(blockDim),0,0,add_val,nnzD);
hipDeviceSynchronize();

record_time("spgemm+spgeam\0");
    luval=NULL;
    lucol=NULL;

    //hipMemset(add_val,0,sizeof(double)*nnzD);

//record_time("norm==R-L*U\0");
    gridDim=n/(blockDim)+1;
    hipLaunchKernelGGL(embedA_intoB_L2, dim3(n/(blockDim/warpSize)+1),dim3(blockDim),0,0,\
        lval,lcol,lrow,n,\
        add_val,add_col,add_row,uval,ucol,urow);
    hipDeviceSynchronize();
    //FullU=ResU U IterUcsc;
{/*
double*tadval;
int*tadcol;
int*tadrow;
hipMalloc(&tadval,doublesize*nnzD);
hipMalloc(&tadcol,intsize*nnzD);
hipMalloc(&tadrow,intsize*(n+1));
hipMemcpyDtoD(tadval,add_val,doublesize*nnzD);
hipMemcpyDtoD(tadcol,add_col,intsize*nnzD);
hipMemcpyDtoD(tadrow,add_row,intsize*(n+1));
	embedA_intoB_L2<<<dim3(n/(blockDim/warpSize)+1),dim3(blockDim),0,0>>>(lval,lcol,lrow,n,\
	tadval,tadcol,tadrow,uval,ucol,urow);
	hipDeviceSynchronize();
	for(int i=0;i<nnzD;i++)
	{
	if(tadcol[i]!=add_col[i]){std::cout<<"wrong_col";getchar();}
	}


	for(int i=0;i<10;i++){
		for(int j=tadrow[i];j<tadrow[i+1];j++){
		std::cout<<"{"<<tadval[j]<<","<<tadcol[j]<<"} ";
		//std::cout<<"{"<<add_val[j]<<","<<add_col[j]<<"}";
		}	
		std::cout<<"\n";
		for(int j=add_row[i];j<add_row[i+1];j++){
		//std::cout<<"{"<<tadval[j]<<","<<tadcol[j]<<"} ";
		std::cout<<"{"<<add_val[j]<<","<<add_col[j]<<"} ";
		}
		std::cout<<"\n";
		for(int j=lrow[i];j<lrow[i+1];j++){
		std::cout<<"{"<<lval[j]<<","<<lcol[j]<<"} ";
		}
		std::cout<<"\n";
		std::cout<<"\n";
	}

	for(int i=0;i<nnzD;i++){
	if(tadval[i]!=add_val[i]){
	std::cout<<"wong_val ";
	std::cout<<i<<" "<<tadval[i]<<" "<<add_val[i];
	getchar();
	}
	}
*/}
    hipLaunchKernelGGL(embedA_intoB_U2, dim3(n/(blockDim/warpSize)+1),dim3(blockDim),0,0,\
        uval,ucol,urow,n,\
        add_val,add_col,add_row);

    hipDeviceSynchronize();
 //record_time("embed\0");

    hipFree(lval);
    hipFree(uval);
    hipFree(lcol);
    hipFree(ucol);

    gridDim=n/blockDim+1;
    hipLaunchKernelGGL(temp_tri_lu_nnz2, dim3(n/(blockDim/warpSize)+1),dim3(blockDim),0,0,\
        add_val,add_row,add_col,n,nnzD,\
        lval,lrow,lcol,nnzl,\
        uval,urow,ucol,nnzu);
hipDeviceSynchronize();

//record_time("trilu_nnz\0");
    prefix_sum(n,lrow,nnzl);
    prefix_sum(n,urow,nnzu);

//    printf("nnzl=%d nnzu=%d\n",nnzl,nnzu);
    hipMalloc(&lval,doublesize*nnzl);
    hipMalloc(&uval,doublesize*nnzu);
    hipMalloc(&lcol,intsize*nnzl);
    hipMalloc(&ucol,intsize*nnzu);

//record_time("prefix_sum\0");
    hipLaunchKernelGGL(temp_tri_lu_v2,dim3(n/(blockDim/warpSize)+1),dim3(blockDim),0,0,\
        add_val,add_row,add_col,n,nnzD,\
        lval,lrow,lcol,nnzl,\
        uval,urow,ucol,nnzu);
  hipDeviceSynchronize();


//record_time("temp_tri_lu\0");  
    //transpose uval to ucsc
    double*ucsc_val;
    int*ucsc_row,*ucsc_col;
    hipMalloc(&ucsc_col,intsize*(n+1));
    hipMalloc(&ucsc_val,doublesize*nnzu);
    hipMalloc(&ucsc_row,intsize*nnzu);

    int*alrowrefer;
    int*aucolrefer;
    hipMalloc(&alrowrefer,intsize*nnzl);
    hipMalloc(&aucolrefer,intsize*nnzu);
    //int*d_row_reference;
    //hipMalloc(&d_row_reference,sizeof(int)*nnzD);

    int blocksize=1024;
    int gridsize=nnzD/blocksize+1;
    hipDeviceSynchronize();

//record_time("tri_lu\0");
    //set_row_referenced
    //hipLaunchKernelGGL(set_reference,dim3(gridsize),dim3(blocksize),0,0,\
        add_row,n,d_row_reference);
    gridsize=n/(blocksize/warpSize)+1;

    hipLaunchKernelGGL(set_reference_v3,dim3(gridsize),dim3(blocksize),0,0,\
    lrow,n,alrowrefer);
    //getchar();
    //careful for ucsc_mark
record_time("before transpose\0");
        hipsparseDcsr2csc(handle,n,n,nnzu,uval,urow,ucol,ucsc_val,ucsc_row,ucsc_col,\
        HIPSPARSE_ACTION_NUMERIC,HIPSPARSE_INDEX_BASE_ZERO);
        //sweep(FullL,FullU)
    hipDeviceSynchronize();

record_time("dcsr2csc\0");
    hipLaunchKernelGGL(set_reference_v3,dim3(gridsize),dim3(blocksize),0,0,\
    ucsc_col,n,aucolrefer);
    hipDeviceSynchronize();


    hipFree(uval);
    hipFree(ucol);
//record_time("set reference\0");
    //hipLaunchKernelGGL(ilu0_aij_Ldiag1_Udiag_try_faster2, dim3(gridsize),dim3(blocksize),0,0,\
        lval,lrow,lcol,ucsc_val,ucsc_row,ucsc_col,n,\
        add_row,add_col,d_row_reference,add_val,nnzD);

    //__global__ void L_sweep(T*lval,int*lrows,int*lcols,int n,int lnnz,int*lrow_refer,\
        T*uval,int*urows,int*ucols,\
        T*val,int*rows,int*cols)
    //temp_lval,temp_outrowl,temp_lcol
    //temp_uval,temp_outrowu,temp_ucol
/*
    gridsize=nnzl/blocksize+1;
    hipLaunchKernelGGL(L_sweep,dim3(gridsize),dim3(blocksize),0,0,\
        lval,lrow,lcol,n,nnzl,alrowrefer,\
        ucsc_val,ucsc_row,ucsc_col,\
        dval,drow,dcol);

    //__global__ void U_sweep(T*lval,int*lrows,int*lcols,\
        T*uval,int*urows,int*ucols,int n,int unnz,int*ucol_refer,\
        T*val,int*rows,int*cols)

    gridsize=nnzu/blocksize+1;
    hipLaunchKernelGGL(U_sweep,dim3(gridsize),dim3(blocksize),0,0,\
        lval,lrow,lcol,\
        ucsc_val,ucsc_row,ucsc_col,n,nnzu,aucolrefer,\
        dval,drow,dcol);

    hipDeviceSynchronize();
*/
//record_time("large_sweep\0");


record_time("sweep prepare\0");

    gridsize=(nnzl+nnzu)/blocksize+1;
    hipLaunchKernelGGL(LU_sweep,dim3(gridsize),dim3(blocksize),0,0,\
	lval,lrow,lcol,nnzl,alrowrefer,\
	ucsc_val,ucsc_row,ucsc_col,nnzu,aucolrefer,n,\
	dval,drow,dcol);
	hipDeviceSynchronize();

record_time("new large_sweep\0");
     //getchar();
    //how to do ?????
    //        (Nl,Nu)=select_remov(FullL,FullU);
    //hipFree(d_row_reference);
    hipFree(add_row);
    hipFree(add_col);
    hipFree(add_val);
    hipFree(alrowrefer);
    hipFree(aucolrefer);
   
    
    int keep_nnz=std::round(nnz_keep_rate*nnz);
//	printf("keep_nnz=%d\n",keep_nnz);
    int remove_nnz=nnzD-keep_nnz;
    int*temp_outrowl;
    int*temp_outrowu;
    double*temp_lval;
    double*temp_uval;

    int*temp_lcol;
    int*temp_ucol;

//    remove_nnz=-1;
//record_time("before remove\0");
    if(remove_nnz>0){
        double*sort_temp=NULL;
        unsigned long sort_temp_byte=0;

        double*lnnz_val;
	double*unnz_val;

        if(nnzD+n!=nnzl+nnzu)printf("nnzD+n!=nnzu+nnzl");

        hipMalloc(&lnnz_val,sizeof(double)*(nnzl));
	hipMalloc(&unnz_val,sizeof(double)*(nnzu));

	
	hipMemcpyDtoD(unnz_val,ucsc_val,doublesize*nnzu);
        hipMemcpyDtoD(lnnz_val,lval,sizeof(double)*nnzl);
        int tbls=1024;
        int tgrs=n/(tbls)+1;

        //moveL_diag_to_end(int*row,double*src,double*dest,int nnz,int n)
        //hipLaunchKernelGGL(moveL_diag_to_end,dim3(tgrs),dim3(tbls),0,0,\
        lrow,ucsc_val,nnz_val,0,n);

        //hipDeviceSynchronize();
        //hipMemcpyDtoD(nnz_val+nnzl,ucsc_val+n,sizeof(double)*(nnzu-n));

        //abs
        hipLaunchKernelGGL(all_abs,dim3(nnzl/blocksize+1),dim3(blocksize),0,0,\
        lnnz_val,nnzl);
        hipLaunchKernelGGL(all_abs,dim3(nnzu/blocksize+1),dim3(blocksize),0,0,\
	unnz_val,nnzu);
	hipDeviceSynchronize();
//record_time("abs\0");
double test_lthresold,test_uthresold;
        double lthresold=100;
	int lkeep_nnz=nnz_keep_rate*l0nnz;
	int lremove_nnz=nnzl-lkeep_nnz;
	lremove_nnz=lremove_nnz>0?lremove_nnz:0;
        double uthresold=100;
	int ukeep_nnz=nnz_keep_rate*u0nnz;
	int uremove_nnz=nnzu-ukeep_nnz;
	uremove_nnz=uremove_nnz>0?uremove_nnz:0;

//record_time("my_get kth start|||||||||||||||||||||||");
	find_kth_element(lnnz_val,nnzl,lremove_nnz,lthresold,NULL,NULL);
//record_time("l_kth_end____________and u start________");
	find_kth_element(unnz_val,nnzu,uremove_nnz,uthresold,NULL,NULL);
//record_time("u_kth_end______________________________");
/*
        hipcub::DeviceRadixSort::SortKeys(sort_temp,sort_temp_byte,lnnz_val,lnnz_val,nnzl);
        //printf("sort temp_byte=%lu nnzl=%d\n",sort_temp_byte,nnzl);
        hipMalloc(&sort_temp,sort_temp_byte);
        hipcub::DeviceRadixSort::SortKeys(sort_temp,sort_temp_byte,lnnz_val,lnnz_val,nnzl);
	hipDeviceSynchronize();
        hipFree(sort_temp);

        //hipFree(nnz_val);
//record_time("radix_sortltime cost_____________________________\0");
	sort_temp=NULL;
	sort_temp_byte=0;
*/
        //printf("lremove_nnz=%d nnzl=%d keep_nnz=%d\n",lremove_nnz,nnzl,lkeep_nnz);
//        hipMemcpyDtoH(&lthresold,lnnz_val+lremove_nnz,sizeof(double));
//        printf("lthresold=%e remove_nnz=%d cmpareto=%e",lthresold,lremove_nnz,test_lthresold);
	//std::cout<<"mylthresold="<<test_lthresold<<"\n";
	hipFree(lnnz_val);
//record_time("printl and get lthresold\0");

/*	hipcub::DeviceRadixSort::SortKeys(sort_temp,sort_temp_byte,unnz_val,unnz_val,nnzu);
        //printf("sort temp_byte=%lu nnzu=%d\n",sort_temp_byte,nnzu);
        hipMalloc(&sort_temp,sort_temp_byte);
        hipcub::DeviceRadixSort::SortKeys(sort_temp,sort_temp_byte,unnz_val,unnz_val,nnzu);
	hipDeviceSynchronize();
	hipFree(sort_temp);
*/
//record_time("radix_sort_utime\0");
        //printf("uremove_nnz=%d nnzu=%d keep_nnz=%d\n",uremove_nnz,nnzu,ukeep_nnz);
  //      hipMemcpyDtoH(&uthresold,unnz_val+uremove_nnz,sizeof(double));
    //    printf("uthresold=%e remove_nnz=%d cmpareto=%e",uthresold,uremove_nnz,test_uthresold);
	//std::cout<<"myuthresold="<<test_uthresold<<"\n";
	hipFree(unnz_val);

        //void thresold_remove_nnz(double*val,int*row,int*col,double thresold,int nnz,int n,\
        int*out_row)
	//getchar();
//record_time("print and getU\0");
        hipMalloc(&temp_outrowl,sizeof(int)*(n+1));
        hipMalloc(&temp_outrowu,sizeof(int)*(n+1));

        hipLaunchKernelGGL(thresold_remove_nnz2,dim3(n/(tbls/warpSize)+1),dim3(tbls),0,0,\
        ucsc_val,ucsc_col,ucsc_row,uthresold,nnzu,n,temp_outrowu);

       // hipDeviceSynchronize();
        //printf("Uremove\n");
        //getchar();
        //printf("Lremove\n");
        hipLaunchKernelGGL(thresold_remove_nnz2,dim3(n/(tbls/warpSize)+1),dim3(tbls),0,0,\
        lval,lrow,lcol,lthresold,nnzl,n,temp_outrowl);

        hipDeviceSynchronize();
//record_time("remove_nnz\0");
        //getchar();
        //prefix_sum(int n,int*rows,int&nnz)
        prefix_sum(n,temp_outrowl,nnzl);
        prefix_sum(n,temp_outrowu,nnzu);
//record_time("prefix_sum\0");
        //getchar();
        hipMalloc(&temp_lval,doublesize*nnzl);
        hipMalloc(&temp_uval,doublesize*nnzu);

        hipMalloc(&temp_lcol,intsize*nnzl);
        hipMalloc(&temp_ucol,intsize*nnzu);

        //void thresold_remove(double*val,int*row,int*col,double thresold,int nnz,int n,\
        double*out_val,int*out_row,int*out_col)
        hipLaunchKernelGGL(thresold_remove2,dim3(n/(tbls/warpSize)+1),dim3(tbls),0,0,\
        lval,lrow,lcol,lthresold,nnzl,n,temp_lval,temp_outrowl,temp_lcol);

        hipLaunchKernelGGL(thresold_remove2,dim3(n/(tbls/warpSize)+1),dim3(tbls),0,0,\
        ucsc_val,ucsc_col,ucsc_row,uthresold,nnzu,n,temp_uval,temp_outrowu,temp_ucol);
	 hipDeviceSynchronize();
//record_time("remove\0");
    hipFree(lval);
    hipFree(lrow);
    hipFree(lcol);

    hipFree(ucsc_val);
    hipFree(ucsc_row);
    hipFree(ucsc_col);

    }
    else{
  //      printf("skip_remove\n");
        hipMalloc(&temp_outrowl,sizeof(int)*(n+1));
        hipMalloc(&temp_outrowu,sizeof(int)*(n+1));
        hipMalloc(&temp_lval,doublesize*nnzl);
        hipMalloc(&temp_uval,doublesize*nnzu);
        hipMalloc(&temp_lcol,intsize*nnzl);
        hipMalloc(&temp_ucol,intsize*nnzu);

        hipMemcpyDtoD(temp_outrowl,lrow,sizeof(int)*(n+1));
        hipMemcpyDtoD(temp_outrowu,ucsc_col,sizeof(int)*(n+1));
        hipMemcpyDtoD(temp_lval,lval,doublesize*nnzl);
        hipMemcpyDtoD(temp_uval,ucsc_val,doublesize*nnzu);
        hipMemcpyDtoD(temp_lcol,lcol,intsize*nnzl);
        hipMemcpyDtoD(temp_ucol,ucsc_row,intsize*nnzu);
    hipFree(lval);
    hipFree(lrow);
    hipFree(lcol);

    hipFree(ucsc_val);
    hipFree(ucsc_row);
    hipFree(ucsc_col);
    }
    int*lrow_refer;
    int*ucol_refer;

    hipMalloc(&lrow_refer,sizeof(int)*(nnzl));
    hipMalloc(&ucol_refer,sizeof(int)*(nnzu));
record_time("after_remove\0");


    gridsize=n/(blocksize/warpSize)+1;
    hipLaunchKernelGGL(set_reference_v3,dim3(gridsize),dim3(blocksize),0,0,\
        temp_outrowl,n,lrow_refer);

    //gridsize=n/(blocksize)+1;
    hipLaunchKernelGGL(set_reference_v3,dim3(gridsize),dim3(blocksize),0,0,\
        temp_outrowu,n,ucol_refer);

    hipDeviceSynchronize();
//record_time("set referecne\0");
    //printf("\nprepare small sweep\n");


    //__global__ void L_sweep(T*lval,int*lrows,int*lcols,int n,int lnnz,int*lrow_refer,\
        T*uval,int*urows,int*ucols,\
        T*val,int*rows,int*cols)
    //temp_lval,temp_outrowl,temp_lcol
    //temp_uval,temp_outrowu,temp_ucol
    blocksize=1024;
    gridsize=nnzl/blocksize+1;
    hipLaunchKernelGGL(L_sweep,dim3(gridsize),dim3(blocksize),0,0,\
        temp_lval,temp_outrowl,temp_lcol,n,nnzl,lrow_refer,\
        temp_uval,temp_ucol,temp_outrowu,\
        dval,drow,dcol);

    //__global__ void U_sweep(T*lval,int*lrows,int*lcols,\
        T*uval,int*urows,int*ucols,int n,int unnz,int*ucol_refer,\
        T*val,int*rows,int*cols)
    gridsize=nnzu/blocksize+1;
    hipLaunchKernelGGL(U_sweep,dim3(gridsize),dim3(blocksize),0,0,\
        temp_lval,temp_outrowl,temp_lcol,\
        temp_uval,temp_ucol,temp_outrowu,n,nnzu,ucol_refer,\
        dval,drow,dcol);

    hipDeviceSynchronize();
record_time("small sweep\0");
    //printf("\naftersweep\n");

    hipFree(lrow_refer);
    hipFree(ucol_refer);

    hipMalloc(&uval,doublesize*nnzu);
    hipMalloc(&ucol,intsize*nnzu);
    //        sweep(Nl,Nu);

    //        IterLcsr=Nl,IterUcsc=Nu;

        hipsparseDcsr2csc(handle,n,n,nnzu,temp_uval,temp_outrowu,temp_ucol,uval,ucol,urow,\
        HIPSPARSE_ACTION_NUMERIC,HIPSPARSE_INDEX_BASE_ZERO);
    hipDeviceSynchronize();

//record_time("ucsc transpose_back\0");
    //printf("ucsc_transpose_back\n");
    //getchar();


    hipFree(temp_uval);
    hipFree(temp_ucol);
    hipFree(temp_outrowu);
    lval=temp_lval;
    lcol=temp_lcol;
    lrow=temp_outrowl;
    //printf("nexiteri this_nnzl=%d this_nnzu=%d sum=%d\n\n",nnzl,nnzu,nnzl+nnzu);
record_time("iter end\0");
    //getchar();
    }

hipFree(lurow);

hipFree(dnnzC);

}

void call_parilut(T*hval,int*hcol,int*hrow,int n,int nnz,\
            T*&lval,int*&lcol,int*&lrow,int&nnzl,\
            T*&uval,int*&ucol,int*&urow,int&nnzu,\
            int sweep){

hipsparseStatus_t status = hipsparseCreate(&handle1);
for(int i=0;i<13;i++){
    hipStreamCreate(&stream[i]);
}
double*dlval,*duval;
int*dlcol,*dlrow;
int*ducol,*durow;

double*dval;
int*dcol;
int*drow;

hipMalloc(&dval,sizeof(double)*nnz);

hipMalloc(&dcol,sizeof(int)*nnz);

hipMalloc(&drow,sizeof(int)*(n+1));

hipMemcpyHtoD(dval,hval,sizeof(double)*nnz);
hipMemcpyHtoD(dcol,hcol,sizeof(int)*nnz);
hipMemcpyHtoD(drow,hrow,sizeof(int)*(n+1));
record_time("before_cold init");
cold_init();
//std::cout<<"startheare\n";
record_time("cold_init\0");
parilut_clean(dval,dcol,drow,n,nnz,\
        dlval,dlcol,dlrow,nnzl,\
        duval,ducol,durow,nnzu,\
        3,sweep);
record_time("parilut_run_time\0");
std::cout<<"run_clean___________________________-"<<"\n";
lval=new double[nnzl];
lcol=new int[nnzl];
lrow=new int[n+1];

uval=new double[nnzu];
ucol=new int[nnzu];
urow=new int[n+1];
hipMemcpyDtoH(lval,dlval,sizeof(double)*nnzl);

hipMemcpyDtoH(uval,duval,sizeof(double)*nnzu);


hipMemcpyDtoH(lcol,dlcol,sizeof(int)*nnzl);
hipMemcpyDtoH(ucol,ducol,sizeof(int)*nnzu);


hipMemcpyDtoH(lrow,dlrow,sizeof(int)*(n+1));

hipMemcpyDtoH(urow,durow,sizeof(int)*(n+1));

for(int i=0; i<13; i++)
{
    hipStreamDestroy(stream[i]);
}

}

#undef T

void nytrsv_Lcsr(double*b,double*x,double*lval,int*lcol,int*lrow,int n){
for(int i=0;i<n;i++){
double sum=b[i];
for(int j=lrow[i];j<lrow[i+1]-1;j++){
int xcol=lcol[j];
sum-=x[xcol]*lval[j];
}
if(lval[lrow[i+1]-1]!=1){
printf("Ldiag not 1\n");
getchar();
}
x[i]=sum;
}

}

void nytrsv_Ucsr(double*b,double*x,double*uval,int*ucol,int*urow,int n){
for(int i=n-1;i>=0;i--){
	double sum=b[i];
	for(int j=urow[i]+1;j<urow[i+1];j++){
		int xcol=ucol[j];
		sum-=x[xcol]*uval[j];
	}

	x[i]=sum/uval[urow[i]];
}

}
}
