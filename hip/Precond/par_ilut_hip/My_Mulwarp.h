/*
template<int WarpSize, typename T>
static __device__ T WarpMin(T value){
	// Use XOR mode to perform butterfly reduction	
	for(int i=WarpSize/2; i>=1; i/=2){
		T tmp=__shfl_xor(value, i);
		value=value<tmp?value:tmp;
	}
	return value;
}
*/
template<int WarpSize, typename T>
static __device__ __host__ T WarpSum(T value){	
	for (int i=WarpSize/2; i>=1; i/=2)
		value+=__shfl_xor(value, i, WarpSize);
	return value;	
}

template<int WarpSize, typename T>
static __device__ void DifMul_1(\
T*cvals,int*crows,int*ccols, int crnnz,\
T*avals,int*arows,int*acols, int arnnz,\
T*bvals,int*brows,int*bcols,\
T*dvals,int*drows,int*dcols,int drnnz,\
T alpha){
	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}


	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues; int* rowIndices;int rowLength=0;//The row for the thread
	T weight=0;//The weight for the row
	if(threadIdx.x<arnnz){
		 int r= (acols[threadIdx.x]);// int rowIndex=a.Index(thread);		
		 int rowStart= (brows[r]);
		rowLength= (brows[r+1])-rowStart;
		rowValues=bvals+rowStart;
		rowIndices=bcols+rowStart;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight= (avals[threadIdx.x]);//a.Value(thread);
	}
	if(blockDim.x-1==threadIdx.x){
		rowLength=drnnz;
		rowValues=dvals;
		rowIndices=dcols;
		weight=alpha;
	}
	//__syncthreads();
	int rowPos=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread
	if(rowPos<rowLength){//Load the front index and row
		frontIndex= *(rowIndices+rowPos);// : explicit cache usage
		frontValue= *(rowValues+rowPos)*weight;// : explicit cache usage
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	 int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos<rowLength){
				frontValue= *(rowValues+rowPos)*weight;// : explicit cache usage
				frontIndex=*(rowIndices+rowPos);// : explicit cache usage
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=( int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}


template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_1(\
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{
	int volatile tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;

    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;

	DifMul_1<WarpSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha);
}

template<int WarpSize, typename T>
static __device__ void DifMul_2(\
T*cvals,int*crows,int*ccols, int crnnz,\
T*avals,int*arows,int*acols, int arnnz,\
T*bvals,int*brows,int*bcols,\
T*dvals,int*drows,int*dcols,int drnnz,\
T alpha){
	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0; int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1; int* rowIndices1;int rowLength1=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	int t=(threadIdx.x+1)*2;

	if(t<=arnnz){
		 int d0=threadIdx.x*2;
		 int r0=*(acols+d0);// int rowIndex=a.Index(thread);		
		 int r1=*(acols+d0+1);
		 int rowStart0=*(brows+r0);
		 int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
	}
	else if(t-1==arnnz){

		 int d0=threadIdx.x*2;
		 int r0=*(acols+d0);
		 int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowLength1=0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
	}
	if(blockDim.x-1==threadIdx.x){
		rowLength1=drnnz;
		rowValues1=dvals;
		rowIndices1=dcols;
		weight1=alpha;
	}
	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}

	if(index0<index1)
	{
		frontIndex=index0;
		frontValue=*(rowValues0+rowPos0)*weight0;
		rowPos0++;
	}
	else if(index0>index1)
	{
		frontIndex=index1;
		frontValue=*(rowValues1+rowPos1)*weight1;
		rowPos1++;
	}
	else
	{
		if(index0!=intMax)
		{
			frontIndex=index0;
			frontValue=*(rowValues0+rowPos0)*weight0+*(rowValues1+rowPos1)*weight1;
			rowPos0++;
			rowPos1++;
		}
		else
		{
		}
	}


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	 int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(index0<index1)
			{
				frontIndex=index0;
				frontValue=*(rowValues0+rowPos0)*weight0;
				rowPos0++;
			}
			else if(index0>index1)
			{
				frontIndex=index1;
				frontValue=*(rowValues1+rowPos1)*weight1;
				rowPos1++;
			}
			else 
			{
				if(index0!=intMax)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0+*(rowValues1+rowPos1)*weight1;
					rowPos0++;
					rowPos1++;
				}
				else
				{
					frontIndex=intMax;
				}
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=( int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_2(\
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;

    int arow0=arows[r];
    //int arnnz=arows[r+1]-arow0;
	//    int crow0=crows[r];
    //int crnnz=crows[r+1]-crow0;
	//    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;
	DifMul_2<WarpSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha);
}

template<int WarpSize, typename T>
static __device__ void DifMul_4(\
T*cvals,int*crows,int*ccols, int crnnz,\
T*avals,int*arows,int*acols, int arnnz,\
T*bvals,int*brows,int*bcols,\
T*dvals,int*drows,int*dcols,int drnnz,\
T alpha){
	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	int t=(threadIdx.x+1)*4;

	if(t<=arnnz){
		int d0=threadIdx.x*4;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
	}
	else if(t-1==arnnz)  //arnnz%4==3
	{
		int d0=threadIdx.x*4;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
	}
	else if(t-2==arnnz) //arnnz%4==2
	{
		int d0=threadIdx.x*4;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
	}
	else if(t-3==arnnz) //arnnz%4==1
	{
		int d0=threadIdx.x*4;
		int r0=*(acols+d0);
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);

	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}

	if(blockDim.x-1==threadIdx.x){
		rowLength3=drnnz;
		rowValues3=dvals;
		rowIndices3=dcols;
		weight3=alpha;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=*(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=*(rowIndices3+rowPos3);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontValue=*(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=*(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=*(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=*(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
	}
	else
	{
		frontIndex=intMax;
	}


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=*(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=*(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=*(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=*(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=*(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, typename T>
static __device__ void DifMul_8(
T*cvals,int*crows,int*ccols, int crnnz,\
T*avals,int*arows,int*acols, int arnnz,\
T*bvals,int*brows,int*bcols,\
T*dvals,int*drows,int*dcols,int drnnz,\
T alpha){
	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;int* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;int* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;int* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;int* rowIndices7;int rowLength7=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	int t=(threadIdx.x+1)*8;

	if(t<=arnnz){
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
	}
	else if(t-1==arnnz)  //arnnz%8==7
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
	}
	else if(t-2==arnnz) //arnnz%8==6
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int r5=*(acols+d0+5);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
	}
	else if(t-3==arnnz)// arnnz%8==5
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
	}
	else if(t-4==arnnz)// arnnz%8==4
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
	}
	else if(t-5==arnnz)// arnnz%8==3
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
	}
	else if(t-6==arnnz)// arnnz%8==2
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
	}
	else if(t-7==arnnz)// arnnz%8==1
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	if(blockDim.x-1==threadIdx.x){
		rowLength7=drnnz;
		rowValues7=dvals;
		rowIndices7=dcols;
		weight7=alpha;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=*(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=*(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=*(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=*(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=*(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=*(rowIndices7+rowPos7);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	frontIndex=min_index;

	
	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=*(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=*(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=*(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=*(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=*(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=*(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=*(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=*(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
	}
	else
	{
		frontIndex=intMax;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=*(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=*(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=*(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=*(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=*(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=*(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=*(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=*(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=*(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=*(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=*(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=*(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=*(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, typename T>
static __device__ void DifMul_16(
T*cvals,int*crows,int*ccols, int crnnz,\
T*avals,int*arows,int*acols, int arnnz,\
T*bvals,int*brows,int*bcols,\
T*dvals,int*drows,int*dcols,int drnnz,\
T alpha){
	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;int* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;int* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;int* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;int* rowIndices7;int rowLength7=0;//The row for the thread
	T* rowValues8;int* rowIndices8;int rowLength8=0;//The row for the thread
	T* rowValues9;int* rowIndices9;int rowLength9=0;//The row for the thread
	T* rowValues10;int* rowIndices10;int rowLength10=0;//The row for the thread
	T* rowValues11;int* rowIndices11;int rowLength11=0;//The row for the thread
	T* rowValues12;int* rowIndices12;int rowLength12=0;//The row for the thread
	T* rowValues13;int* rowIndices13;int rowLength13=0;//The row for the thread
	T* rowValues14;int* rowIndices14;int rowLength14=0;//The row for the thread
	T* rowValues15;int* rowIndices15;int rowLength15=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	T weight8=0;//The weight for the row
	T weight9=0;//The weight for the row
	T weight10=0;//The weight for the row
	T weight11=0;//The weight for the row
	T weight12=0;//The weight for the row
	T weight13=0;//The weight for the row
	T weight14=0;//The weight for the row
	T weight15=0;//The weight for the row
	int t=(threadIdx.x+1)*16;

	if(t<=arnnz){
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int r12=*(acols+d0+12);//int rowIndex=a.Index(thread);		
		int r13=*(acols+d0+13);
		int r14=*(acols+d0+14);
		int r15=*(acols+d0+15);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		int rowStart12=*(brows+r12);
		int rowStart13=*(brows+r13);
		int rowStart14=*(brows+r14);
		int rowStart15=*(brows+r15);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=*(brows+r12+1)-rowStart12;
		rowLength13=*(brows+r13+1)-rowStart13;
		rowLength14=*(brows+r14+1)-rowStart14;
		rowLength15=*(brows+r15+1)-rowStart15;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		rowValues8=bvals+rowStart8;
		rowIndices8=bcols+rowStart8;
		rowValues9=bvals+rowStart9;
		rowIndices9=bcols+rowStart9;
		rowValues10=bvals+rowStart10;
		rowIndices10=bcols+rowStart10;
		rowValues11=bvals+rowStart11;
		rowIndices11=bcols+rowStart11;
		rowValues12=bvals+rowStart12;
		rowIndices12=bcols+rowStart12;
		rowValues13=bvals+rowStart13;
		rowIndices13=bcols+rowStart13;
		rowValues14=bvals+rowStart14;
		rowIndices14=bcols+rowStart14;
		rowValues15=bvals+rowStart15;
		rowIndices15=bcols+rowStart15;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
		weight8=*(avals+d0+8);//a.Value(thread);
		weight9=*(avals+d0+9);//a.Value(thread);
		weight10=*(avals+d0+10);//a.Value(thread);
		weight11=*(avals+d0+11);//a.Value(thread);
		weight12=*(avals+d0+12);//a.Value(thread);
		weight13=*(avals+d0+13);//a.Value(thread);
		weight14=*(avals+d0+14);//a.Value(thread);
		weight15=*(avals+d0+15);//a.Value(thread);
	}
	else if(t-1==arnnz)  //arnnz%16==15
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int r12=*(acols+d0+12);//int rowIndex=a.Index(thread);		
		int r13=*(acols+d0+13);
		int r14=*(acols+d0+14);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		int rowStart12=*(brows+r12);
		int rowStart13=*(brows+r13);
		int rowStart14=*(brows+r14);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=*(brows+r12+1)-rowStart12;
		rowLength13=*(brows+r13+1)-rowStart13;
		rowLength14=*(brows+r14+1)-rowStart14;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		rowValues8=bvals+rowStart8;
		rowIndices8=bcols+rowStart8;
		rowValues9=bvals+rowStart9;
		rowIndices9=bcols+rowStart9;
		rowValues10=bvals+rowStart10;
		rowIndices10=bcols+rowStart10;
		rowValues11=bvals+rowStart11;
		rowIndices11=bcols+rowStart11;
		rowValues12=bvals+rowStart12;
		rowIndices12=bcols+rowStart12;
		rowValues13=bvals+rowStart13;
		rowIndices13=bcols+rowStart13;
		rowValues14=bvals+rowStart14;
		rowIndices14=bcols+rowStart14;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
		weight8=*(avals+d0+8);//a.Value(thread);
		weight9=*(avals+d0+9);//a.Value(thread);
		weight10=*(avals+d0+10);//a.Value(thread);
		weight11=*(avals+d0+11);//a.Value(thread);
		weight12=*(avals+d0+12);//a.Value(thread);
		weight13=*(avals+d0+13);//a.Value(thread);
		weight14=*(avals+d0+14);//a.Value(thread);
	}
	else if(t-2==arnnz) //arnnz%16==14
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int r12=*(acols+d0+12);//int rowIndex=a.Index(thread);		
		int r13=*(acols+d0+13);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		int rowStart12=*(brows+r12);
		int rowStart13=*(brows+r13);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=*(brows+r12+1)-rowStart12;
		rowLength13=*(brows+r13+1)-rowStart13;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		rowValues8=bvals+rowStart8;
		rowIndices8=bcols+rowStart8;
		rowValues9=bvals+rowStart9;
		rowIndices9=bcols+rowStart9;
		rowValues10=bvals+rowStart10;
		rowIndices10=bcols+rowStart10;
		rowValues11=bvals+rowStart11;
		rowIndices11=bcols+rowStart11;
		rowValues12=bvals+rowStart12;
		rowIndices12=bcols+rowStart12;
		rowValues13=bvals+rowStart13;
		rowIndices13=bcols+rowStart13;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
		weight8=*(avals+d0+8);//a.Value(thread);
		weight9=*(avals+d0+9);//a.Value(thread);
		weight10=*(avals+d0+10);//a.Value(thread);
		weight11=*(avals+d0+11);//a.Value(thread);
		weight12=*(avals+d0+12);//a.Value(thread);
		weight13=*(avals+d0+13);//a.Value(thread);
	}
	else if(t-3==arnnz)// arnnz%16==13
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int r12=*(acols+d0+12);//int rowIndex=a.Index(thread);		
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		int rowStart12=*(brows+r12);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=*(brows+r12+1)-rowStart12;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		rowValues8=bvals+rowStart8;
		rowIndices8=bcols+rowStart8;
		rowValues9=bvals+rowStart9;
		rowIndices9=bcols+rowStart9;
		rowValues10=bvals+rowStart10;
		rowIndices10=bcols+rowStart10;
		rowValues11=bvals+rowStart11;
		rowIndices11=bcols+rowStart11;
		rowValues12=bvals+rowStart12;
		rowIndices12=bcols+rowStart12;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
		weight8=*(avals+d0+8);//a.Value(thread);
		weight9=*(avals+d0+9);//a.Value(thread);
		weight10=*(avals+d0+10);//a.Value(thread);
		weight11=*(avals+d0+11);//a.Value(thread);
		weight12=*(avals+d0+12);//a.Value(thread);
	}
	else if(t-4==arnnz)// arnnz%16==12
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		rowValues8=bvals+rowStart8;
		rowIndices8=bcols+rowStart8;
		rowValues9=bvals+rowStart9;
		rowIndices9=bcols+rowStart9;
		rowValues10=bvals+rowStart10;
		rowIndices10=bcols+rowStart10;
		rowValues11=bvals+rowStart11;
		rowIndices11=bcols+rowStart11;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
		weight8=*(avals+d0+8);//a.Value(thread);
		weight9=*(avals+d0+9);//a.Value(thread);
		weight10=*(avals+d0+10);//a.Value(thread);
		weight11=*(avals+d0+11);//a.Value(thread);
	}
	else if(t-5==arnnz)// arnnz%16==11
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		rowValues8=bvals+rowStart8;
		rowIndices8=bcols+rowStart8;
		rowValues9=bvals+rowStart9;
		rowIndices9=bcols+rowStart9;
		rowValues10=bvals+rowStart10;
		rowIndices10=bcols+rowStart10;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
		weight8=*(avals+d0+8);//a.Value(thread);
		weight9=*(avals+d0+9);//a.Value(thread);
		weight10=*(avals+d0+10);//a.Value(thread);
	}
	else if(t-6==arnnz)// arnnz%16==10
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		rowValues8=bvals+rowStart8;
		rowIndices8=bcols+rowStart8;
		rowValues9=bvals+rowStart9;
		rowIndices9=bcols+rowStart9;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
		weight8=*(avals+d0+8);//a.Value(thread);
		weight9=*(avals+d0+9);//a.Value(thread);
	}
	else if(t-7==arnnz)// arnnz%16==9
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		rowValues8=bvals+rowStart8;
		rowIndices8=bcols+rowStart8;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
		weight8=*(avals+d0+8);//a.Value(thread);
	}
	else if(t-8==arnnz)// arnnz%16==8
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
	}
	else if(t-9==arnnz)// arnnz%16==7
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
	}
	else if(t-10==arnnz)// arnnz%16==6
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
	}
	else if(t-11==arnnz)// arnnz%16==5
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
	}
	else if(t-12==arnnz)// arnnz%16==4
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
	}
	else if(t-13==arnnz)// arnnz%16==3
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
	}
	else if(t-14==arnnz)// arnnz%16==2
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
	}
	else if(t-15==arnnz)// arnnz%16==1
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	if(blockDim.x-1==threadIdx.x){
		rowLength15=drnnz;
		rowValues15=dvals;
		rowIndices15=dcols;
		weight15=alpha;
	}	

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int rowPos8=0;//Current position into row
	int rowPos9=0;//Current position into row
	int rowPos10=0;//Current position into row
	int rowPos11=0;//Current position into row
	int rowPos12=0;//Current position into row
	int rowPos13=0;//Current position into row
	int rowPos14=0;//Current position into row
	int rowPos15=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	int index8=intMax;
	int index9=intMax;
	int index10=intMax;
	int index11=intMax;
	int index12=intMax;
	int index13=intMax;
	int index14=intMax;
	int index15=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=*(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=*(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=*(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=*(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=*(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=*(rowIndices7+rowPos7);
	}
	if(rowPos8<rowLength8){
		index8=*(rowIndices8+rowPos8);
	}
	if(rowPos9<rowLength9){
		index9=*(rowIndices9+rowPos9);
	}
	if(rowPos10<rowLength10){
		index10=*(rowIndices10+rowPos10);
	}
	if(rowPos11<rowLength11){
		index11=*(rowIndices11+rowPos11);
	}
	if(rowPos12<rowLength12){
		index12=*(rowIndices12+rowPos12);
	}
	if(rowPos13<rowLength13){
		index13=*(rowIndices13+rowPos13);
	}
	if(rowPos14<rowLength14){
		index14=*(rowIndices14+rowPos14);
	}
	if(rowPos15<rowLength15){
		index15=*(rowIndices15+rowPos15);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	min_index=index8<min_index?index8:min_index;
	min_index=index9<min_index?index9:min_index;
	min_index=index10<min_index?index10:min_index;
	min_index=index11<min_index?index11:min_index;
	min_index=index12<min_index?index12:min_index;
	min_index=index13<min_index?index13:min_index;
	min_index=index14<min_index?index14:min_index;
	min_index=index15<min_index?index15:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=*(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=*(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=*(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=*(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=*(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=*(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=*(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=*(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
		if(index8==min_index)
		{
			frontValue+=*(rowValues8+rowPos8)*weight8;
			rowPos8++;
		}
		if(index9==min_index)
		{
			frontValue+=*(rowValues9+rowPos9)*weight9;
			rowPos9++;
		}
		if(index10==min_index)
		{
			frontValue+=*(rowValues10+rowPos10)*weight10;
			rowPos10++;
		}
		if(index11==min_index)
		{
			frontValue+=*(rowValues11+rowPos11)*weight11;
			rowPos11++;
		}
		if(index12==min_index)
		{
			frontValue+=*(rowValues12+rowPos12)*weight12;
			rowPos12++;
		}
		if(index13==min_index)
		{
			frontValue+=*(rowValues13+rowPos13)*weight13;
			rowPos13++;
		}
		if(index14==min_index)
		{
			frontValue+=*(rowValues14+rowPos14)*weight14;
			rowPos14++;
		}
		if(index15==min_index)
		{
			frontValue+=*(rowValues15+rowPos15)*weight15;
			rowPos15++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
	//		frontIndex=index0>index1?index1:index0;
	//		frontValue=index0>index1?*(rowValues1+rowPos1)*weight1:*(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//	if(threadIdx.x==1&&threadIdx.y==0)
	//	{
	//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
	//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
	//		printf("weight0=%f,weight1=%f,weight2=%f,weight3=%f,weight4=%f,weight5=%f,weight6=%f,weight7=%f\n",weight0,weight1,weight2,weight3,weight4,weight5,weight6,weight7);
	//		printf("weight8=%f,weight9=%f,weight10=%f,weight11=%f,weight12=%f,weight13=%f,weight14=%f,weight15=%f\n",weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15);
	//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
	//		printf("minFront=%d\n",minFront);
	//		printf("------------------------------------\n");
	//	}
	//	if(threadIdx.x==0&&threadIdx.y==0)
	//	{
	//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
	//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
	//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
	//		printf("minFront=%d\n",minFront);
	//		printf("------------------------------------\n");
	//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=*(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=*(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=*(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=*(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=*(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=*(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}
			if(rowPos8<rowLength8){
				index8=*(rowIndices8+rowPos8);
			}
			else{
				index8=intMax;
			}
			if(rowPos9<rowLength9){
				index9=*(rowIndices9+rowPos9);
			}
			else{
				index9=intMax;
			}
			if(rowPos10<rowLength10){
				index10=*(rowIndices10+rowPos10);
			}
			else{
				index10=intMax;
			}
			if(rowPos11<rowLength11){
				index11=*(rowIndices11+rowPos11);
			}
			else{
				index11=intMax;
			}
			if(rowPos12<rowLength12){
				index12=*(rowIndices12+rowPos12);
			}
			else{
				index12=intMax;
			}
			if(rowPos13<rowLength13){
				index13=*(rowIndices13+rowPos13);
			}
			else{
				index13=intMax;
			}
			if(rowPos14<rowLength14){
				index14=*(rowIndices14+rowPos14);
			}
			else{
				index14=intMax;
			}
			if(rowPos15<rowLength15){
				index15=*(rowIndices15+rowPos15);
			}
			else{
				index15=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			min_index=index8<min_index?index8:min_index;
			min_index=index9<min_index?index9:min_index;
			min_index=index10<min_index?index10:min_index;
			min_index=index11<min_index?index11:min_index;
			min_index=index12<min_index?index12:min_index;
			min_index=index13<min_index?index13:min_index;
			min_index=index14<min_index?index14:min_index;
			min_index=index15<min_index?index15:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=*(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=*(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=*(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=*(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=*(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=*(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=*(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
				if(index8==min_index)
				{
					frontValue+=*(rowValues8+rowPos8)*weight8;
					rowPos8++;
				}
				if(index9==min_index)
				{
					frontValue+=*(rowValues9+rowPos9)*weight9;
					rowPos9++;
				}
				if(index10==min_index)
				{
					frontValue+=*(rowValues10+rowPos10)*weight10;
					rowPos10++;
				}
				if(index11==min_index)
				{
					frontValue+=*(rowValues11+rowPos11)*weight11;
					rowPos11++;
				}
				if(index12==min_index)
				{
					frontValue+=*(rowValues12+rowPos12)*weight12;
					rowPos12++;
				}
				if(index13==min_index)
				{
					frontValue+=*(rowValues13+rowPos13)*weight13;
					rowPos13++;
				}
				if(index14==min_index)
				{
					frontValue+=*(rowValues14+rowPos14)*weight14;
					rowPos14++;
				}
				if(index15==min_index)
				{
					frontValue+=*(rowValues15+rowPos15)*weight15;
					rowPos15++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_4(
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;

    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;
	DifMul_4<WarpSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha);
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_8(
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;

    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;
	DifMul_8<WarpSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha);
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_16(
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;

    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;
	DifMul_16<WarpSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha);
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_1(\
T*cvals,int*crows,int*ccols,int crnnz, \
T*avals,int*arows,int*acols,int arnnz, \
T*bvals,int*brows,int*bcols,\
T*dvals,int*drows,int*dcols,int drnnz,T alpha,\
T *c_val, int* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues;int* rowIndices;int rowLength=0;//The row for the thread
	T weight=0;//The weight for the row
	if(threadIdx.x<arnnz){
		int r=*(acols+threadIdx.x);//int rowIndex=a.Index(thread);		
		int rowStart=*(brows+r);
		rowLength=*(brows+r+1)-rowStart;
		rowValues=bvals+rowStart;
		rowIndices=bcols+rowStart;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight=*(avals+threadIdx.x);//a.Value(thread);
	}

	if(blockDim.x-1==threadIdx.x){
		rowLength=drnnz;
		rowValues=dvals;
		rowIndices=dcols;
		weight=alpha;
	}

	int rowPos=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread
	if(rowPos<rowLength){//Load the front index and row
		frontIndex=*(rowIndices+rowPos);//*: explicit cache usage
		frontValue=*(rowValues+rowPos)*weight;//*: explicit cache usage
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos<rowLength){
				frontValue=*(rowValues+rowPos)*weight;//*: explicit cache usage
				frontIndex=(int)*(rowIndices+rowPos);//*: explicit cache usage
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}

		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();

		sum=WarpSum<WarpSize>(sum);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_2(\
T*cvals,int*crows,int*ccols,int crnnz, \
T*avals,int*arows,int*acols,int arnnz, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,int drnnz,T alpha,\
T *c_val, int* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	int t=(threadIdx.x+1)*2;

	if(t<=arnnz){
		int d0=threadIdx.x*2;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
	}
	else if(t-1==arnnz){

		int d0=threadIdx.x*2;
		int r0=*(acols+d0);
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowLength1=0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
	}

	if(blockDim.x-1==threadIdx.x){
		rowLength1=drnnz;
		rowValues1=dvals;
		rowIndices1=dcols;
		weight1=alpha;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}

	if(index0<index1)
	{
		frontIndex=index0;
		frontValue=*(rowValues0+rowPos0)*weight0;
		rowPos0++;
	}
	else if(index0>index1)
	{
		frontIndex=index1;
		frontValue=*(rowValues1+rowPos1)*weight1;
		rowPos1++;
	}
	else
	{
		if(index0!=intMax)
		{
			frontIndex=index0;
			frontValue=*(rowValues0+rowPos0)*weight0+*(rowValues1+rowPos1)*weight1;
			rowPos0++;
			rowPos1++;
		}
		else
		{
		}
	}


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(index0<index1)
			{
				frontIndex=index0;
				frontValue=*(rowValues0+rowPos0)*weight0;
				rowPos0++;
			}
			else if(index0>index1)
			{
				frontIndex=index1;
				frontValue=*(rowValues1+rowPos1)*weight1;
				rowPos1++;
			}
			else 
			{
				if(index0!=intMax)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0+*(rowValues1+rowPos1)*weight1;
					rowPos0++;
					rowPos1++;
				}
				else
				{
					frontIndex=intMax;
				}
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}

		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();

		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);
		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_4(\
T*cvals,int*crows,int*ccols,int crnnz, \
T*avals,int*arows,int*acols,int arnnz, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,int drnnz,T alpha,\
T *c_val, int* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}
	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	int t=(threadIdx.x+1)*4;

	if(t<=arnnz){
		int d0=threadIdx.x*4;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
	}
	else if(t-1==arnnz)  //arnnz%4==3
	{
		int d0=threadIdx.x*4;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
	}
	else if(t-2==arnnz) //arnnz%4==2
	{
		int d0=threadIdx.x*4;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
	}
	else if(t-3==arnnz) //arnnz%4==1
	{
		int d0=threadIdx.x*4;
		int r0=*(acols+d0);
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);

	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}

	if(blockDim.x-1==threadIdx.x){
		rowLength3=drnnz;
		rowValues3=dvals;
		rowIndices3=dcols;
		weight3=alpha;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=*(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=*(rowIndices3+rowPos3);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontValue=*(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=*(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=*(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=*(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
	}
	else
	{
		frontIndex=intMax;
	}


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=*(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=*(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=*(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=*(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=*(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}

		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();

		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);
		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_8(\
T*cvals,int*crows,int*ccols,int crnnz, \
T*avals,int*arows,int*acols,int arnnz, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,int drnnz,T alpha,\
T *c_val, int* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			cvals[i]=alpha*dvals[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;int* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;int* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;int* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;int* rowIndices7;int rowLength7=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	int t=(threadIdx.x+1)*8;

	if(t<=arnnz){
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
	}
	else if(t-1==arnnz)  //arnnz%8==7
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
	}
	else if(t-2==arnnz) //arnnz%8==6
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int r5=*(acols+d0+5);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
	}
	else if(t-3==arnnz)// arnnz%8==5
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
	}
	else if(t-4==arnnz)// arnnz%8==4
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
	}
	else if(t-5==arnnz)// arnnz%8==3
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
	}
	else if(t-6==arnnz)// arnnz%8==2
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
	}
	else if(t-7==arnnz)// arnnz%8==1
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}

	if(blockDim.x-1==threadIdx.x){
		rowLength7=drnnz;
		rowValues7=dvals;
		rowIndices7=dcols;
		weight7=alpha;
	}
	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=*(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=*(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=*(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=*(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=*(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=*(rowIndices7+rowPos7);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	frontIndex=min_index;

	
	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=*(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=*(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=*(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=*(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=*(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=*(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=*(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=*(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
//		frontIndex=index0>index1?index1:index0;
//		frontValue=index0>index1?*(rowValues1+rowPos1)*weight1:*(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

//	if(threadIdx.x==1&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
//	if(threadIdx.x==0&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=*(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=*(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=*(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=*(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=*(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=*(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=*(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=*(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=*(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=*(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=*(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=*(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=*(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}
		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();
		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		

		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			cvals[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		

	}
}

template<int WarpSize, int SegmentSize, int halfNUM, typename T>
static __device__ void MulOverWarp_8_halfdown(\
T*cvals,int*crows,int*ccols,int crnnz, \
T*avals,int*arows,int*acols,int arnnz, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,int drnnz,T alpha,\
T *c_val, int* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

/*
	if(arnnz-halfNUM==0)//nothing to do
		return;
	else if(arnnz-halfNUM==1){//simply scale the vector (faster)
		T weight=avals[0+halfNUM];
		int tma=acols[0+halfNUM];
        int tmb=brows[tma];
        int bnnz=brows[tma+1]-tmb;
        int*tbcols=bcols+tmb;
        T*tbvals=bvals+tmb;
		//CSparseVector<T> b=B.GetRow(acols[0+halfNUM]);
        for(int j=threadIdx.x; j<bnnz; j+=blockDim.x)
        {
            for(int i=0;i<crnnz;i++)
            {
                if(ccols[i]==tbcols[j])
                {
                    cvals[i]+=weight*tbvals[j];
                    break;
                }
            }
        
        }
		return;
	}
*/

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;int* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;int* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;int* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;int* rowIndices7;int rowLength7=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	int t=(threadIdx.x+1)*8;

	if(t<=arnnz-halfNUM){
		int d0=threadIdx.x*8;
		int r0=*(acols+halfNUM+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+halfNUM+d0+1);
		int r2=*(acols+halfNUM+d0+2);
		int r3=*(acols+halfNUM+d0+3);
		int r4=*(acols+halfNUM+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+halfNUM+d0+5);
		int r6=*(acols+halfNUM+d0+6);
		int r7=*(acols+halfNUM+d0+7);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+halfNUM+d0);//a.Value(thread);
		weight1=*(avals+halfNUM+d0+1);//a.Value(thread);
		weight2=*(avals+halfNUM+d0+2);//a.Value(thread);
		weight3=*(avals+halfNUM+d0+3);//a.Value(thread);
		weight4=*(avals+halfNUM+d0+4);//a.Value(thread);
		weight5=*(avals+halfNUM+d0+5);//a.Value(thread);
		weight6=*(avals+halfNUM+d0+6);//a.Value(thread);
		weight7=*(avals+halfNUM+d0+7);//a.Value(thread);
	}
	else if(t-1==arnnz-halfNUM)  //arnnz%8==7
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+halfNUM+d0);
		int r1=*(acols+halfNUM+d0+1);
		int r2=*(acols+halfNUM+d0+2);
		int r3=*(acols+halfNUM+d0+3);
		int r4=*(acols+halfNUM+d0+4);
		int r5=*(acols+halfNUM+d0+5);
		int r6=*(acols+halfNUM+d0+6);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+halfNUM+d0);//a.Value(thread);
		weight1=*(avals+halfNUM+d0+1);//a.Value(thread);
		weight2=*(avals+halfNUM+d0+2);//a.Value(thread);
		weight3=*(avals+halfNUM+d0+3);//a.Value(thread);
		weight4=*(avals+halfNUM+d0+4);//a.Value(thread);
		weight5=*(avals+halfNUM+d0+5);//a.Value(thread);
		weight6=*(avals+halfNUM+d0+6);//a.Value(thread);
	}
	else if(t-2==arnnz-halfNUM) //arnnz%8==6
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+halfNUM+d0);
		int r1=*(acols+halfNUM+d0+1);
		int r2=*(acols+halfNUM+d0+2);
		int r3=*(acols+halfNUM+d0+3);
		int r4=*(acols+halfNUM+d0+4);
		int r5=*(acols+halfNUM+d0+5);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+halfNUM+d0);//a.Value(thread);
		weight1=*(avals+halfNUM+d0+1);//a.Value(thread);
		weight2=*(avals+halfNUM+d0+2);//a.Value(thread);
		weight3=*(avals+halfNUM+d0+3);//a.Value(thread);
		weight4=*(avals+halfNUM+d0+4);//a.Value(thread);
		weight5=*(avals+halfNUM+d0+5);//a.Value(thread);
	}
	else if(t-3==arnnz-halfNUM)// arnnz%8==5
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+halfNUM+d0);
		int r1=*(acols+halfNUM+d0+1);
		int r2=*(acols+halfNUM+d0+2);
		int r3=*(acols+halfNUM+d0+3);
		int r4=*(acols+halfNUM+d0+4);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+halfNUM+d0);//a.Value(thread);
		weight1=*(avals+halfNUM+d0+1);//a.Value(thread);
		weight2=*(avals+halfNUM+d0+2);//a.Value(thread);
		weight3=*(avals+halfNUM+d0+3);//a.Value(thread);
		weight4=*(avals+halfNUM+d0+4);//a.Value(thread);
	}
	else if(t-4==arnnz-halfNUM)// arnnz%8==4
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+halfNUM+d0);
		int r1=*(acols+halfNUM+d0+1);
		int r2=*(acols+halfNUM+d0+2);
		int r3=*(acols+halfNUM+d0+3);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+halfNUM+d0);//a.Value(thread);
		weight1=*(avals+halfNUM+d0+1);//a.Value(thread);
		weight2=*(avals+halfNUM+d0+2);//a.Value(thread);
		weight3=*(avals+halfNUM+d0+3);//a.Value(thread);
	}
	else if(t-5==arnnz-halfNUM)// arnnz%8==3
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+halfNUM+d0);
		int r1=*(acols+halfNUM+d0+1);
		int r2=*(acols+halfNUM+d0+2);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+halfNUM+d0);//a.Value(thread);
		weight1=*(avals+halfNUM+d0+1);//a.Value(thread);
		weight2=*(avals+halfNUM+d0+2);//a.Value(thread);
	}
	else if(t-6==arnnz-halfNUM)// arnnz%8==2
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+halfNUM+d0);
		int r1=*(acols+halfNUM+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+halfNUM+d0);//a.Value(thread);
		weight1=*(avals+halfNUM+d0+1);//a.Value(thread);
	}
	else if(t-7==arnnz-halfNUM)// arnnz%8==1
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+halfNUM+d0);
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+halfNUM+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}

	if(blockDim.x-1==threadIdx.x){
		weight7=alpha;
		rowLength7=drnnz;
		rowValues7=dvals;
		rowIndices7=dcols;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=*(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=*(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=*(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=*(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=*(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=*(rowIndices7+rowPos7);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	frontIndex=min_index;

	
	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=*(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=*(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=*(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=*(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=*(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=*(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=*(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=*(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
//		frontIndex=index0>index1?index1:index0;
//		frontValue=index0>index1?*(rowValues1+rowPos1)*weight1:*(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

//	int dstPos=0;

//	if(threadIdx.x==1&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
//	if(threadIdx.x==0&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=*(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=*(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=*(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=*(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=*(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=*(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=*(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=*(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=*(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=*(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=*(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=*(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=*(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}
		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();
		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		

		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)

            for(int i=0;i<crnnz;i++)
            {
                if(ccols[i]==bufferedIndex)
                {
                    cvals[i]+=bufferedValue;
                    break;
                }
            }

//            dstPos+=blockDim.x;
			bufferPos=0;
		}		

	}
}

template<int WarpSize, int SegmentSize, int halfNUM, typename T>
static __device__ void MulOverWarp_8_halfup(\
T*cvals,int*crows,int*ccols,int crnnz, \
T*avals,int*arows,int*acols,int arnnz, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,int drnnz,T alpha,\
T *c_val, int* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;int* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;int* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;int* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;int* rowIndices7;int rowLength7=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	int t=(threadIdx.x+1)*8;

	if(t<=halfNUM){
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		rowValues7=bvals+rowStart7;
		rowIndices7=bcols+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
		weight7=*(avals+d0+7);//a.Value(thread);
	}
	else if(t-1==halfNUM)  //arnnz%8==7
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		rowValues6=bvals+rowStart6;
		rowIndices6=bcols+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
		weight6=*(avals+d0+6);//a.Value(thread);
	}
	else if(t-2==halfNUM) //arnnz%8==6
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int r5=*(acols+d0+5);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		rowValues5=bvals+rowStart5;
		rowIndices5=bcols+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
		weight5=*(avals+d0+5);//a.Value(thread);
	}
	else if(t-3==halfNUM)// arnnz%8==5
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		rowValues4=bvals+rowStart4;
		rowIndices4=bcols+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
		weight4=*(avals+d0+4);//a.Value(thread);
	}
	else if(t-4==halfNUM)// arnnz%8==4
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		rowValues3=bvals+rowStart3;
		rowIndices3=bcols+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
		weight3=*(avals+d0+3);//a.Value(thread);
	}
	else if(t-5==halfNUM)// arnnz%8==3
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		rowValues2=bvals+rowStart2;
		rowIndices2=bcols+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
		weight2=*(avals+d0+2);//a.Value(thread);
	}
	else if(t-6==halfNUM)// arnnz%8==2
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int r1=*(acols+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		rowValues1=bvals+rowStart1;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
		weight1=*(avals+d0+1);//a.Value(thread);
	}
	else if(t-7==halfNUM)// arnnz%8==1
	{
		int d0=threadIdx.x*8;
		int r0=*(acols+d0);
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=bvals+rowStart0;
		rowIndices0=bcols+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=*(avals+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=*(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=*(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=*(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=*(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=*(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=*(rowIndices7+rowPos7);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	frontIndex=min_index;

	
	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=*(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=*(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=*(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=*(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=*(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=*(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=*(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=*(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
//		frontIndex=index0>index1?index1:index0;
//		frontValue=index0>index1?*(rowValues1+rowPos1)*weight1:*(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);


//	if(threadIdx.x==1&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
//	if(threadIdx.x==0&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=*(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=*(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=*(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=*(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=*(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=*(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=*(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=*(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=*(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=*(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=*(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=*(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=*(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=*(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}
		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();
		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		

		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
            for(int i=0; i<crnnz; i++)
            {
                if(ccols[i] == bufferedIndex)
                {
                    cvals[i] = bufferedValue;
                }
            }
//			ccols[dstPos+threadIdx.x]=bufferedIndex;
//			cvals[dstPos+threadIdx.x]=bufferedValue;
//			dstPos+=blockDim.x;
			bufferPos=0;
		}		

	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarpColumn_16(\
T*cvals,int*crows,int*ccols,int crnnz, \
T*avals,int*arows,int*acols,int arnnz, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,int drnnz,T alpha,\
T *c_val, int* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;
	if(arnnz==0){//nothing to do
		for(int i=threadIdx.x;i<crnnz;i+=WarpSize){
			ccols[i]=dcols[i];
			//cvals[i]=alpha*dvals[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
    int* rowIndices0;int rowLength0=0;//The row for the thread
	int* rowIndices1;int rowLength1=0;//The row for the thread
	int* rowIndices2;int rowLength2=0;//The row for the thread
	int* rowIndices3;int rowLength3=0;//The row for the thread
	int* rowIndices4;int rowLength4=0;//The row for the thread
	int* rowIndices5;int rowLength5=0;//The row for the thread
	int* rowIndices6;int rowLength6=0;//The row for the thread
	int* rowIndices7;int rowLength7=0;//The row for the thread
	int* rowIndices8;int rowLength8=0;//The row for the thread
	int* rowIndices9;int rowLength9=0;//The row for the thread
	int* rowIndices10;int rowLength10=0;//The row for the thread
	int* rowIndices11;int rowLength11=0;//The row for the thread
	int* rowIndices12;int rowLength12=0;//The row for the thread
	int* rowIndices13;int rowLength13=0;//The row for the thread
	int* rowIndices14;int rowLength14=0;//The row for the thread
	int* rowIndices15;int rowLength15=0;//The row for the thread
	int t=(threadIdx.x+1)*16;

	if(t<=arnnz){
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int r12=*(acols+d0+12);//int rowIndex=a.Index(thread);		
		int r13=*(acols+d0+13);
		int r14=*(acols+d0+14);
		int r15=*(acols+d0+15);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		int rowStart12=*(brows+r12);
		int rowStart13=*(brows+r13);
		int rowStart14=*(brows+r14);
		int rowStart15=*(brows+r15);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=*(brows+r12+1)-rowStart12;
		rowLength13=*(brows+r13+1)-rowStart13;
		rowLength14=*(brows+r14+1)-rowStart14;
		rowLength15=*(brows+r15+1)-rowStart15;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		rowIndices8=bcols+rowStart8;
		rowIndices9=bcols+rowStart9;
		rowIndices10=bcols+rowStart10;
		rowIndices11=bcols+rowStart11;
		rowIndices12=bcols+rowStart12;
		rowIndices13=bcols+rowStart13;
		rowIndices14=bcols+rowStart14;
		rowIndices15=bcols+rowStart15;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-1==arnnz)  //arnnz%16==15
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int r12=*(acols+d0+12);//int rowIndex=a.Index(thread);		
		int r13=*(acols+d0+13);
		int r14=*(acols+d0+14);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		int rowStart12=*(brows+r12);
		int rowStart13=*(brows+r13);
		int rowStart14=*(brows+r14);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=*(brows+r12+1)-rowStart12;
		rowLength13=*(brows+r13+1)-rowStart13;
		rowLength14=*(brows+r14+1)-rowStart14;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		rowIndices8=bcols+rowStart8;
		rowIndices9=bcols+rowStart9;
		rowIndices10=bcols+rowStart10;
		rowIndices11=bcols+rowStart11;
		rowIndices12=bcols+rowStart12;
		rowIndices13=bcols+rowStart13;
		rowIndices14=bcols+rowStart14;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-2==arnnz) //arnnz%16==14
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int r12=*(acols+d0+12);//int rowIndex=a.Index(thread);		
		int r13=*(acols+d0+13);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		int rowStart12=*(brows+r12);
		int rowStart13=*(brows+r13);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=*(brows+r12+1)-rowStart12;
		rowLength13=*(brows+r13+1)-rowStart13;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		rowIndices8=bcols+rowStart8;
		rowIndices9=bcols+rowStart9;
		rowIndices10=bcols+rowStart10;
		rowIndices11=bcols+rowStart11;
		rowIndices12=bcols+rowStart12;
		rowIndices13=bcols+rowStart13;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-3==arnnz)// arnnz%16==13
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int r12=*(acols+d0+12);//int rowIndex=a.Index(thread);		
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		int rowStart12=*(brows+r12);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=*(brows+r12+1)-rowStart12;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		rowIndices8=bcols+rowStart8;
		rowIndices9=bcols+rowStart9;
		rowIndices10=bcols+rowStart10;
		rowIndices11=bcols+rowStart11;
		rowIndices12=bcols+rowStart12;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-4==arnnz)// arnnz%16==12
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int r11=*(acols+d0+11);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		int rowStart11=*(brows+r11);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=*(brows+r11+1)-rowStart11;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		rowIndices8=bcols+rowStart8;
		rowIndices9=bcols+rowStart9;
		rowIndices10=bcols+rowStart10;
		rowIndices11=bcols+rowStart11;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-5==arnnz)// arnnz%16==11
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int r10=*(acols+d0+10);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		int rowStart10=*(brows+r10);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=*(brows+r10+1)-rowStart10;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		rowIndices8=bcols+rowStart8;
		rowIndices9=bcols+rowStart9;
		rowIndices10=bcols+rowStart10;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-6==arnnz)// arnnz%16==10
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int r9=*(acols+d0+9);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		int rowStart9=*(brows+r9);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=*(brows+r9+1)-rowStart9;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		rowIndices8=bcols+rowStart8;
		rowIndices9=bcols+rowStart9;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-7==arnnz)// arnnz%16==9
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int r8=*(acols+d0+8);//int rowIndex=a.Index(thread);		
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		int rowStart8=*(brows+r8);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=*(brows+r8+1)-rowStart8;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		rowIndices8=bcols+rowStart8;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-8==arnnz)// arnnz%16==8
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int r7=*(acols+d0+7);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		int rowStart7=*(brows+r7);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=*(brows+r7+1)-rowStart7;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		rowIndices7=bcols+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-9==arnnz)// arnnz%16==7
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int r6=*(acols+d0+6);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		int rowStart6=*(brows+r6);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=*(brows+r6+1)-rowStart6;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		rowIndices6=bcols+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-10==arnnz)// arnnz%16==6
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int r5=*(acols+d0+5);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		int rowStart5=*(brows+r5);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=*(brows+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		rowIndices5=bcols+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-11==arnnz)// arnnz%16==5
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int r4=*(acols+d0+4);//int rowIndex=a.Index(thread);		
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		int rowStart4=*(brows+r4);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=*(brows+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		rowIndices4=bcols+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-12==arnnz)// arnnz%16==4
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int r3=*(acols+d0+3);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		int rowStart3=*(brows+r3);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=*(brows+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		rowIndices3=bcols+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-13==arnnz)// arnnz%16==3
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int r2=*(acols+d0+2);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		int rowStart2=*(brows+r2);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=*(brows+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		rowIndices2=bcols+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-14==arnnz)// arnnz%16==2
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int r1=*(acols+d0+1);
		int rowStart0=*(brows+r0);
		int rowStart1=*(brows+r1);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=*(brows+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		rowIndices1=bcols+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-15==arnnz)// arnnz%16==1
	{
		int d0=threadIdx.x*16;
		int r0=*(acols+d0);//int rowIndex=a.Index(thread);		
		int rowStart0=*(brows+r0);
		rowLength0=*(brows+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=bcols+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}


	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int rowPos8=0;//Current position into row
	int rowPos9=0;//Current position into row
	int rowPos10=0;//Current position into row
	int rowPos11=0;//Current position into row
	int rowPos12=0;//Current position into row
	int rowPos13=0;//Current position into row
	int rowPos14=0;//Current position into row
	int rowPos15=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	int index8=intMax;
	int index9=intMax;
	int index10=intMax;
	int index11=intMax;
	int index12=intMax;
	int index13=intMax;
	int index14=intMax;
	int index15=intMax;
	if(rowPos0<rowLength0){
		index0=*(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=*(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=*(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=*(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=*(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=*(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=*(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=*(rowIndices7+rowPos7);
	}
	if(rowPos8<rowLength8){
		index8=*(rowIndices8+rowPos8);
	}
	if(rowPos9<rowLength9){
		index9=*(rowIndices9+rowPos9);
	}
	if(rowPos10<rowLength10){
		index10=*(rowIndices10+rowPos10);
	}
	if(rowPos11<rowLength11){
		index11=*(rowIndices11+rowPos11);
	}
	if(rowPos12<rowLength12){
		index12=*(rowIndices12+rowPos12);
	}
	if(rowPos13<rowLength13){
		index13=*(rowIndices13+rowPos13);
	}
	if(rowPos14<rowLength14){
		index14=*(rowIndices14+rowPos14);
	}
	if(rowPos15<rowLength15){
		index15=*(rowIndices15+rowPos15);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	min_index=index8<min_index?index8:min_index;
	min_index=index9<min_index?index9:min_index;
	min_index=index10<min_index?index10:min_index;
	min_index=index11<min_index?index11:min_index;
	min_index=index12<min_index?index12:min_index;
	min_index=index13<min_index?index13:min_index;
	min_index=index14<min_index?index14:min_index;
	min_index=index15<min_index?index15:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			rowPos1++;
		}
		if(index2==min_index)
		{
			rowPos2++;
		}
		if(index3==min_index)
		{
			rowPos3++;
		}
		if(index4==min_index)
		{
			rowPos4++;
		}
		if(index5==min_index)
		{
			rowPos5++;
		}
		if(index6==min_index)
		{
			rowPos6++;
		}
		if(index7==min_index)
		{
			rowPos7++;
		}
		if(index8==min_index)
		{
			rowPos8++;
		}
		if(index9==min_index)
		{
			rowPos9++;
		}
		if(index10==min_index)
		{
			rowPos10++;
		}
		if(index11==min_index)
		{
			rowPos11++;
		}
		if(index12==min_index)
		{
			rowPos12++;
		}
		if(index13==min_index)
		{
			rowPos13++;
		}
		if(index14==min_index)
		{
			rowPos14++;
		}
		if(index15==min_index)
		{
			rowPos15++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
	//		frontIndex=index0>index1?index1:index0;
	//		frontValue=index0>index1?*(rowValues1+rowPos1)*weight1:*(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	int bufferedIndex;//Thread i stores result i in its register
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		if(frontIndex==minFront){//put these into tmp and load next elements
			//load next
			if(rowPos0<rowLength0){
				index0=*(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=*(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=*(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=*(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=*(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=*(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=*(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=*(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}
			if(rowPos8<rowLength8){
				index8=*(rowIndices8+rowPos8);
			}
			else{
				index8=intMax;
			}
			if(rowPos9<rowLength9){
				index9=*(rowIndices9+rowPos9);
			}
			else{
				index9=intMax;
			}
			if(rowPos10<rowLength10){
				index10=*(rowIndices10+rowPos10);
			}
			else{
				index10=intMax;
			}
			if(rowPos11<rowLength11){
				index11=*(rowIndices11+rowPos11);
			}
			else{
				index11=intMax;
			}
			if(rowPos12<rowLength12){
				index12=*(rowIndices12+rowPos12);
			}
			else{
				index12=intMax;
			}
			if(rowPos13<rowLength13){
				index13=*(rowIndices13+rowPos13);
			}
			else{
				index13=intMax;
			}
			if(rowPos14<rowLength14){
				index14=*(rowIndices14+rowPos14);
			}
			else{
				index14=intMax;
			}
			if(rowPos15<rowLength15){
				index15=*(rowIndices15+rowPos15);
			}
			else{
				index15=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			min_index=index8<min_index?index8:min_index;
			min_index=index9<min_index?index9:min_index;
			min_index=index10<min_index?index10:min_index;
			min_index=index11<min_index?index11:min_index;
			min_index=index12<min_index?index12:min_index;
			min_index=index13<min_index?index13:min_index;
			min_index=index14<min_index?index14:min_index;
			min_index=index15<min_index?index15:min_index;
			frontIndex=min_index;

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					rowPos1++;
				}
				if(index2==min_index)
				{
					rowPos2++;
				}
				if(index3==min_index)
				{
					rowPos3++;
				}
				if(index4==min_index)
				{
					rowPos4++;
				}
				if(index5==min_index)
				{
					rowPos5++;
				}
				if(index6==min_index)
				{
					rowPos6++;
				}
				if(index7==min_index)
				{
					rowPos7++;
				}
				if(index8==min_index)
				{
					rowPos8++;
				}
				if(index9==min_index)
				{
					rowPos9++;
				}
				if(index10==min_index)
				{
					rowPos10++;
				}
				if(index11==min_index)
				{
					rowPos11++;
				}
				if(index12==min_index)
				{
					rowPos12++;
				}
				if(index13==min_index)
				{
					rowPos13++;
				}
				if(index14==min_index)
				{
					rowPos14++;
				}
				if(index15==min_index)
				{
					rowPos15++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}


		__syncthreads();

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(int)minFront;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			ccols[dstPos+threadIdx.x]=bufferedIndex;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}


template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_1(\
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{


	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;

	int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;

	__shared__ T c_val[32];
	__shared__ int c_indices[32];

	MulOverWarp_1<WarpSize, SegmentSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha,\
    c_val,c_indices);
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_2(\
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{


	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;
	    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;


	__shared__ T c_val[32];
	__shared__ int c_indices[32];

	MulOverWarp_2<WarpSize, SegmentSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha,\
    c_val,c_indices);
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_4(\
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{


	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;
	    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;


	__shared__ T c_val[32];
	__shared__ int c_indices[32];

	MulOverWarp_4<WarpSize, SegmentSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha,\
    c_val,c_indices);
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_8(\
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;
	    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;


	__shared__ T c_val[32];
	__shared__ int c_indices[32];

	MulOverWarp_8<WarpSize, SegmentSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha,\
    c_val,c_indices);

}


template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmColumnOverWarpKernel_16(T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;
	    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;

    T*c_val;
	__shared__ int c_indices[32];

	MulOverWarpColumn_16<WarpSize, SegmentSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha,\
    c_val,c_indices);
}
/*
template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmValueOverWarpKernel_16(T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;
	    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;


    __shared__ T c_val[32];
	__shared__ int c_indices[32];

	MulOverWarpValue_16<WarpSize, SegmentSize>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha,\
    c_val,c_indices);

}
*/
template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_8_halfup(T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{

    int tid=threadIdx.y+blockIdx.x*blockDim.y;
    if(tid>=(Queue_one[position+1]-Queue_one[position]))
    {
        return; 
    }
    int r=Queue[Queue_one[position]+tid];

	    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;
	    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;


    __shared__ T c_val[32];
    __shared__ int c_indices[32];

    MulOverWarp_8_halfup<WarpSize, SegmentSize, 4096>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,\
	alpha,\
    c_val,c_indices);

}
    template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_8_halfdown(T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue,\
int* Queue_one, \
int position)
{

    int tid=threadIdx.y+blockIdx.x*blockDim.y;
    if(tid>=(Queue_one[position+1]-Queue_one[position]))
    {
        return; 
    }
    int r=Queue[Queue_one[position]+tid];

	    int crow0=crows[r];
    int crnnz=crows[r+1]-crow0;
	    int arow0=arows[r];
    int arnnz=arows[r+1]-arow0;

	int drow0=drows[r];
	int drnnz=drows[r+1]-drow0;


    __shared__ T c_val[32];
    __shared__ int c_indices[32];

    MulOverWarp_8_halfdown<WarpSize, SegmentSize, 4096>(\
    cvals+crow0,crows,ccols+crow0,crnnz,\
    avals+arow0,arows,acols+arow0,arnnz,\
    bvals,brows,bcols,\
	dvals+drow0,drows,dcols+drow0,drnnz,alpha,\
    c_val,c_indices);
}


/*
static inline int DivUp(int a,int b){
    return (a+b-1)/b;
}
*/
extern hipStream_t stream[13];
template< typename T>
void __cdecl DifSpmmWarp(\
T*cvals,int*crows,int*ccols, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int*Queue, \
int*Queue_one, \
int*h_queue_one)
{
    int threadnum=1024;
/*
    hipStream_t stream[13];
    for(int i=0; i<13; i++)
    {
        hipStreamCreate(&stream[i]);
    }
*/
     int count;
	hipDeviceSynchronize();
	//record_time("calstart");
    for(int i=0; i<13; i++)
    {
        count = h_queue_one[i+1] - h_queue_one[i];

	//hipDeviceSynchronize();
	//record_time("lastend");
	//std::cout<<"last i="<<i<<"\n";
        if(count==0)//count==0
            continue;
        if(i==0)  //0<rowLength<=2
        {
            dim3 blockDim(2,threadnum/2,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmWarpKernel_1<2> <<< gridDim, blockDim, 0, stream[0]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmWarpKernel_1<2>,gridDim, blockDim, 0, stream[0],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==1) //2<rowLength<=4
        {
            dim3 blockDim(4,threadnum/4,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmWarpKernel_1<4> <<< gridDim, blockDim, 0, stream[1]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmWarpKernel_1<4>,gridDim, blockDim, 0, stream[1],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==2) //4<rowLength<=8
        {
            dim3 blockDim(8,threadnum/8,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmWarpKernel_1<8> <<< gridDim, blockDim, 0, stream[2]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmWarpKernel_1<8>,gridDim, blockDim, 0, stream[2],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==3) //8<rowLength<=16
        {
            dim3 blockDim(16,threadnum/16,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmWarpKernel_1<16> <<< gridDim, blockDim, 0, stream[3]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
		hipLaunchKernelGGL(DifSpmmWarpKernel_1<16>,gridDim, blockDim, 0, stream[3],\
		cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
		dvals,drows,dcols,\
		alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==4) //16<rowLength<=32
        {
            dim3 blockDim(32,threadnum/32,1);
            dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
            //DifSpmmWarpKernel_1<32> <<< gridDim, blockDim, 0, stream[4]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmWarpKernel_1<32>,gridDim, blockDim, 0, stream[4],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==5) //32<rowLength<=64
        {
            dim3 blockDim(64,threadnum/64,1);
            dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
            //DifSpmmWarpKernel_1<64> <<< gridDim, blockDim, 0, stream[5]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
            //			DifSpmmOverWarpKernel_1<32, 2> <<<gridDim, blockDim, 0, stream[9]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmWarpKernel_1<64>,gridDim, blockDim, 0, stream[5],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==6) //64<rowLength<=128
        {
            dim3 blockDim(64,8,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmWarpKernel_4<32> <<< gridDim, blockDim, 0, stream[6]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
            //			DifSpmmOverWarpKernel_2<32, 2> <<<gridDim, blockDim, 0, stream[9]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmWarpKernel_2<64>,gridDim, blockDim, 0, stream[6],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
		
        }
        else if(i==7) //128<rowLength<=256
        {
            dim3 blockDim(64,8,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //			DifSpmmOverWarpKernel_4<32, 2> <<<gridDim, blockDim, 0, stream[9]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
            //DifSpmmWarpKernel_8<32> <<< gridDim, blockDim, 0, stream[7]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmWarpKernel_4<64>,gridDim, blockDim, 0, stream[7],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==8) //256<rowLength<=512
        {
            dim3 blockDim(512,1,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmOverWarpKernel_1<64, 8> <<<gridDim, blockDim, 0, stream[8]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
            //DifSpmmWarpKernel_16<32> <<< gridDim, blockDim, 0, stream[8]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmOverWarpKernel_1<64, 8>),gridDim, blockDim, 0, stream[8],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==9) //512<rowLength<=1024
        {
            dim3 blockDim(256,2,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmOverWarpKernel_2<64, 8> <<<gridDim, blockDim, 0, stream[9]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
	hipLaunchKernelGGL((DifSpmmOverWarpKernel_2<64, 8>),gridDim, blockDim, 0, stream[9],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==10) //1024<rowLength<=2048
        {
            dim3 blockDim(512,1,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmOverWarpKernel_4<64, 8> <<<gridDim, blockDim, 0, stream[10]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmOverWarpKernel_4<64, 8>),gridDim, blockDim, 0, stream[10],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
        }
        else if(i==11) //2048<rowLength<=4096
        {
            dim3 blockDim(512,1,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //DifSpmmOverWarpKernel_8<64, 8> <<<gridDim, blockDim, 0, stream[11]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
		hipLaunchKernelGGL((DifSpmmOverWarpKernel_8<64, 8>),gridDim, blockDim, 0, stream[11],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);

        }
        else //if(i==12) //rowLength>4096
        {
            dim3 blockDim(512,1,1);
            dim3 gridDim(DivUp(count,( int)blockDim.y),1,1);
            //            DifSpmmOverWarpKernel_16<32, 16> <<<gridDim, blockDim, 0, stream[12]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
            //DifSpmmColumnOverWarpKernel_16<64,8> <<<gridDim, blockDim, 0, stream[12]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
		hipLaunchKernelGGL((DifSpmmColumnOverWarpKernel_16<64, 8>),gridDim, blockDim, 0, stream[12],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
            //DifSpmmOverWarpKernel_8_halfup<64, 8> <<<gridDim, blockDim, 0, stream[12]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmOverWarpKernel_8_halfup<64, 8>),gridDim, blockDim, 0, stream[12],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
            //DifSpmmOverWarpKernel_8_halfdown<64,8> <<<gridDim, blockDim, 0, stream[12]>>>(cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmOverWarpKernel_8_halfdown<64,8>),gridDim, blockDim, 0, stream[12],\
			cvals,crows,ccols,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
				alpha,\
                m,k,n,\
                Queue,Queue_one,i);
		}
	}
	hipDeviceSynchronize();
	//record_time("calend");
/*
	for(int i=0; i<13; i++)
	{
		hipStreamDestroy(stream[i]);
	}
*/
}
