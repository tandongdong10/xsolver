template<int WarpSize, typename T>
static __device__ T WarpMin(T value){
	// Use XOR mode to perform butterfly reduction	
	for(int i=WarpSize/2; i>=1; i/=2){
		T tmp=__shfl_xor(value, i);
		value=value<tmp?value:tmp;
	}
	return value;
}

template<typename T>
static inline __device__ void bgetrow(int*brows,int r,T*&values,int*&cols,int&rowlen){
	int br=brows[r];
	rowlen=brows[r+1]-br;
	values+=br;
	cols+=br;
}

template<int WarpSize, typename T>
static __device__ int MulWarpPredictSize(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n){
	if(nnza==0)
	{
		return nnzd;
	}
		
	const int intMax=n;	
	T* rowValues;int* rowIndices;int rowLength=0;//The row for the thread


	if(blockDim.x-1==threadIdx.x){
		rowValues=dvals;
		rowIndices=dcols;
		rowLength=nnzd;
	}
	else if(threadIdx.x<nnza)
	{
            int tema=acols[threadIdx.x];
            int temb=brows[tema];
            rowValues=bvals+temb;
            rowIndices=bcols+temb;
            rowLength=brows[tema+1]-temb;
            //bgetrow(brows,acols[threadIdx.x],rowValues,rowIndices,rowLength);
        }



	int rowPos=0;//position into row
	int frontIndex=intMax;//Means that the row ended
	if(rowPos<rowLength){
		frontIndex=*(rowIndices+rowPos);		
		rowPos++;
	}
	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
			//load next
			if(rowPos<rowLength){				
				frontIndex=*(rowIndices+rowPos);
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		dstPos++;
	}
	return dstPos;
}

template<int WarpSize, typename T>
static __device__ int MulWarpPredictSize_2(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n){
	if(nnza==0)
	{
		return nnzd;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread	

	int t=(threadIdx.x+1)*2;
 
	if(t<=nnza)
	{
		bgetrow(brows,acols[threadIdx.x*2],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*2+1],rowValues1,rowIndices1,rowLength1);
	}
	else if(t-1==nnza)
	{
		bgetrow(brows,acols[threadIdx.x*2],rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
	}

	if(blockDim.x-1==threadIdx.x){
		rowValues1=dvals;
		rowIndices1=dcols;
		rowLength1=nnzd;
	}

	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int frontIndex=intMax;//Means that the row ended

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
		rowPos0++;
	}
	else if(index0>index1)
	{
		frontIndex=index1;
		rowPos1++;
	}
	else
	{
		if(index0!=intMax)
		{
			frontIndex=index0;
			rowPos0++;
			rowPos1++;
		}
		else
		{
		}
	}

	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
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
				rowPos0++;
			}
			else if(index0>index1)
			{
				frontIndex=index1;
				rowPos1++;
			}
			else
			{
				if(index0!=intMax)
				{
					frontIndex=index0;
					rowPos0++;
					rowPos1++;
				}
				else
				{
					frontIndex=intMax;
				}
			}

		}
		minFront=WarpMin<WarpSize>(frontIndex);
		dstPos++;
	}
	return dstPos;
}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, typename T>
static __device__ int MulWarpPredictSize_4(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n){
	if(nnza==0)
	{
		return nnzd;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread	
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread	
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread	

	int t=(threadIdx.x+1)*4;
	if(t<=nnza){
		bgetrow(brows,acols[threadIdx.x*4],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*4+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*4+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*4+3],rowValues3,rowIndices3,rowLength3);
	}
	else if(t-1==nnza){
		bgetrow(brows,acols[threadIdx.x*4],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*4+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*4+2],rowValues2,rowIndices2,rowLength2);
		rowLength3=0;
	}
	else if(t-2==nnza){
		bgetrow(brows,acols[threadIdx.x*4],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*4+1],rowValues1,rowIndices1,rowLength1);
		rowLength2=0;
		rowLength3=0;
	}
	else if(t-3==nnza){
		bgetrow(brows,acols[threadIdx.x*4],rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}
	else{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}

	if(blockDim.x-1==threadIdx.x){
		rowValues3=dvals;
		rowIndices3=dcols;
		rowLength3=nnzd;
	}
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row


	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;

	int frontIndex=intMax;//Means that the row ended

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
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;
//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0&&threadIdx.y==0)
//		{
//			printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//			printf("index0=%d,index1=%d,index2=%d,index3=%d\n",index0,index1,index2,index3);
//			printf("frontIndex=%d\n",frontIndex);
//			printf("minFront=%d\n",minFront);
//			printf("------------------------------------\n");
//		}
//	}
	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
			}
			else
			{
				frontIndex=intMax;
			}
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		dstPos++;

	}

	return dstPos;
}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, typename T>
static __device__ int MulWarpPredictSize_8(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n){
	if(nnza==0)
	{
		return nnzd;
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

	int t=(threadIdx.x+1)*8;
	if(t<=nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*8+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*8+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*8+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*8+7],rowValues7,rowIndices7,rowLength7);
	}
	else if(t-1==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*8+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*8+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*8+6],rowValues6,rowIndices6,rowLength6);
		rowLength7=0;
	}
	else if(t-2==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*8+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*8+5],rowValues5,rowIndices5,rowLength5);
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-3==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*8+4],rowValues4,rowIndices4,rowLength4);
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-4==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-5==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-6==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-7==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else{
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
		rowValues7=dvals;
		rowIndices7=dcols;
		rowLength7=nnzd;
	}
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row


	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;

	int frontIndex=intMax;//Means that the row ended

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
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;
//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0&&threadIdx.y==0)
//		{
//			printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//			printf("index0=%d,index1=%d,index2=%d,index3=%d\n",index0,index1,index2,index3);
//			printf("frontIndex=%d\n",frontIndex);
//			printf("minFront=%d\n",minFront);
//			printf("------------------------------------\n");
//		}
//	}
	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
			}
			else
			{
				frontIndex=intMax;
			}
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		dstPos++;

	}

	return dstPos;
}
//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, typename T>
static __device__ int MulWarpPredictSize_16(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n){
	if(nnza==0)
	{
		return nnzd;
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

	int t=(threadIdx.x+1)*16;
	if(t<=nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		bgetrow(brows,acols[threadIdx.x*16+12],rowValues12,rowIndices12,rowLength12);
		bgetrow(brows,acols[threadIdx.x*16+13],rowValues13,rowIndices13,rowLength13);
		bgetrow(brows,acols[threadIdx.x*16+14],rowValues14,rowIndices14,rowLength14);
		bgetrow(brows,acols[threadIdx.x*16+15],rowValues15,rowIndices15,rowLength15);
	}
	else if(t-1==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		bgetrow(brows,acols[threadIdx.x*16+12],rowValues12,rowIndices12,rowLength12);
		bgetrow(brows,acols[threadIdx.x*16+13],rowValues13,rowIndices13,rowLength13);
		bgetrow(brows,acols[threadIdx.x*16+14],rowValues14,rowIndices14,rowLength14);
		rowLength15=0;
	}
	else if(t-2==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		bgetrow(brows,acols[threadIdx.x*16+12],rowValues12,rowIndices12,rowLength12);
		bgetrow(brows,acols[threadIdx.x*16+13],rowValues13,rowIndices13,rowLength13);
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-3==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		bgetrow(brows,acols[threadIdx.x*16+12],rowValues12,rowIndices12,rowLength12);
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-4==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-5==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-6==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-7==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-8==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-9==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
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
	else if(t-10==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
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
	else if(t-11==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
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
	else if(t-12==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
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
	else if(t-13==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
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
	else if(t-14==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
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
	else if(t-15==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
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
	else{
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
		rowValues15=dvals;
		rowIndices15=dcols;
		rowLength15=nnzd;
	}
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row
	int rowPos8=0;//position into row
	int rowPos9=0;//position into row
	int rowPos10=0;//position into row
	int rowPos11=0;//position into row
	int rowPos12=0;//position into row
	int rowPos13=0;//position into row
	int rowPos14=0;//position into row
	int rowPos15=0;//position into row


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

	int frontIndex=intMax;//Means that the row ended

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



	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;
//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0&&threadIdx.y==0)
//		{
//			printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//			printf("index0=%d,index1=%d,index2=%d,index3=%d\n",index0,index1,index2,index3);
//			printf("frontIndex=%d\n",frontIndex);
//			printf("minFront=%d\n",minFront);
//			printf("------------------------------------\n");
//		}
//	}
	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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
				//	frontIndex=index0;
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
		minFront=WarpMin<WarpSize>(frontIndex);

		dstPos++;

	}

	return dstPos;
}

//(int *, T *, int *, int *, T *, int *, int *, int, int, int, int *, int *, int)
template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_1(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulWarpPredictSize<WarpSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n);
	//int *data = crows;
	if(threadIdx.x==0)
	{
		crows[r] = dstLength;
	}

}

	template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_2(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;
	int dstLength=MulWarpPredictSize_2<WarpSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n);
	int *data = crows; 
	if(threadIdx.x==0) 
	{ 
		data[r] = dstLength;
	}

}
	template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_4(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulWarpPredictSize_4<WarpSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n);
	int *data = crows;
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}

}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_8(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulWarpPredictSize_8<WarpSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n);
	int *data = crows;
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}

}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_16(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulWarpPredictSize_16<WarpSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n);
	int *data = crows;
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}

}


template<int WarpSize, int SegmentSize, typename T>
static __device__ int MulOverWarpPredictSize_1(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n,int*temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(nnza==0)
	{
		return nnzd;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues;int* rowIndices;int rowLength=0;//The row for the thread	
	if(threadIdx.x<nnza)
		bgetrow(brows,acols[threadIdx.x],rowValues,rowIndices,rowLength);

	if(blockDim.x-1==threadIdx.x){
		rowValues=dvals;
		rowIndices=dcols;
		rowLength=nnzd;
	}

	int rowPos=0;//position into row
	int frontIndex=intMax;//Means that the row ended
	if(rowPos<rowLength){
		frontIndex=*(rowIndices+rowPos);		
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);	

	if(laneId==0)
	{
		temp[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
			//load next
			if(rowPos<rowLength){				
				frontIndex=*(rowIndices+rowPos);
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;
	}
	return dstPos;

}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ int MulOverWarpPredictSize_2(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n,int*temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(nnza==0)
	{
		return nnzd;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread	

	int t=(threadIdx.x+1)*2;
	
	if(t<=nnza)
	{
		bgetrow(brows,acols[threadIdx.x*2],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*2+1],rowValues1,rowIndices1,rowLength1);
	}
	else if(t-1==nnza)
	{
		bgetrow(brows,acols[threadIdx.x*2],rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
	}

	if(blockDim.x-1==threadIdx.x){
		rowValues1=dvals;
		rowIndices1=dcols;
		rowLength1=nnzd;
	}


	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int frontIndex=intMax;//Means that the row ended

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
		rowPos0++;
	}
	else if(index0>index1)
	{
		frontIndex=index1;
		rowPos1++;
	}
	else
	{
		if(index0!=intMax)
		{
			frontIndex=index0;
			rowPos0++;
			rowPos1++;
		}
		else
		{
		}
	}

	int minFront=WarpMin<WarpSize>(frontIndex);	

	if(laneId==0)
	{
		temp[warpId] = minFront;
	}
	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;
	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
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
				rowPos0++;
			}
			else if(index0>index1)
			{
				frontIndex=index1;
				rowPos1++;
			}
			else
			{
				if(index0!=intMax)
				{
					frontIndex=index0;
					rowPos0++;
					rowPos1++;
				}
				else
				{
					frontIndex=intMax;
				}
			}

		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;
	}
	return dstPos;
}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ int MulOverWarpPredictSize_4(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n,int*temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(nnza==0)
	{
		return nnzd;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;int* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;int* rowIndices1;int rowLength1=0;//The row for the thread	
	T* rowValues2;int* rowIndices2;int rowLength2=0;//The row for the thread	
	T* rowValues3;int* rowIndices3;int rowLength3=0;//The row for the thread	

	int t=(threadIdx.x+1)*4;
	if(t<=nnza){
		bgetrow(brows,acols[threadIdx.x*4],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*4+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*4+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*4+3],rowValues3,rowIndices3,rowLength3);
	}
	else if(t-1==nnza){
		bgetrow(brows,acols[threadIdx.x*4],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*4+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*4+2],rowValues2,rowIndices2,rowLength2);
		rowLength3=0;
	}
	else if(t-2==nnza){
		bgetrow(brows,acols[threadIdx.x*4],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*4+1],rowValues1,rowIndices1,rowLength1);
		rowLength2=0;
		rowLength3=0;
	}
	else if(t-3==nnza){
		bgetrow(brows,acols[threadIdx.x*4],rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}
	else{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}

	if(blockDim.x-1==threadIdx.x){
		rowValues3=dvals;
		rowIndices3=dcols;
		rowLength3=nnzd;
	}
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row


	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;

	int frontIndex=intMax;//Means that the row ended

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
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	

	if(laneId==0)
	{
		temp[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;
	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
			}
			else
			{
				frontIndex=intMax;
			}
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;
		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;

	}

	return dstPos;

}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ int MulOverWarpPredictSize_8(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n,int*temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(nnza==0)
	{
		return nnzd;
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

	int t=(threadIdx.x+1)*8;
	if(t<=nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*8+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*8+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*8+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*8+7],rowValues7,rowIndices7,rowLength7);
	}
	else if(t-1==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*8+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*8+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*8+6],rowValues6,rowIndices6,rowLength6);
		rowLength7=0;
	}
	else if(t-2==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*8+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*8+5],rowValues5,rowIndices5,rowLength5);
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-3==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*8+4],rowValues4,rowIndices4,rowLength4);
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-4==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*8+3],rowValues3,rowIndices3,rowLength3);
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-5==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*8+2],rowValues2,rowIndices2,rowLength2);
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-6==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*8+1],rowValues1,rowIndices1,rowLength1);
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-7==nnza){
		bgetrow(brows,acols[threadIdx.x*8],rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else{
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
		rowValues7=dvals;
		rowIndices7=dcols;
		rowLength7=nnzd;
	}

	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row


	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;

	int frontIndex=intMax;//Means that the row ended

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
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	
	if(laneId==0)
	{
		temp[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
			}
			else
			{
				frontIndex=intMax;
			}
		}

		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;

	}

	return dstPos;

}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ int MulOverWarpPredictSize_16(
    T*avals,int*arows,int*acols,int nnza,\
    T*bvals,int*brows,int*bcols,\
	T*dvals,int*drows,int*dcols,int nnzd,\
    int m,int k,int n,int*temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(nnza==0)
	{
		return nnzd;
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

	int t=(threadIdx.x+1)*16;
	if(t<=nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		bgetrow(brows,acols[threadIdx.x*16+12],rowValues12,rowIndices12,rowLength12);
		bgetrow(brows,acols[threadIdx.x*16+13],rowValues13,rowIndices13,rowLength13);
		bgetrow(brows,acols[threadIdx.x*16+14],rowValues14,rowIndices14,rowLength14);
		bgetrow(brows,acols[threadIdx.x*16+15],rowValues15,rowIndices15,rowLength15);
	}
	else if(t-1==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		bgetrow(brows,acols[threadIdx.x*16+12],rowValues12,rowIndices12,rowLength12);
		bgetrow(brows,acols[threadIdx.x*16+13],rowValues13,rowIndices13,rowLength13);
		bgetrow(brows,acols[threadIdx.x*16+14],rowValues14,rowIndices14,rowLength14);
		rowLength15=0;
	}
	else if(t-2==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		bgetrow(brows,acols[threadIdx.x*16+12],rowValues12,rowIndices12,rowLength12);
		bgetrow(brows,acols[threadIdx.x*16+13],rowValues13,rowIndices13,rowLength13);
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-3==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		bgetrow(brows,acols[threadIdx.x*16+12],rowValues12,rowIndices12,rowLength12);
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-4==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		bgetrow(brows,acols[threadIdx.x*16+11],rowValues11,rowIndices11,rowLength11);
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-5==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		bgetrow(brows,acols[threadIdx.x*16+10],rowValues10,rowIndices10,rowLength10);
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-6==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		bgetrow(brows,acols[threadIdx.x*16+9],rowValues9,rowIndices9,rowLength9);
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-7==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		bgetrow(brows,acols[threadIdx.x*16+8],rowValues8,rowIndices8,rowLength8);
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-8==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
		bgetrow(brows,acols[threadIdx.x*16+7],rowValues7,rowIndices7,rowLength7);
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-9==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
		bgetrow(brows,acols[threadIdx.x*16+6],rowValues6,rowIndices6,rowLength6);
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
	else if(t-10==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
		bgetrow(brows,acols[threadIdx.x*16+5],rowValues5,rowIndices5,rowLength5);
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
	else if(t-11==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
		bgetrow(brows,acols[threadIdx.x*16+4],rowValues4,rowIndices4,rowLength4);
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
	else if(t-12==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
		bgetrow(brows,acols[threadIdx.x*16+3],rowValues3,rowIndices3,rowLength3);
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
	else if(t-13==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
		bgetrow(brows,acols[threadIdx.x*16+2],rowValues2,rowIndices2,rowLength2);
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
	else if(t-14==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
		bgetrow(brows,acols[threadIdx.x*16+1],rowValues1,rowIndices1,rowLength1);
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
	else if(t-15==nnza){
		bgetrow(brows,acols[threadIdx.x*16],rowValues0,rowIndices0,rowLength0);
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
	else{
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
		rowValues15=dvals;
		rowIndices15=dcols;
		rowLength15=nnzd;
	}

	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row
	int rowPos8=0;//position into row
	int rowPos9=0;//position into row
	int rowPos10=0;//position into row
	int rowPos11=0;//position into row
	int rowPos12=0;//position into row
	int rowPos13=0;//position into row
	int rowPos14=0;//position into row
	int rowPos15=0;//position into row


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

	int frontIndex=intMax;//Means that the row ended

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



	int minFront=WarpMin<WarpSize>(frontIndex);	

	if(laneId==0)
	{
		temp[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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
				//	frontIndex=index0;
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
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;

	}

	return dstPos;
}


template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_1(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ int temp[]; 
    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulOverWarpPredictSize_1<WarpSize,SegmentSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n,temp);

	
	int*data=crows;
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_2(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ int temp[]; 
    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulOverWarpPredictSize_2<WarpSize,SegmentSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n,temp);

	
	int*data=crows;
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_4(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ int temp[]; 
    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulOverWarpPredictSize_4<WarpSize,SegmentSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n,temp);

	
	int*data=crows;
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_8(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{
	
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ int temp[]; 
    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulOverWarpPredictSize_8<WarpSize,SegmentSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n,temp);

	
	int*data=crows;
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}


template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_16(int*crows, \
T*avals,int*arows,int*acols, \
T*bvals,int*brows,int*bcols, \
T*dvals,int*drows,int*dcols,\
int m,int k,int n,\
int*Queue, int*Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ int temp[]; 
    int ptr_arowr=arows[r];
    int nnza=arows[r+1]-ptr_arowr;

	int ptr_drowr=drows[r];
	int nnzd=drows[r+1]-ptr_drowr;

	int dstLength=MulOverWarpPredictSize_16<WarpSize,SegmentSize>(avals+ptr_arowr,arows+r,acols+ptr_arowr,nnza,\
    bvals,brows,bcols,dvals+ptr_drowr,drows+r,dcols+ptr_drowr,nnzd,m,k,n,temp);

	
	int*data=crows;
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

static inline int DivUp(int a,int b){
    return (a+b-1)/b;
}

#include<iostream>

extern hipStream_t stream[13];

template<typename T>
void __cdecl PredictCSize(int*crows,\
T*avals,int*arows,int*acols,\
T*bvals,int*brows,int*bcols,\
T*dvals,int*drows,int*dcols,\
T alpha,\
int m,int k,int n,\
int* Queue, \
int* Queue_one, \
int* h_queue_one)
{
	int threadnum = 512;
	//record_time("predict_inner_start");
	/*hipStream_t stream[13];
	for(int i=0; i<13; i++)
	{
		hipStreamCreate(&stream[i]);
	}*/

	int count;
	//record_time("loop_start");
	for(int i=0; i<13; i++)
	{
		count = h_queue_one[i+1] - h_queue_one[i];
        
		//hipDeviceSynchronize();
		//record_time("last_loop");
		//std::cout<<"last finished next="<<i<<" "<<count;
        	//record_time("next loop");
		if(count==0)
			continue;

		if(i==0) //rowLength<=2
		{
			dim3 blockDim(2,threadnum/2,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//(int *, T *, int *, int *, T *, int *, int *, int, int, int, int *, int *, int)
			//DifSpmmPredictSizeWarpKernel_1<2> <<< gridDim, blockDim, 0, stream[0]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmPredictSizeWarpKernel_1<2>,gridDim, blockDim, 0, stream[0],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==1) //2<rowLength<=4
		{
			dim3 blockDim(4,threadnum/4,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeWarpKernel_1<4> <<< gridDim, blockDim, 0, stream[1]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmPredictSizeWarpKernel_1<4>,gridDim, blockDim, 0, stream[1],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==2) //5<rowLength<=8
		{
			dim3 blockDim(8,threadnum/8,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeWarpKernel_1<8> <<< gridDim, blockDim, 0, stream[2]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmPredictSizeWarpKernel_1<8>,gridDim, blockDim, 0, stream[2],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==3) //9<rowLength<=16
		{
			dim3 blockDim(16,threadnum/16,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeWarpKernel_1<16> <<< gridDim, blockDim, 0, stream[3]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmPredictSizeWarpKernel_1<16>,gridDim, blockDim, 0, stream[3],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==4) //16<rowLength<=32
		{
			dim3 blockDim(32,threadnum/32,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeWarpKernel_1<32> <<< gridDim, blockDim, 0, stream[4]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmPredictSizeWarpKernel_1<32>,gridDim, blockDim, 0, stream[4],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==5) //32<rowLeng<=64
		{
			dim3 blockDim(64,threadnum/64,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeWarpKernel_2<32> <<< gridDim, blockDim, 0, stream[5]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmPredictSizeWarpKernel_1<64>,gridDim, blockDim, 0, stream[5],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
//			DifSpmmPredictSizeOverWarpKernel_1<32,2> <<<gridDim, blockDim, threadnum/32, stream[9]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==6) //64<rowLength<=128
		{
			dim3 blockDim(64,threadnum/64,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeWarpKernel_4<32> <<< gridDim, blockDim, 0, stream[6]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmPredictSizeWarpKernel_2<64>,gridDim, blockDim, 0, stream[6],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
//			DifSpmmPredictSizeOverWarpKernel_2<32,2> <<<gridDim, blockDim, threadnum/32, stream[9]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==7) //128<rowLength<=256
		{
			dim3 blockDim(64,4,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
//			DifSpmmPredictSizeOverWarpKernel_4<32,2> <<<gridDim, blockDim, threadnum/32, stream[9]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			//DifSpmmPredictSizeWarpKernel_8<32> <<< gridDim, blockDim, 0, stream[7]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL(DifSpmmPredictSizeWarpKernel_4<64>,gridDim, blockDim, 0, stream[7],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==8) //256<rowLength<=512
		{
			dim3 blockDim(512,1,1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
//			DifSpmmPredictSizeWarpKernel_16<32> <<< gridDim, blockDim, 0, stream[8]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			//DifSpmmPredictSizeOverWarpKernel_1<64,8> <<<gridDim, blockDim, 8, stream[8]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmPredictSizeOverWarpKernel_1<64,8>),gridDim, blockDim, 8, stream[8],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==9) //512<rowLength<=1024
		{
			dim3 blockDim(512, 1, 1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeOverWarpKernel_2<64,8> <<<gridDim, blockDim, 8, stream[9]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmPredictSizeOverWarpKernel_2<64,8>) ,gridDim, blockDim, 0, stream[9],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==10) //1024<rowLength<=2048
		{
			dim3 blockDim(512, 1, 1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeOverWarpKernel_4<64,8> <<<gridDim, blockDim, 8, stream[10]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmPredictSizeOverWarpKernel_4<64,8>),gridDim, blockDim, 0, stream[10],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else if(i==11) //2048<rowLength<=4096
		{
			dim3 blockDim(512, 1, 1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeOverWarpKernel_8<64,8> <<<gridDim, blockDim, 8, stream[11]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmPredictSizeOverWarpKernel_8<64,8>),gridDim, blockDim, 0, stream[11],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
		else //if(i==12) //rowLength>4096
		{
			dim3 blockDim(512,1, 1);
			dim3 gridDim(DivUp(count,(int)blockDim.y),1,1);
			//DifSpmmPredictSizeOverWarpKernel_16<64,8> <<<gridDim, blockDim, 8, stream[12]>>>(crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
			hipLaunchKernelGGL((DifSpmmPredictSizeOverWarpKernel_16<64,8>),gridDim, blockDim, 0, stream[12],\
			crows,\
                avals,arows,acols,\
                bvals,brows,bcols,\
				dvals,drows,dcols,\
                m,k,n,\
                Queue,Queue_one,i);
		}
/*
	hipDeviceSynchronize();
	hipError err = hipGetLastError();
	if(err!=hipSuccess){
		std::cout<<"hip_error:"<<hipGetErrorString( err )<<"\n";
	}
	std::cout<<"location="<<i;getchar();
*/
	}


	hipDeviceSynchronize();
//	record_time("loop_end");
/*	for(int i=0; i<13; i++)
	{
		hipStreamDestroy(stream[i]);
	}
*/
//	record_time("destory_stream");
}
