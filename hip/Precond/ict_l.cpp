#include "ict_l.h"
/***************heapSort****************/
template<typename T>
void adjustHeap_l(T *val1, int *val2, int i,int maxNum){
        T tempVal1 = val1[i];
        int tempVal2 = val2[i];
        for(int k=2*i+1; k<maxNum; k=k*2+1){
                if(k+1<maxNum && fabs(val1[k])>fabs(val1[k+1])){
                        k++;
                }
                if(fabs(val1[k]) < fabs(tempVal1)){
                        val1[i] = val1[k];
                        val2[i] = val2[k];
                        i = k;
                }
                else{
                        break;
                }
        }
        val1[i] = tempVal1;
        val2[i] = tempVal2;
}
template<typename T>
void heapSort_part_l(T *val1, int *val2, int maxNum, int n){
        if(maxNum>n){
                cout<<"The maxNum is bigger than n in function heapSort!"<<endl;
                exit(0);
        }
        for(int i=maxNum/2; i>=0; i--){
                adjustHeap_l(val1,val2,i,maxNum);
        }
        for(int i=maxNum; i<n; i++){
                if(fabs(val1[i])>fabs(val1[0])){
                        val1[0]=val1[i];
                        val2[0]=val2[i];
                        adjustHeap_l(val1,val2,0,maxNum);
                }
        }
}
template<typename T>
void swap_all(T *arr, int a, int b){
        T temp;
        temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
}
template<typename T>
void adjustHeap_all(int *val1, T *val2, int i,int maxNum){
        int tempVal1 = val1[i];
        T tempVal2 = val2[i];
        for(int k=2*i+1; k<maxNum; k=k*2+1){
                if(k+1<maxNum && val1[k]<val1[k+1]){
                        k++;
                }
                if(val1[k] > tempVal1){
                        val1[i] = val1[k];
                        val2[i] = val2[k];
                        i = k;
                }
                else{
                        break;
                }
        }
        val1[i] = tempVal1;
        val2[i] = tempVal2;
}
template<typename T>
void heapSort_all_l(int *val1, T *val2,int maxNum){
        for(int i=maxNum/2; i>=0; i--){
                adjustHeap_all(val1,val2,i,maxNum);
        }
        for(int i=maxNum-1; i>=0; i--){
                swap_all(val1,0,i);
                swap_all(val2,0,i);
                adjustHeap_all(val1,val2,0,i);
        }
}
/*****************quikSort, ascending order******************/
template<typename T>
void asQuikSort(int *val1, T *val2, int start, int end){
	if(start>end) return;
	int i = start;
	int j = end;
	int temp1;
	T temp2;
	int key = val1[i];
	while(i != j){
		while(j>i && val1[j]>=key)
			j--;
		while(j>i && val1[i]<=key)
			i++;
		if(j > i){
			temp1 = val1[i];
			val1[i] = val1[j];
			val1[j] = temp1;
			temp2 = val2[i];
			val2[i] = val2[j];
			val2[j] = temp2;
		}
	}
	val1[start] = val1[i];
	val1[i] = key;
	temp2 = val2[start];
	val2[start] = val2[i];
	val2[i] = temp2;
	asQuikSort(val1,val2,start,i-1);
	asQuikSort(val1,val2,i+1,end);
}
/*******************init u by a, A stored all elements**************/
template<typename T>
void calTnorm(int n, int *ptr, T *val, int *col, T *tnorm, T droptol){
	bool flag = false;
        for (int i = 0; i < n; i++) {
		flag = false;
		for (int j = ptr[i]; j < ptr[i+1]; j++) {
                        tnorm[i] += val[j]*val[j];
                        if(col[j] == i){
                                flag = true;
                        }
                }
                if(flag == false){
                        cout<<"The row "<<i<<" diagonal element of A is zone!"<<endl;
                        exit(0);
                }
                tnorm[i] = sqrt(tnorm[i]);
                tnorm[i] *= droptol;
        }
}
/*****************ic0 factorization*******************/
template<typename T>
void ict_fact_l(int n,int *ptr, T *val, int *col,int *lPtr, T *lVal, int *lCol, T *tnorm, T smallNum, T droptol, int maxfil){
	struct timeval t1, t2;
	double calt=0.0,sortt=0.0;
	int diagu=0;
	T *working_val = new T[n];
        T *tempVal = new T[n];
        int *tempCol = new int[n];
	int count = 0;
	lPtr[0] = 0;
	lVal[0] = val[0];
	lCol[0]=0;
       	if(fabs(lVal[0])<1e-20){
               	lVal[0] = smallNum*(1/droptol)*tnorm[0] + tnorm[0];
       	}
	lVal[0] = 1/sqrt(fabs(lVal[0]));
	lPtr[1] = 1;
        for(int i=1; i<n; i++){
		count = 0;
			//gettimeofday(&t1, NULL);			
		for(int j=lCol[lPtr[col[ptr[i]]]]; j<=i; j++){
			working_val[j] = 0.0;
		}
		for(int j=ptr[i]; j<ptr[i+1]; j++){
			working_val[col[j]] = val[j];
		}
		for(int j=col[ptr[i]]; j<i; j++){//要计算的数
			diagu = lPtr[j+1]-1;
			double ttt=0.0;
			for(int p=lPtr[j]; p<lPtr[j+1]-1; p++){//U上的数
				ttt += working_val[lCol[p]]*lVal[p];
                        }
			working_val[j] -= ttt;
			working_val[j] *= lVal[diagu];
			if(fabs(working_val[j])>tnorm[i]){
				tempVal[count] = working_val[j];
				tempCol[count] = j;
				count++;
				working_val[i] -= working_val[j]*working_val[j];
			}
		}
		//gettimeofday(&t2, NULL);
		//calt += (t2.tv_sec - t1.tv_sec) * 1000. +(t2.tv_usec - t1.tv_usec) / 1000.;
        	if(fabs(working_val[i])<1e-20){
                	working_val[i] = smallNum*(1/droptol)*tnorm[i] + tnorm[i];
        	}
		working_val[i] = 1/sqrt(fabs(working_val[i]));
	        //gettimeofday(&t1, NULL);			
		if(count>maxfil){
			heapSort_part_l(tempVal, tempCol, maxfil, count);
			asQuikSort(tempCol,tempVal,0,maxfil-1);
			int startId = lPtr[i];
                        for(int j=0; j<maxfil; j++){
                                lVal[startId] = tempVal[j];
                                lCol[startId] = tempCol[j];
                                startId++;
                        }
                        lVal[startId] = working_val[i];
                        lCol[startId] = i;
                        lPtr[i+1] = lPtr[i] + maxfil+1;
		}else{
			int startId = lPtr[i];
                        for(int j=0; j<count; j++){
                                lVal[startId] = tempVal[j];
                                lCol[startId] = tempCol[j];
                                startId++;
                        }
                        lVal[startId] = working_val[i];
                        lCol[startId] = i;
                        lPtr[i+1] = lPtr[i] + count + 1;
		}
		        //gettimeofday(&t2, NULL);
			//sortt += (t2.tv_sec - t1.tv_sec) * 1000. +(t2.tv_usec - t1.tv_usec) / 1000.;
        }
	//cout<<"cal_time="<<calt<<"ms"<<endl;
	//cout<<"sort_time="<<sortt<<"ms"<<endl;
	delete []working_val;
        delete []tempVal;
        delete []tempCol;
}
template<typename T>
void ict_csr_A_l(int n, int *ptr, T *val, int *col, int *lPtr, T *lVal, int *lCol,T smallNum, T droptol, int maxfil){
	T *tnorm = new T[n]();
	calTnorm(n, ptr, val, col, tnorm, droptol);
	ict_fact_l(n,ptr, val, col,lPtr, lVal, lCol, tnorm, smallNum, droptol, maxfil);
	delete []tnorm;
}
template<typename T>
void solver_ict_l(int n, int *lPtr, T *lVal, int *lCol, T *y, T *x){
	struct timeval t1, t2, t3;
	T *tempx = new T[n];
	T *tempy = new T[n];
	for(int i=0; i<n; i++){
		tempy[i] = y[i];
	}
	//gettimeofday(&t1, NULL);
	for(int i=0; i<n; i++){
		for(int j=lPtr[i]; j<lPtr[i+1]-1; j++){
			tempy[i] -= lVal[j]*tempx[lCol[j]];
		}
		tempx[i] = tempy[i]*lVal[lPtr[i+1]-1];
	}
	//gettimeofday(&t2, NULL);
	for(int i=n-1; i>=0; i--){
		x[i] = tempx[i]*lVal[lPtr[i+1]-1];
		for(int j=lPtr[i]; j<lPtr[i+1]-1; j++){
			tempx[lCol[j]] -= lVal[j]*x[i];
		}
	}
	//gettimeofday(&t3, NULL);
	//cout<<"l_row_time="<<(t2.tv_sec - t1.tv_sec) * 1000. +(t2.tv_usec - t1.tv_usec) / 1000.<<endl;
	//cout<<"u_col_time="<<(t3.tv_sec - t2.tv_sec) * 1000. +(t3.tv_usec - t2.tv_usec) / 1000.<<endl;
	delete []tempx;
	delete []tempy;
}
template void ict_csr_A_l(int n,int *ptr, double *val, int *col,int *lPtr, double *lVal, int *lCol, double smallNum, double droptol, int maxfil);
template void ict_csr_A_l(int n,int *ptr, float *val, int *col,int *lPtr, float *lVal, int *lCol, float smallNum, float droptol, int maxfil);
