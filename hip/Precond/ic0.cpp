#include "ic0.h"
double solve_l_time;
double solve_u_time;
template<typename T>
void initLByA(int n, int *ptr, T *val, int *col, int *lPtr, T *lVal, int *lCol, T *tnorm, T droptol){
	int len = 0;
        lPtr[0] = 0;
        for (int i = 0; i < n; i++) {
                for (int j = ptr[i]; j < ptr[i+1]; j++) {
                        if(col[j]<=i){
                                lVal[len] = val[j];
                                lCol[len] = col[j];
                                len++;
                        }
			tnorm[i] += val[j]*val[j];
                }
                lPtr[i+1] = len;
                tnorm[i] = sqrt(tnorm[i]);
                tnorm[i] *= droptol;
                if(lCol[lPtr[i+1]-1] != i){
                        cout<<"The row "<<i<<" diagonal element of A is zone!"<<endl;
                        exit(0);
                }
        }
}
/*****************ic0 factorization*******************/
template<typename T>
void ic0_fact(int n,int *lPtr, T *lVal, int *lCol, T *tnorm, T smallNum, T droptol){
	int idxu=0;
	int diagl=0;
	int diagu=0;
	lVal[0] = sqrt(lVal[0]);
       	if(fabs(lVal[0])<1e-20){
               	lVal[0] = smallNum/droptol*tnorm[0] + tnorm[0];
       	}
	lVal[0]=1/lVal[0];
        for(int i=1; i<n; i++){
		diagl = lPtr[i+1]-1; 
		for(int j=lPtr[i]; j<diagl; j++){//要计算的数
			idxu = lPtr[lCol[j]];
			diagu = lPtr[lCol[j]+1]-1;
			for(int k=lPtr[i]; k<j; k++){//L上的数
				for(int p=idxu; p<diagu; p++){//U上的数
					if(lCol[p] == lCol[k]){
						lVal[j] -= lVal[k]*lVal[p];
					}else if(lCol[p] > lCol[k]){
						idxu=p;
						break;
					}
				}
			}
			lVal[j] *= lVal[diagu];
		}
		for(int j=lPtr[i]; j<diagl; j++){//L上的数
			lVal[diagl] -= lVal[j]*lVal[j];
		}
		lVal[diagl] = sqrt(lVal[diagl]);
        	if(fabs(lVal[diagl])<1e-20){
                	lVal[diagl] = smallNum/droptol*tnorm[i] + tnorm[i];
        	}
		lVal[diagl]=1/lVal[diagl];
        }
}
template<typename T>
void ic0_csr_A(int n, int *ptr, T *val, int *col, int *lPtr, T *lVal, int *lCol,T smallNum, T droptol){
	//struct timeval t1, t2,t3;
	T *tnorm = new T[n]();
	//gettimeofday(&t1, NULL);
	initLByA(n, ptr, val, col, lPtr, lVal, lCol, tnorm, droptol);
	//gettimeofday(&t2, NULL);
	ic0_fact(n,lPtr, lVal, lCol, tnorm, smallNum, droptol);
	//gettimeofday(&t3, NULL);
        //cout<<"init_time="<<(t2.tv_sec - t1.tv_sec) * 1000. +(t2.tv_usec - t1.tv_usec) / 1000.<<endl;
        //cout<<"ic0_time="<<(t3.tv_sec - t2.tv_sec) * 1000. +(t3.tv_usec - t2.tv_usec) / 1000.<<endl;
        delete[]tnorm;
}
template<typename T>
void solver_ic0(int n, int *lPtr, T *lVal, int *lCol, T *y, T *x){
	T *tempx = new T[n];
	T *tempy = new T[n];
	struct timeval t1, t2, t3;
	for(int i=0; i<n; i++){
		tempy[i] = y[i];
	}
	gettimeofday(&t1, NULL);
	for(int i=0; i<n; i++){
		for(int j=lPtr[i]; j<lPtr[i+1]-1; j++){
			tempy[i] -= tempx[lCol[j]]*lVal[j];
		}
		tempx[i] = tempy[i]*lVal[lPtr[i+1]-1];
	}
	gettimeofday(&t2, NULL);
	for(int i=n-1; i>=0; i--){
		x[i] = tempx[i]*lVal[lPtr[i+1]-1];
		for(int j=lPtr[i]; j<lPtr[i+1]-1; j++){
			tempx[lCol[j]] -= lVal[j]*x[i];
		}
	}
	gettimeofday(&t3, NULL);
	solve_l_time += (t2.tv_sec - t1.tv_sec) * 1000. +(t2.tv_usec - t1.tv_usec) / 1000.;	
	solve_u_time += (t3.tv_sec - t2.tv_sec) * 1000. +(t3.tv_usec - t2.tv_usec) / 1000.;
	delete []tempx;
	delete []tempy;	
}
template void ic0_csr_A(int n,int *ptr, double *val, int *col,int *lPtr, double *lVal, int *lCol, double smallNum, double droptol);
template void ic0_csr_A(int n,int *ptr, float *val, int *col,int *lPtr, float *lVal, int *lCol, float smallNum, float droptol);
