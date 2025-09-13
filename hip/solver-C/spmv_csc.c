#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h> 


void spmv_csc(const int rows,const int cols,const int *ptr,const double*value,const int*row_idx,const double *x,double*y){
    int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {    
        num_threads = omp_get_num_threads();
    } 
    double* temp_y=(double *) malloc (sizeof(double) * (cols*num_threads));
    memset(temp_y,0,sizeof(double)*num_threads*cols);
    
    #pragma omp parallel for num_threads(num_threads) 
    for(int i=0;i<cols;i++){
       double colx=x[i];
       int myid_mul_cols=omp_get_thread_num()*rows;
       int end=ptr[i+1];
       int j;
       for(j=ptr[i];j<end-4;j+=4){
           int j1=j+1,j2=j+2,j3=j+3;//j4=j+4,j5=j+5,j6=j+6,j7=j+7;
           temp_y[row_idx[j]+myid_mul_cols]+=colx*value[j];
           temp_y[row_idx[j1]+myid_mul_cols]+=colx*value[j1];
           temp_y[row_idx[j2]+myid_mul_cols]+=colx*value[j2];
           temp_y[row_idx[j3]+myid_mul_cols]+=colx*value[j3];
/*
           temp_y[row_idx[j4]+myid_mul_cols]+=colx*value[j4];
           temp_y[row_idx[j5]+myid_mul_cols]+=colx*value[j5];
           temp_y[row_idx[j6]+myid_mul_cols]+=colx*value[j6];
           temp_y[row_idx[j5]+myid_mul_cols]+=colx*value[j7];
           */
       }
       for(;j < end;j++){
           temp_y[row_idx[j]+myid_mul_cols]+=colx*value[j];
       }
    }
    
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0;i<rows;i++){
        double sum=0.0;
        for(int j=0;j<num_threads;j++){
        sum+=temp_y[i+j*rows];
        }
        y[i]=sum;
    }
    free(temp_y);
}
