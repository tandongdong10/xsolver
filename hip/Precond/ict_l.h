#pragma once
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
using namespace std;
/***************heapSort****************/
template<typename T>
void adjustHeap_l(T *val1, int *val2, int i,int maxNum);
template<typename T>
void heapSort_part_l(T *val1, int *val2, int maxNum, int n);
template<typename T>
void swap_all(T *arr, int a, int b);
template<typename T>
void adjustHeap_all(int *val1, T *val2, int i,int maxNum);
template<typename T>
void heapSort_all_l(int *val1, T *val2,int maxNum);
/*****************quikSort, ascending order******************/
template<typename T>
void asQuikSort(int *val1, T *val2, int start, int end);
/*******************init u by a, A stored all elements**************/
template<typename T>
void calTnorm(int n, int *ptr, T *val, int *col, T *tnorm, T droptol);
/*****************ic0 factorization*******************/
template<typename T>
void ict_fact_l(int n,int *ptr, T *val, int *col,int *lPtr, T *lVal, int *lCol, T *tnorm, T smallNum, T droptol, int maxfil);
template<typename T>
void ict_csr_A_l(int n, int *ptr, T *val, int *col, int *lPtr, T *lVal, int *lCol,T smallNum, T droptol, int maxfil);
template<typename T>
void solver_ict_l(int n, int *lPtr, T *lVal, int *lCol, T *y, T *x);
