#pragma once
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <sys/time.h>
double solve_l_time;
double solve_u_time;
using namespace std;
/*******************init u by a, A stored all elements**************/
template<typename T>
void initLByA(int n, int *ptr, T *val, int *col, int *lPtr, T *lVal, int *lCol, T *tnorm, T droptol);
/*****************ic0 factorization*******************/
template<typename T>
void ic0_fact(int n,int *lPtr, T *lVal, int *lCol, T *tnorm, T smallNum, T droptol);
template<typename T>
void ic0_csr_A(int n, int *ptr, T *val, int *col, int *lPtr, T *lVal, int *lCol,T smallNum, T droptol);
template<typename T>
void solver_ic0(int n, int *lPtr, T *lVal, int *lCol, T *y, T *x);
