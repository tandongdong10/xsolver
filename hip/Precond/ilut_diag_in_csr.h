#pragma once
#include <vector>
#include <algorithm>
#include <functional>
#include <math.h>
#include <string.h>
#define small_num 1e-4
template<typename D>
struct cmp_id {

	cmp_id(D*v) :value(v) {
	}
	D*value;
	bool operator()(int id1, int id2) {
		return fabs(value[id1]) > fabs(value[id2]);
	}
};
template<typename T>
void heap_swap(int*cols, int len, T*val);
template<typename T>
void ilutp_final(T*luval, T*val, int*rows, int*cols, int n, T*diag, int lfil, int*lucols, int*lurows, int*uptr, T permtol, T droptol, int*iperm, int*ipermn);

template<typename T>
void lusolve_no_pivot_no_diag_csr(T*luvalue, int*rows, int*cols, T*diag, int*uptr, T*x, int n, int*ipermn, T*y);
template<typename T>
void ilu0_simple(T*luval, T*val, int*rows, int*cols, int n, T*diag, int lfil, int*lucols, int*lurows, int*uptr, T permtol, T droptol, int*iperm, int*ipermn,\
int sweep=1);
template<typename T>
void ilu0_simple(T*luval, T*val, int*rows, int*cols, int n, int*uptr);

template<typename T>
void iluk_simple(T*luval, T*val, int*rows, int*cols, int n, int*lucols, int*lurows, int*uptr);

template<typename T>
void ilu0_diag_is_ptr(T*luval, T*val, int*rows, int*cols, int n, T*diag, int lfil, int*lucols, int*lurows, int*uptr, T permtol, T droptol, int*iperm, int*ipermn,\
int sweep=1);

template<typename T>
void ilu0_diag_is_ptr_uptr_input_simplified(T*luval, T*val, int*rows, int*cols, int n, T*diag, int lfil, int*lucols, int*lurows, int*uptr, T permtol, T droptol, int*iperm, int*ipermn, \
	int sweep = 1);

template<typename T>
void lusolve_for_ilu0(T*luvalue, int*rows, int*cols, T*diag, int*uptr, T*x, int n, int*ipermn, T*y);

