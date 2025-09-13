#include "ilut_diag_in_csr.h"
template<typename T>
void heap_swap(int*cols, int len, T*val) {
 int i = 0;
 int j = 1;
 int k = cols[0];
 T valk = fabs(val[k]);
 while (j < len) {
  if (j + 1 < len&&fabs(val[cols[j]]) > fabs(val[cols[j + 1]]))j = j + 1;

  if (fabs(val[cols[j]]) > valk)break;

  cols[i] = cols[j];
  i = j;
  j = 2 * j + 1;

 }
 cols[i] = k;
}
template<typename T>
void ilutp_final(T*luval, T*val, int*rows, int*cols, int n, T*diag, int lfil, int*lucols, int*lurows, int*uptr, T permtol, T droptol, int*iperm, int*ipermn) {
	T*working_array = new T[n];
	int*ujump = new int[n];
	int len_ujump = 0;

	int*ljump = new int[n];
	int len_ljump = 0;
	memset(working_array, 0, sizeof(T)*n);
	int nnz = 0;
	lurows[0] = 0;
	memset(diag, 0, sizeof(T)*n);
	//int maxfil=lfil;
	int len_vheap = 0;

	cmp_id<T>cmp1 = cmp_id<T>(working_array);
	auto gt1 = std::greater<int>();


	int*value_heap;

	for (int i = 0; i < n; i++) {

		double tnorm1 = 0;
		T tnorm = 0;

		len_ujump = 0;
		len_ljump = 0;

		for (int j = rows[i]; j < rows[i + 1]; j++) {
			int k = cols[j];
			working_array[k] = val[j];
			tnorm1 += val[j] * val[j];

			if (k < i) {
				ljump[len_ljump] = k;
				len_ljump += 1;

			}

			if (i < k) {
				ujump[len_ujump] = k;
				len_ujump += 1;
			}

		}
		//tnorm /= (rows[i + 1] - rows[i]);
		tnorm = (T)sqrt(tnorm1);
		tnorm = tnorm * droptol;
		value_heap = lucols + nnz;
		len_vheap = 0;
		//gettimeofday(&t2, NULL);
		//lfil=max(maxfil,len_ljump/2);
		while (len_ljump > 0) {
			int k = ljump[0];
			std::pop_heap(ljump, ljump + len_ljump, gt1);
			len_ljump -= 1;
			//gettimeofday(&tc1, NULL);

			T factor = working_array[k] * diag[k];
			if (fabs(factor) < tnorm) {
				working_array[k] = 0;
				continue;
			}


			for (int ii = uptr[k]+1; ii < lurows[k + 1]; ii++) {
				int j = lucols[ii];
				T temp_waj = working_array[j];

				if (temp_waj == 0) {
					if (j < i) {
						ljump[len_ljump] = j;
						len_ljump += 1;
						std::push_heap(ljump, ljump + len_ljump, gt1);
					}
					if (i < j) {
						ujump[len_ujump] = j;
						len_ujump += 1;
					}
				}

				working_array[j] = temp_waj - factor * luval[ii];

			}


			working_array[k] = factor;

			if (len_vheap < lfil) {
				value_heap[len_vheap] = k;
				len_vheap += 1;
				std::push_heap(value_heap, value_heap + len_vheap, cmp1);
			}
			else {
				int temp = value_heap[0];
				if (fabs(factor) > fabs(working_array[temp])) {
					working_array[temp] = 0;
					//wa[temp] = false;
					value_heap[0] = k;
					heap_swap(value_heap, len_vheap, working_array);
				}
				else {
					working_array[k] = 0;
					//wa[k] = 0;
				}
			}


		}


		for (int kkl = 0; kkl < len_vheap; kkl++) {
			int imax = kkl;
			for (int akkl = kkl + 1; akkl < len_vheap; akkl++) {
				if (value_heap[akkl] < value_heap[imax]) {
					imax = akkl;
				}
			}
			int tempak = value_heap[kkl];
			value_heap[kkl] = value_heap[imax];
			value_heap[imax] = tempak;
		}


		for (int it = 0; it < len_vheap; it++) {
			//lucols[nnz] = value_heap[it];
			//if (it < lfil) {
			luval[nnz] = working_array[value_heap[it]];
			nnz += 1;
			//}
			working_array[value_heap[it]] = 0;

		}


		uptr[i] = nnz;

		
		if (working_array[i] != 0) {
			diag[i] = 1 / working_array[i];
			luval[nnz] = working_array[i];
			lucols[nnz] = i;
			working_array[i] = 0;
			nnz += 1;

		}
		else {
			diag[i] = 1 / ((small_num / droptol * tnorm + tnorm));
			luval[nnz]= ((small_num / droptol * tnorm + tnorm));
			lucols[nnz] = i;
			nnz += 1;
		}


		value_heap = lucols + nnz;

		len_vheap = 0;

		//lfil=max(maxfil,len_ujump/2);
		for (int iki = 0; iki < len_ujump; iki++) {
			int ki = ujump[iki];


			if (fabs(working_array[ki]) > tnorm) {


				if (len_vheap < lfil) {
					value_heap[len_vheap] = ki;
					len_vheap += 1;
					std::push_heap(value_heap, value_heap + len_vheap, cmp1);
				}
				else {
					int temp = value_heap[0];
					if (fabs(working_array[ki]) > fabs(working_array[temp])) {

						working_array[temp] = 0;
						//wa[temp] = 0;
						value_heap[0] = ki;
						heap_swap(value_heap, len_vheap, working_array);
					}
					else {
						working_array[ki] = 0;
						//wa[ki] = 0;
					}
				}

			}
			else {
				working_array[ki] = 0;

			}

		}

		//gettimeofday(&t6, NULL);

		for (int kkl = 0; kkl < len_vheap; kkl++) {
			int imax = kkl;
			for (int akkl = kkl + 1; akkl < len_vheap; akkl++) {
				if (value_heap[akkl] < value_heap[imax]) {
					imax = akkl;
				}
			}
			int tempak = value_heap[kkl];
			value_heap[kkl] = value_heap[imax];
			value_heap[imax] = tempak;
		}



		for (int it = 0; it < len_vheap; it++) {

			luval[nnz] = working_array[value_heap[it]];
			nnz += 1;
			//}
			working_array[value_heap[it]] = 0;

		}

		len_vheap = 0;

		lurows[i + 1] = nnz;

	}
#undef skip_count
#undef skip_space
	delete[]working_array;
	delete[]ljump;
	delete[]ujump;

}


template<typename T>
void lusolve_no_pivot_no_diag_csr(T*luvalue, int*rows, int*cols, T*diag, int*uptr, T*x, int n, int*ipermn, T*y) {
	//T*tempx = new T[n];
	for (int i = 0; i < n; i++) {
		T sum = y[i];
		for (int ii = rows[i]; ii < uptr[i]; ii++) {
			int j = cols[ii];
			sum -= x[j] * luvalue[ii];
		}
		x[i] = sum;
	}
	for (int i = n - 1; i >= 0; i--) {
		T sum = x[i];
		for (int ii = uptr[i] + 1; ii < rows[i + 1]; ii++) {
			sum -= x[cols[ii]] * luvalue[ii];
		}
		x[i] = sum / luvalue[uptr[i]];
	}
}
template<typename T>
void ilu0_simple(T*luval, T*val, int*rows, int*cols, int n, T*diag, int lfil, int*lucols, int*lurows, int*uptr, T permtol, T droptol, int*iperm, int*ipermn,\
int sweep) {

	memset(uptr, -1, sizeof(int)*n);

    int num_threads=1;
#pragma omp parallel
{
    //num_threads=omp_get_num_threads();
}
	    sweep=1;
//	#pragma omp parallel for //schedule(dynamic,task)
	for (int i = 0; i < lurows[n]; i++) {
		luval[i]=val[i];
	}
    	//printf("num_threads = %d!!!!\n",num_threads);
//#pragma omp parallel for //schedule(dynamic,32)
	for (int i = 0; i < n; i++) {
		int small = 0;
		for (int j = lurows[i]; j < lurows[i + 1]; j++) {
			if (cols[j] == i) {
				uptr[i] = j;
			}
		}
	}

	for (int sweepi = 0; sweepi < sweep; sweepi++) {

//#pragma omp parallel for //schedule(dynamic,32)
			for (int i = 0; i < n; i++) {
				int j = 0;
				int upart = uptr[i];

				for (j = lurows[i]; j < upart; j++) {
					int k = lucols[j];

					if (fabs(luval[uptr[k]])<1e-6){
						luval[uptr[k]] = 1e-6;//small_num/droptol*tnorm[k] + tnorm[k];
					}
					luval[j] = luval[j]/luval[uptr[k]];

					int subptr = uptr[k];//////lurows[k];
					for (int jj = j + 1; jj < lurows[i + 1]; jj++) {

						int kk = lucols[jj];

						while (subptr < lurows[k+1] && lucols[subptr] < kk) {
							subptr++;
						}

						if (subptr < lurows[k + 1]) {
							if (lucols[subptr] == kk) {
								T tmp=0;
								tmp= luval[j] * luval[subptr];
								luval[jj] = luval[jj]-tmp;
							}
						}

					}
				}

				if (fabs(luval[uptr[i]])<0.000001f){
					//printf("LU diag val[%d] too small %lg!!!\n",i,luval[diag_ptr[i]]);
					//luval[diag_ptr[i]] = droptol;
					luval[uptr[i]] = 0.000001f;//small_num/droptol*tnorm[i] + tnorm[i];
				}


			}
		
		
	}
}
template<typename T>
void ilu0_simple(T*luval, T*val, int*rows, int*cols, int n, int*uptr) {

	/*for test*/
	memset(uptr, -1, sizeof(int)*n);

	#pragma omp parallel for //schedule(dynamic,task)
	for (int i = 0; i < rows[n]; i++) {
		luval[i]=val[i];
	}
#pragma omp parallel for //schedule(dynamic,32)
	for (int i = 0; i < n; i++) {
		int small = 0;
		for (int j = rows[i]; j < rows[i + 1]; j++) {
			if (cols[j] == i) {
				uptr[i] = j;
				break;
			}
		}
	}

	for (int i = 0; i < n; i++) {
		int j = 0;
		int upart = uptr[i];
		for (j = rows[i]; j < upart; j++) {
			int k = cols[j];

			if (fabs(luval[uptr[k]])<1e-20){
				luval[uptr[k]] = 1e-20;//small_num/droptol*tnorm[k] + tnorm[k];
			}
			luval[j] = luval[j]/luval[uptr[k]];

			int subptr = uptr[k];
			for (int jj = j + 1; jj < rows[i + 1]; jj++) {

				int kk = cols[jj];

				while (cols[subptr] < kk&&subptr < rows[k + 1]) {
					subptr++;
				}

				if (subptr < rows[k + 1]) {
					if (cols[subptr] == kk) {
						T tmp=0;
						tmp= luval[j] * luval[subptr];
						luval[jj] = luval[jj]-tmp;
					}
				}
				else
				   	break;
			}
		}

		if (fabs(luval[uptr[i]])<1e-20){
			luval[uptr[i]] = 1e-20;//small_num/droptol*tnorm[i] + tnorm[i];
		}
	}
		
}


template<typename T>
void iluk_simple(T*luval, T*val, int*rows, int*cols, int n, int*lucols, int*lurows, int*uptr) {

	/*for test*/
	T*old_val = new T[lurows[n]];
	#pragma omp parallel for //schedule(dynamic,task)
	for (int i = 0; i < lurows[n]; i++) {
		old_val[i]=0;//0
		luval[i]=0;
	}
	T*working_array = new T[lurows[n]];
	
	#pragma omp parallel for //schedule(dynamic,task)
	for (int i = 0; i < n; i++) {
		int jj = rows[i];
		for (int j = lurows[i]; j < lurows[i + 1]; j++) {
			int k = lucols[j];
			if (lucols[j] == i) {
				uptr[i] = j;
			}
			for (; jj < rows[i + 1]; jj++) {
				int kk = cols[jj];
				if (k == kk) {
					old_val[j] = val[jj];
					luval[j] = val[jj];
					break;
				}
				else if (kk > k)break;
			}
		}
	}
		for (int i = 0; i < n; i++) {
			int j = 0;
			int upart = uptr[i];

			for (j = lurows[i]; j < lurows[i + 1]; j++) {
				working_array[j] = old_val[j];
			}

			for (j = lurows[i]; j < upart; j++) {
				int k = lucols[j];

				working_array[j] /= luval[uptr[k]];
				T factor = working_array[j];

				int subptr=uptr[k];
				for (int jj = j + 1; jj < lurows[i + 1]; jj++) {

					int kk = lucols[jj];

					while (lucols[subptr] < kk&&subptr < lurows[k + 1]) {
						subptr++;
					}

					if (subptr < lurows[k + 1]) {
						if (lucols[subptr] == kk) {
							working_array[jj] -= (factor* luval[subptr]);
						}
					}

				}
			}
			for (int j = lurows[i]; j < lurows[i + 1]; j++) {
				luval[j] = working_array[j];
			}
		}
	delete[]working_array;
	delete[]old_val;
}


template<typename T>
void ilu0_diag_is_ptr(T*luval, T*val, int*rows, int*cols, int n, T*diag, int lfil, int*lucols, int*lurows, int*uptr, T permtol, T droptol, int*iperm, int*ipermn,\
int sweep) {

	/*for test*/
	memset(uptr, -1, sizeof(int)*n);
	T*old_val= (T*)malloc(lurows[n]*sizeof(T));//new T[lurows[n]];

    int num_threads=1;
#pragma omp parallel
{
    //num_threads=omp_get_num_threads();
}
//	if(num_threads==1)
	    sweep=1;
//	#pragma omp parallel for //schedule(dynamic,task)
	for (int i = 0; i < lurows[n]; i++) {
		old_val[i]=val[i];//0
		luval[i]=val[i];
	}
    	//printf("num_threads = %d!!!!\n",num_threads);
	T*working_array = new T[lurows[n]];
//#pragma omp parallel for //schedule(dynamic,32)
	for (int i = 0; i < n; i++) {
		int small = 0;
		for (int j = lurows[i]; j < lurows[i + 1]; j++) {
					//old_val[j] = val[j];
					//luval[j] = val[j];
			if (cols[j] == i) {
				uptr[i] = j;
			}
		}
		//if (diag_ptr[i] == -1) { printf("diag[%d]==0\n", i); }
	}

	for (int sweepi = 0; sweepi < sweep; sweepi++) {

//#pragma omp parallel for //schedule(dynamic,32)
			for (int i = 0; i < n; i++) {
				int j = 0;
				int upart = uptr[i];

				for (j = lurows[i]; j < lurows[i + 1]; j++) {
					working_array[j] = old_val[j];

				}

				for (j = lurows[i]; j < upart; j++) {
					int k = lucols[j];

					if (fabs(luval[uptr[k]])<1e-6){
						luval[uptr[k]] = 1e-6;//small_num/droptol*tnorm[k] + tnorm[k];
					}
					working_array[j] = working_array[j]/luval[uptr[k]];

					int subptr = uptr[k];//////lurows[k];
					for (int jj = j + 1; jj < lurows[i + 1]; jj++) {

						int kk = lucols[jj];

						while (subptr < lurows[k+1] && lucols[subptr] < kk) {
							subptr++;
						}

						if (subptr < lurows[k + 1]) {
							if (lucols[subptr] == kk) {
								T tmp=0;
								tmp= working_array[j] * luval[subptr];
								working_array[jj] = working_array[jj]-tmp;
								//working_array[jj] -= (working_array[j] * luval[subptr]);
							}
						}

					}
				}

				for (int j = lurows[i]; j < lurows[i + 1]; j++) {

					luval[j] = working_array[j];
				}
				if (fabs(luval[uptr[i]])<0.000001f){
					//printf("LU diag val[%d] too small %lg!!!\n",i,luval[diag_ptr[i]]);
					//luval[diag_ptr[i]] = droptol;
					luval[uptr[i]] = 0.000001f;//small_num/droptol*tnorm[i] + tnorm[i];
				}


			}
		
		
	}
	//for(int i=0;i<n;i++){
	//	for(int j=lurows[i];j<lurows[i+1];j++)
	//		printf("%d %d %lg\n",i,lucols[j],luval[j]);
	//}


	//getchar();
	delete[]working_array;

	//free(old_val);
}

template<typename T>
void ilu0_diag_is_ptr_uptr_input_simplified(T*luval, T*val, int*rows, int*cols, int n, T*diag, int lfil, int*lucols, int*lurows, int*uptr, T permtol, T droptol, int*iperm, int*ipermn, \
	int sweep) {

	/*for test*/
	T*old_val = new T[lurows[n]];
	//memset(old_val,0,lurows[n]*sizeof(T));
	//memset(luval,0,lurows[n]*sizeof(T));
	#pragma omp parallel for //schedule(dynamic,task)
	for (int i = 0; i < lurows[n]; i++) {
		old_val[i]=0;//0
		luval[i]=0;
	}
	T*working_array = new T[lurows[n]];
	T*tnorm = new T[n];
	
    //struct timeval time1, time2;
    //gettimeofday(&time1, NULL); 
	#pragma omp parallel for //schedule(dynamic,task)
	for (int i = 0; i < n; i++) {
		int jj = rows[i];
		T tempnorm = 0;
		for (int j = lurows[i]; j < lurows[i + 1]; j++) {
			int k = lucols[j];
			for (; jj < rows[i + 1]; jj++) {
				int kk = cols[jj];
				if (k == kk) {
					old_val[j] = val[jj];
					luval[j] = val[jj];
					tempnorm += (val[jj] * val[jj]);
					break;
				}
				else if (kk > k)break;
			}
			
			
		}
		tempnorm = sqrt(tempnorm);
		tempnorm *= droptol;
		tnorm[i] = tempnorm;
		
	}
    //gettimeofday(&time2, NULL); 
    //double elapsed_time1 = (time2.tv_sec - time1.tv_sec) * 1000. +(time2.tv_usec - time1.tv_usec) / 1000.; 
    //printf("Init ilu matrix time: %lf(ms)\n", elapsed_time1); 
    //gettimeofday(&time1, NULL); 
	for (int sweepi = 0; sweepi < sweep; sweepi++) {


		#pragma omp parallel for //schedule(dynamic,300)
		for (int i = 0; i < n; i++) {
			int j = 0;
			int upart = uptr[i];

			for (j = lurows[i]; j < lurows[i + 1]; j++) {
				working_array[j] = old_val[j];
			}

			for (j = lurows[i]; j < upart; j++) {
				int k = lucols[j];

				///if (fabs(luval[uptr[k]]) < tnorm[k])
				///	luval[uptr[k]] = small_num/droptol*tnorm[k] + tnorm[k];				
				working_array[j] /= luval[uptr[k]];
				T factor = working_array[j];

				int subptr=uptr[k];
				for (int jj = j + 1; jj < lurows[i + 1]; jj++) {

					int kk = lucols[jj];

					while (lucols[subptr] < kk&&subptr < lurows[k + 1]) {
						subptr++;
					}

					if (subptr < lurows[k + 1]) {
						if (lucols[subptr] == kk) {
							working_array[jj] -= (factor* luval[subptr]);
						}
					}

				}
			}
			for (int j = lurows[i]; j < lurows[i + 1]; j++) {
				luval[j] = working_array[j];
			}
			if (fabs(luval[uptr[i]])<tnorm[i]) {
				luval[uptr[i]] = small_num / droptol * tnorm[i] + tnorm[i];
			}
		}
	}
    //gettimeofday(&time2, NULL); 
    //double elapsed_time2 = (time2.tv_sec - time1.tv_sec) * 1000. +(time2.tv_usec - time1.tv_usec) / 1000.; 
    //printf("Compute ilu matrix time: %lf(ms)\n", elapsed_time2); 
	delete[]working_array;
	delete[]tnorm;
	delete[]old_val;
}

template<typename T>
void lusolve_for_ilu0(T*luvalue, int*rows, int*cols, T*diag, int*uptr, T*x, int n, int*ipermn, T*y) {
	//T*tempx = new T[n];
	T sum,dive,tmp;
	for (int i = 0; i < n; i++) {
		sum = y[i];
		for (int ii = rows[i]; ii < uptr[i]; ii++) {
			int j = cols[ii];
			tmp= x[j] * luvalue[ii];
			sum=sum-tmp;
			//sum -= x[j] * luvalue[ii];
		}
		x[i] = sum;
	}
	for (int i = n - 1; i >= 0; i--) {
		sum = x[i];
		int ii = uptr[i];
		if (cols[ii] == i) {
			dive = luvalue[ii];
			ii = ii + 1;
			//if(vabsh_f16(dive)<1e-20){
			//	printf("U diag [%d] too small %lg!!!\n",i,dive);
			//	dive=1e-20;
			//}
		}
		else {
			dive = 1e-6;
			printf("U diag [%d] has 0!!!\n",i);
		}
		for (; ii < rows[i + 1]; ii++) {
			tmp = x[cols[ii]] * luvalue[ii];
			sum = sum-tmp;
			//sum -= x[cols[ii]] * luvalue[ii];
		}
		x[i] = sum / dive;
	}

}

template void ilutp_final(double*luval, double*val, int*rows, int*cols, int n, double*diag, int lfil, int*lucols, int*lurows, int*uptr, double permtol, double droptol, int*iperm, int*ipermn);
template void ilutp_final(float*luval, float*val, int*rows, int*cols, int n, float*diag, int lfil, int*lucols, int*lurows, int*uptr, float permtol, float droptol, int*iperm, int*ipermn);
template void ilu0_simple(double*luval, double*val, int*rows, int*cols, int n, double*diag, int lfil, int*lucols, int*lurows, int*uptr, double permtol, double droptol, int*iperm, int*ipermn, int sweep);
template void ilu0_simple(float*luval, float*val, int*rows, int*cols, int n, float*diag, int lfil, int*lucols, int*lurows, int*uptr, float permtol, float droptol, int*iperm, int*ipermn,int sweep);
template void ilu0_simple(double*luval, double*val, int*rows, int*cols, int n, int* uptr);
template void ilu0_simple(float*luval, float*val, int*rows, int*cols, int n, int* uptr);
