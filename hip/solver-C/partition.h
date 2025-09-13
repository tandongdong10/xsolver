#pragma once 

void balanced_partition_row_by_nnz(const int *acc_sum_arr, int rows, int num_threads, int *partition);
void csr_row_partition( int rows, int num_threads, int* partition);
