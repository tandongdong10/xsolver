#pragma once
#ifndef _SORT_KEY_VAL_PAIR_
#define _SORT_KEY_VAL_PAIR_
template<typename T>
void swap(T *a , T *b)
{
    T tmp = *a;
    *a = *b;
    *b = tmp;
}


template<typename iT, typename vT>
int partition(iT *key, vT *val, int length, int pivot_index)
{
    int i  = 0 ;
    int small_length = pivot_index;

    iT pivot = key[pivot_index];
    swap<iT>(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap<vT>(&val[pivot_index], &val[pivot_index + (length - 1)]);

    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap<iT>(&key[pivot_index+i],  &key[small_length]);
            swap<vT>(&val[pivot_index+i],&val[small_length]);
            small_length++;
        }
    }

    swap<iT>(&key[pivot_index + length - 1],  &key[small_length]);
    swap<vT>(&val[pivot_index + length - 1],&val[small_length]);

    return small_length;
}


template<typename iT, typename vT>
void quick_sort_key_val_pair(iT *key, vT *val, int length)
{
    if(length == 0 || length == 1)
        return;

    int small_length = partition<iT, vT>(key, val, length, 0) ;
    quick_sort_key_val_pair<iT, vT>(key, val, small_length);
    quick_sort_key_val_pair<iT, vT>(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
}
#endif // sort_key_val_pair.h
