#ifndef SOLVER_H
#define SOLVER_H

#define TYPE double

void my_solvecg_c_(int maxiter, TYPE tol, int nIntCells, TYPE *a_p, TYPE *a_l, int *NbCell_ptr_c, int *NbCell_s, TYPE *b0, TYPE *x0, TYPE *res0, int *usediter);
void my_solvebicgstab_c_(int maxiter, TYPE tol, int nIntCells, TYPE *a_p, TYPE *a_l, int *NbCell_ptr_c, int *NbCell_s, TYPE *b0, TYPE *x0, TYPE *res0, int *usediter);

#endif
