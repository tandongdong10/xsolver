#ifndef _DISTRIBUTE_MATRIX_H_
#define _DISTRIBUTE_MATRIX_H_
#include <mpi.h>
#include <string.h>
#include <vector>
#include <map>
template <typename ValueType>
void distribute_matrix(HostMatrix *lmat,
                       HostMatrix *gmat,
                       topology_c *topo_c)
{
    int rank;
    int num_procs;
    MPI_Comm comm=MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);
    size_t global_nrow = gmat->n;
    int *global_row_offset = NULL;
    int *global_col = NULL;
    ValueType *global_val = NULL;

    global_row_offset=gmat->getptr();
    global_col=gmat->getidx();
    global_val=gmat->getval();

    // Compute local matrix sizes
    std::vector<int> local_size(num_procs);

    for (int i=0; i<num_procs; ++i)
    {
        local_size[i] = global_nrow / num_procs;
    }

    if (global_nrow % num_procs != 0)
    {
        for (size_t i=0; i<global_nrow % num_procs; ++i)
        {
            ++local_size[i];
        }
    }
    topo_c->scount=new int[num_procs];
    topo_c->displs=new int[num_procs+1];
    
    // Compute index offsets
    std::vector<int> index_offset(num_procs+1);
    index_offset[0] = 0;
    topo_c->displs[0]=0;
    for (int i=0; i<num_procs; ++i)
    {
	topo_c->scount[i]=local_size[i];
        index_offset[i+1] = index_offset[i] + local_size[i];
	topo_c->displs[i+1]=index_offset[i+1];
    }

    // Read sub matrix - row_offset
    int local_nrow = local_size[rank];
    std::vector<int> local_row_offset(local_nrow+1);

    for (int i=index_offset[rank], k=0; k<local_nrow+1; ++i, ++k)
    {
        local_row_offset[k] = global_row_offset[i];
    }

    //free_host(&global_row_offset);

    // Read sub matrix - col and val
    int local_nnz = local_row_offset[local_nrow] - local_row_offset[0];
    std::vector<int> local_col(local_nnz);
    std::vector<ValueType> local_val(local_nnz);

    for (int i=local_row_offset[0], k=0; k<local_nnz; ++i, ++k)
    {
        local_col[k] = global_col[i];
        local_val[k] = global_val[i];
    }

    // Shift row_offset entries
    int shift = local_row_offset[0];
    for (int i=0; i<local_nrow+1; ++i)
    {
        local_row_offset[i] -= shift;
    }

    int interior_nnz = 0;
    int ghost_nnz = 0;
    int boundary_nnz = 0;
    int neighbors = 0;

    std::vector<std::vector<int> > boundary(num_procs, std::vector<int>());
    std::vector<bool> neighbor(num_procs, false);
    std::vector<std::map<int, bool> > checked(num_procs, std::map<int, bool>());

    for (int i=0; i<local_nrow; ++i)
    {
        for (int j=local_row_offset[i]; j<local_row_offset[i+1]; ++j)
        {

            // Interior point
            if (local_col[j] >= index_offset[rank] && local_col[j] < index_offset[rank+1])
            {
                ++interior_nnz;
            }
            else
            {
                // Boundary point above current process
                if (local_col[j] < index_offset[rank])
                {
                    // Loop over ranks above current process
                    for (int r=rank-1; r>=0; --r)
                    {
                        // Check if boundary belongs to rank r
                        if (local_col[j] >= index_offset[r] && local_col[j] < index_offset[r+1])
                        {
                            // Add boundary point to rank r if it has not been added yet
                            if (!checked[r][i+index_offset[rank]])
                            {
                                boundary[r].push_back(i+index_offset[rank]);
                                neighbor[r] = true;
                                ++boundary_nnz;
                                checked[r][i+index_offset[rank]] = true;
                            }
                            ++ghost_nnz;
                            // Rank for current boundary point local_col[j] has been found
                            // Continue with next boundary point
                            break;
                        }
                    }
                }

                // boundary point below current process
                if (local_col[j] >= index_offset[rank+1])
                {
                    // Loop over ranks above current process
                    for (int r=rank+1; r<num_procs; ++r)
                    {
                        // Check if boundary belongs to rank r
                        if (local_col[j] >= index_offset[r] && local_col[j] < index_offset[r+1])
                        {
                            // Add boundary point to rank r if it has not been added yet
                            if (!checked[r][i+index_offset[rank]])
                            {
                                boundary[r].push_back(i+index_offset[rank]);
                                neighbor[r] = true;
                                ++boundary_nnz;
                                checked[r][i+index_offset[rank]] = true;
                            }
                            ++ghost_nnz;
                            // Rank for current boundary point local_col[j] has been found
                            // Continue with next boundary point
                            break;
                        }
                    }
                }
            }

        }
    }

    for (int i=0; i<num_procs; ++i)
    {
        if (neighbor[i] == true)
        {
            ++neighbors;
        }
    }
    //topo_c->total_nbs_thisproc=neighbors;
    //topo_c->nGhstCells=ghost_nnz;//////??????
    //printf("rank = %d  , neighbors = %d , nGhstCells=%d \n",rank,neighbors, ghost_nnz);
    std::vector<MPI_Request> mpi_req(neighbors*2);
    int n = 0;
    // Array to hold boundary size for each interface
    std::vector<int> boundary_size(neighbors);
    std::vector<int> send_size(num_procs);

    // MPI receive boundary sizes
    for (int i=0; i<num_procs; ++i)
    {
        // If neighbor receive from rank i is expected...
        if (neighbor[i] == true)
        {
            // Receive size of boundary from rank i to current rank
            MPI_Irecv(&(boundary_size[n]), 1, MPI_INT, i, 0, comm, &mpi_req[n]);
            ++n;
        }
    }

    // MPI send boundary sizes
    for (int i=0; i<num_procs; ++i)
    {
        // Send required if boundary for rank i available
        if (boundary[i].size() > 0)
        {
            send_size[i] = boundary[i].size();
            // Send size of boundary from current rank to rank i
            MPI_Isend(&(send_size[i]), 1, MPI_INT, i, 0, comm, &mpi_req[n]);
            ++n;
        }
    }
    // Wait to finish communication
    //printf("n=%d\n",n);
    MPI_Waitall(n, &(mpi_req[0]), MPI_STATUSES_IGNORE);
    
    n = 0;
    // Array to hold boundary offset for each interface
    int k = 0;
    std::vector<int> recv_offset(neighbors+1);
    std::vector<int> send_offset(neighbors+1);
    recv_offset[0] = 0;
    send_offset[0] = 0;
    for (int i=0; i<neighbors; ++i)
    {
        recv_offset[i+1] = recv_offset[i] + boundary_size[i];
	    //printf("myid=%d, neighbor id=%d, receive size=%d\n",rank, i, boundary_size[i]);
    }

    for (int i=0; i<num_procs; ++i)
    {
        if (neighbor[i] == true)
        {
            send_offset[k+1] = send_offset[k] + boundary[i].size();
	    //printf("myid=%d, neighbor id=%d, send size=%d\n",rank, i, boundary[i].size());
            ++k;
        }
    }

    // Array to hold boundary for each interface
    topo_c->nbs_thisproc=new int[neighbors];
    std::vector<std::vector<int> > local_boundary(neighbors);// this rank needs to get rowidx from other ranks
    for (int i=0; i<neighbors; ++i)
    {
        local_boundary[i].resize(boundary_size[i]);
    }
    k=0;
    int displs_proc_num=0;
    topo_c->exchange_displs_proc=new int[num_procs+1];
    int *displs=topo_c->exchange_displs_proc;
    // MPI receive boundary
    for (int i=0; i<num_procs; ++i)
    {
        // If neighbor receive from rank i is expected...
        if (neighbor[i] == true)
        {
            // Receive boundary from rank i to current rank
	    //topo_c->nbs_thisproc[k++]=i;
	    //displs[i]=displs_proc_num;	    
            MPI_Irecv(local_boundary[n].data(), boundary_size[n], MPI_INT, i, 0, comm, &mpi_req[n]);
	    //displs_proc_num+=boundary_size[n];	    
	    //displs[i+1]=displs_proc_num;	    
            ++n;
        }
    }
    //for(int i=0;i<neighbors;i++)
    //	printf("rank = %d  , neighbors rank=%d\n",rank,topo_c->nbs_thisproc[i]);

    // MPI send boundary
    for (int i=0; i<num_procs; ++i)
    {
        // Send required if boundary for rank i is available
        if (boundary[i].size() > 0)
        {
            // Send boundary from current rank to rank i
	    topo_c->nbs_thisproc[k++]=i;
	    displs[k-1]=displs_proc_num;	    
            MPI_Isend(&(boundary[i][0]), boundary[i].size(), MPI_INT, i, 0, comm, &mpi_req[n]);
	    displs_proc_num+=boundary[i].size();	    
	    displs[k]=displs_proc_num;	    
            ++n;
        }
    }

    // Wait to finish communication
    MPI_Waitall(n, &(mpi_req[0]), MPI_STATUSES_IGNORE);
    // Total boundary size
    topo_c->exchange_displs_proc_receive=new int[neighbors+1];
    int *displs_receive=topo_c->exchange_displs_proc_receive;
    int nnz_boundary = 0;
    //for (int i=0; i<num_procs; ++i)
    for (int i=0; i<neighbors; ++i)
    {
	    displs_receive[i]=nnz_boundary;
            nnz_boundary += boundary_size[i];
	    displs_receive[i+1]=nnz_boundary;
    }
    // Create local boundary index array
    k = 0;
    topo_c->exchange_ptr=new int[displs_proc_num];
    int *bnd=topo_c->exchange_ptr;
    //std::vector<int> bnd(boundary_nnz);
    for (int i=0; i<num_procs; ++i)
    {
        for (unsigned int j=0; j<boundary[i].size(); ++j)
        {
            bnd[k] = boundary[i][j]-index_offset[rank];
            ++k;
        }
    }
    //for(int i=0;i<=num_procs;i++)
    //	printf("rank = %d  , exchange_displs_proc=%d\n",rank,topo_c->exchange_displs_proc[i]);

    // Create boundary index array
    k=0;
    std::vector<int> boundary_index(nnz_boundary);
    for (int i=0; i<neighbors; ++i)
    {
	//int k=displs[topo_c->nbs_thisproc[i]];
	//printf("myid=%d, k=%d\n",rank,k);
        for (int j=0; j<boundary_size[i]; ++j)
        {
            boundary_index[k] = local_boundary[i][j];
            ++k;
        }
    }

    //for(int i=0;i<displs_proc_num;i++)
    //	printf("rank = %d  , exchange_ptr=%d\n",rank,topo_c->exchange_ptr[i]);
    // Create map with boundary index relations
    std::map<int, int> boundary_map;

    for (int i=0; i<nnz_boundary; ++i)
    {
        boundary_map[boundary_index[i]] = i;
    }
    // Build up ghost and interior matrix
    int *ghost_row = new int[local_nrow+1];
    int *ghost_col = new int[ghost_nnz];
    ValueType *ghost_val = new ValueType[ghost_nnz];

    memset(ghost_row, 0, sizeof(int)*(local_nrow+1));
    memset(ghost_col, 0, sizeof(int)*ghost_nnz);
    memset(ghost_val, 0, sizeof(ValueType)*ghost_nnz);

    int *row_offset = new int[local_nrow+1];
    int *col = new int[interior_nnz];
    ValueType *val = new ValueType[interior_nnz];

    memset(row_offset, 0, sizeof(int)*(local_nrow+1));
    memset(col, 0, sizeof(int)*interior_nnz);
    memset(val, 0, sizeof(ValueType)*interior_nnz);
    lmat->MallocMatrix(local_nrow,nnz_boundary, interior_nnz+ghost_nnz);//???????????
    //printf("local mtx size=%d , nHalo = %d, allnnz=%d\n",lmat->n,lmat->getnHalo(),lmat->nnz);
    row_offset[0] = 0;
    k = 0;
    int l = 0;
    for (int i=0; i<local_nrow; ++i)
    {
        for (int j=local_row_offset[i]; j<local_row_offset[i+1]; ++j)
        {

            // Boundary point -- create ghost part
            if (local_col[j] < index_offset[rank] || local_col[j] >= index_offset[rank+1])
            {
                //ghost_col[k] = local_col[j];
                ghost_col[k] = boundary_map[local_col[j]];
                ghost_val[k] = local_val[j];
                ++k;
            }
            else
            {
                // Interior point -- create interior part
                int c = local_col[j] - index_offset[rank];

                col[l] = c;
                val[l] = local_val[j];
                ++l;
            }
        }
        row_offset[i+1] = l;
        ghost_row[i+1] = k;
    }
    std::vector<int> recv(neighbors);
    std::vector<int> sender(neighbors);
    int *local_all_rowptr=lmat->getptr();
    int *local_all_colidx=lmat->getidx();
    double *local_all_val=lmat->getval();
    int nnz_now=0;
    local_all_rowptr[0]=0;
    for(int i=0;i<local_nrow;i++){
        for(int j=row_offset[i];j<row_offset[i+1];j++){
            local_all_colidx[nnz_now]=col[j];
            local_all_val[nnz_now]=val[j];
	    //printf("%d %d %lg\n",i+1,local_all_colidx[nnz_now]+1,local_all_val[nnz_now]);
            nnz_now++;
        }
        for(int j=ghost_row[i];j<ghost_row[i+1];j++){
            local_all_colidx[nnz_now]=ghost_col[j]+local_nrow;
            local_all_val[nnz_now]=ghost_val[j];
	    //printf("!!!!!!!%d %d %lg\n",i+1,local_all_colidx[nnz_now]+1,local_all_val[nnz_now]);
            nnz_now++;
        }
        local_all_rowptr[i+1]=nnz_now;
    }
    //printf("total local mtx n=%d nnz=%d\n",local_nrow,nnz_now);
    int nbc = 0;
    for (int i=0; i<num_procs; ++i)
    {
        if (neighbor[i] == true)
        {
            recv[nbc] = i;
            sender[nbc] = i;
            ++nbc;
        }
    }
    xsolver_communicator_setup(neighbors,local_size[rank],nnz_boundary, topo_c->nbs_thisproc,topo_c->exchange_ptr,topo_c->exchange_displs_proc,displs_proc_num,topo_c->exchange_displs_proc_receive);
    delete []ghost_row;
    delete []ghost_col;
    delete []ghost_val;
    delete []row_offset;
    delete []col;
    delete []val;
}
#endif
