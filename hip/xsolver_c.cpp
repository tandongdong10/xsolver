#include "hip/hip_runtime.h"
#include <stdio.h>
#include "xsolver_c.h"

#include <sys/time.h>

#ifdef HAVE_MPI
extern struct topology_c topo_c;
int *scount;
#endif

int pam[3];

SOLVER solver;
PRECON precon;
HostMatrix *hostmtx;
HostMatrix *devicemtx;
HostMatrix *hostmtxspmv;
HostMatrix *hstpre;
Xsolver *xsol;
int isdistribute;
//int time_num=0;

double elapsed_time_xsolver;
struct timeval time1,time2;
void gettime1(){
#ifdef HAVE_MPI
	communicator_barrier();
#endif
	gettimeofday(&time1,NULL);
}
void gettime2(std::string s){
	gettimeofday(&time2,NULL);
	elapsed_time_xsolver=(time2.tv_sec - time1.tv_sec) * 1000. +(time2.tv_usec - time1.tv_usec) / 1000.;
#ifdef HAVE_MPI
	if(topo_c.myid==0)
#endif
		printf("%s time =%lg ms\n",s.c_str(),elapsed_time_xsolver);
}

 void xsolver_mat_setup(int nInterior, int *rowptr, int *colidx, double *val, int *advanced_pam){
    memset(pam,0,3*sizeof(int));
    if(advanced_pam[0]==1){
#ifdef HAVE_MPI
	HostMatrix *hostmtxtotal=new HostMatrixCSR();
    	hostmtx=matrixform_set("CSR");
    	hostmtxtotal->create_matrix(nInterior, val, rowptr,colidx);
	distribute_matrix<double>(hostmtx,hostmtxtotal,&topo_c);
    	hostmtx->nHalo=topo_c.nGhstCells;
    	hostmtx->exchange_ptr=topo_c.exchange_ptr;
	delete hostmtxtotal;
	hostmtxtotal=NULL;
	isdistribute=1;
#endif
    }
    else{
	isdistribute=0;
    	hostmtx=matrixform_set("CSR");
    	hostmtx->create_matrix(nInterior, val, rowptr,colidx);
#ifdef HAVE_MPI
    	hostmtx->nHalo=topo_c.nGhstCells;
    	hostmtx->exchange_ptr=topo_c.exchange_ptr;
#endif
    }
    if(advanced_pam[2]==1){
	//GPU cluster
	//need to add
	pam[2]=1;

	gettime1();
	devicemtx=new DeviceMatrixCSR();
	devicemtx->ToDeviceMatrix(hostmtx);
	std::string str="CSR matrix trans from CPU to GPU";
	gettime2(str);

	if(advanced_pam[1]==1){
	    pam[1]=1;
	    //matrix format trans to ELL
	gettime1();
	    hostmtxspmv = new DeviceMatrixELL();
	    hostmtxspmv->ToDeviceMatrix(hostmtx);
	str="Device CSR matrix trans to Device ELL";
	gettime2(str);
	}
    }
    else if(advanced_pam[1]!=0){
	//change csr to other matrix format
	//need to add
	pam[1]=1;
	if(advanced_pam[1]==1){
	    //matrix format trans to ELL
	    hostmtxspmv = new HostMatrixELL();
	    hostmtx->CSRTOELL(hostmtxspmv);
	}
	else if(advanced_pam[1]==2){
	    //matrix format trans to CSC
	    hostmtxspmv = new HostMatrixCSC();
	    hostmtx->CSRTOCSC(hostmtxspmv);
	}
    }
    return;
}
 void xsolver_ksp_settype(char *fmt){
    if(strcmp(fmt,"Bicgstab")==0 || strcmp(fmt,"bicgstab")==0 || strcmp(fmt,"BICGSTAB")==0){
	solver=BICGSTAB;
	if(pam[2]==1)
    	    xsol=new Xsolver_bicgstab<DeviceVector>();//solver_set(solver);
	else 
    	    xsol=new Xsolver_bicgstab<HostVector>();//solver_set(solver);
	return ; 
    }
    else if(strcmp(fmt,"Igcr")==0 || strcmp(fmt,"igcr")==0 || strcmp(fmt,"IGCR")==0){
	solver=IGCR;
	if(pam[2]==1)
    	    xsol=new Xsolver_igcr<DeviceVector>();//solver_set(solver);
	else 
    	    xsol=new Xsolver_igcr<HostVector>();//solver_set(solver);
	return ; 
    }
    else if(strcmp(fmt,"Cg")==0 || strcmp(fmt,"cg")==0 || strcmp(fmt,"CG")==0){
	solver=CG;
	if(pam[2]==1)
    	    xsol=new Xsolver_cg<DeviceVector>();//solver_set(solver);
	else 
    	    xsol=new Xsolver_cg<HostVector>();//solver_set(solver);
	return ; 
    }
    // need to add ksp algorithm icg, ibicgstab, gcr, gmres, fgmres
    //xsol=solver_set(solver);
}
 void xsolver_ksp_set_absolute_tol(double absolute_tol){
    	xsol->set_xsolver_absolute_tol(absolute_tol);
}
 void xsolver_ksp_setoption(double tol, int maxiter, int &usediter, double *resvec, int restart){
    if(solver==BICGSTAB){
    	xsol->set_xsolver(tol,maxiter,usediter,resvec);
    }
    if(solver==IGCR){
	if(restart==0)
    	    xsol->set_xsolver(tol,maxiter,usediter,resvec);
	else
    	    xsol->set_xsolver(tol,maxiter,usediter,resvec,restart);
    }
    if(solver==CG){
    	xsol->set_xsolver(tol,maxiter,usediter,resvec);
    }
    // need to add ksp algorithm icg, ibicgstab, gcr, gmres, fgmres

}
 void xsolver_pc_settype(char *fmt){
    //char *str_Jacobi="jacobi";
    if(strcmp(fmt,"Jacobi")==0 || strcmp(fmt,"jacobi")==0 || strcmp(fmt,"JACOBI")==0){
	precon=JACOBI;
    	xsol->preconditioner_set(precon);
    	HostMatrix *tmp=hostmtx;
    	if(pam[2]==1&&devicemtx!=NULL)
	    hostmtx=devicemtx;
    	xsol->create_precond(hostmtx->m, hostmtx);
    	hostmtx=tmp;
    }
	gettime1();
    if(strcmp(fmt,"Ilu0")==0 || strcmp(fmt,"ilu0")==0 || strcmp(fmt,"ILU0")==0){
	precon=ILU0;
    }
    if(strcmp(fmt,"Ic0")==0 || strcmp(fmt,"ic0")==0 || strcmp(fmt,"IC0")==0){
	precon=IC0;
    }
    if(strcmp(fmt,"Ict")==0 || strcmp(fmt,"ict")==0 || strcmp(fmt,"ICT")==0){
	precon=ICT;
    }
    if(precon==IC0|| precon==ICT){
    	xsol->preconditioner_set(precon);
    	HostMatrix *tmp=hostmtx;
    	if(pam[2]==1&&devicemtx!=NULL){
	        printf("IC0/ICT precond can not run on GPU! Please use other precond!\n");
		exit(0);
	}
	else{
		hstpre=new HostMatrixCSR();
	}
	hstpre->MallocMatrix(hostmtx->m,hostmtx->nnz);
#ifdef HAVE_MPI
	hstpre->ToDiagMatrix(hostmtx);
#else
        hstpre->CopyMatrix(hostmtx);
#endif
	//if(pam[2]==1)
	//    hstpre->based1To0Matrix();
    	xsol->create_precond(hostmtx->m, hstpre);
    	hostmtx=tmp;
    }
    if(strcmp(fmt,"Ilut")==0 || strcmp(fmt,"ilut")==0 || strcmp(fmt,"ILUT")==0){
	precon=ILUP;
    }
    if(strcmp(fmt,"Ilu0_mkl")==0 || strcmp(fmt,"ilu0_mkl")==0 || strcmp(fmt,"ILU0_MKL")==0){
	if(pam[2]==1&&devicemtx!=NULL){
	    printf("Ilu0_mkl can only run on CPU, please use Ilu0!!!\n");
	    printf("The precond has been changed to ilu0 on GPU!!!\n");
	    precon=ILU0;
	}
	else
	    precon=ILU0_MKL;
    }
    if(strcmp(fmt,"Ilut_mkl")==0 || strcmp(fmt,"ilut_mkl")==0 || strcmp(fmt,"ILUT_MKL")==0){
	if(pam[2]==1&&devicemtx!=NULL){
	    printf("Ilut_mkl can only run on CPU, please use ilut!!!\n");
	    printf("The precond has been changed to ilut on GPU!!!\n");
	    precon=ILUP;
	}
	else
	    precon=ILUP_MKL;
    }
    if(precon==ILU0||precon==ILUP){
    	xsol->preconditioner_set(precon);
    	HostMatrix *tmp=hostmtx;
    	if(pam[2]==1&&devicemtx!=NULL){
		hstpre=new DeviceMatrixCSR();
		hostmtx=devicemtx;
	}
	else{
		hstpre=new HostMatrixCSR();
	}
	hstpre->MallocMatrix(hostmtx->m,hostmtx->nnz);
#ifdef HAVE_MPI
	hstpre->ToDiagMatrix(hostmtx);
#else
        hstpre->CopyMatrix(hostmtx);
#endif
	if(pam[2]==1)
	    hstpre->based1To0Matrix();
	//if(pam[2]!=1&&hstpre->getptr()[0]!=0)
	//    based1To0Matrix(hstpre);
    	xsol->create_precond(hostmtx->m, hstpre);
    	hostmtx=tmp;
    }
    if(precon==ILU0_MKL||precon==ILUP_MKL){
    	xsol->preconditioner_set(precon);
    	HostMatrix *tmp=hostmtx;
    	if(pam[2]==1&&devicemtx!=NULL){
		hstpre=new DeviceMatrixCSR();
		hostmtx=devicemtx;
	}
	else{
		hstpre=new HostMatrixCSR();
	}
	hstpre->MallocMatrix(hostmtx->m,hostmtx->nnz);
#ifdef HAVE_MPI
	hstpre->ToDiagMatrix(hostmtx);
#else
        hstpre->CopyMatrix(hostmtx);
#endif
	if(pam[2]==1)
	    hstpre->based1To0Matrix();
	if(pam[2]!=1&&hstpre->getptr()[0]==0)  //need to test
	    based0To1Matrix(hstpre);
    	xsol->create_precond(hostmtx->m, hstpre);
    	hostmtx=tmp;
    }
	std::string str="create precond matrix";
	gettime2(str);
    if(pam[2]==1)
	setupgpu(hostmtx,precon);	
}
 void xsolver_pc_setict(double tol, int max_fill){
    if(precon==ICT)
	xsol->precond->set_ict(tol, max_fill);
}
 void xsolver_pc_setilut(double tol, int max_fill){
    if(precon==ILUP||precon==ILUP_MKL)
	xsol->precond->set_ilut(tol, max_fill);
    //else
	//printf("The function \"xsolver_pc_setilut\" is only used when preconditioned is ilut, which has been ignored!!!\n");
}
 void xsolver_solve(double *rhs, double *sol){
    HostMatrix *tmp=hostmtx;
    if(pam[1]!=0&&hostmtxspmv!=NULL)
	hostmtx=hostmtxspmv;
    else if(pam[2]==1&&devicemtx!=NULL){
	hostmtx=devicemtx;
    }
	gettime1();
    xsol->create_xsolver(hostmtx->m, hostmtx, rhs, sol);
    xsol->xsolver_init();
	std::string str="xsolver init";
	gettime2(str);
	gettime1();
    xsol->xsolver();
	str="xsolver solve";
	gettime2(str);
    xsol->xsolver_free();
    hostmtx=tmp;
}
#ifdef HAVE_MPI
void scatter_mpi(double *q){
    double *q_buf=new double[topo_c.displs[topo_c.size]];
    memcpy(q_buf,q,topo_c.displs[topo_c.size]*sizeof(double));
    MPI_Scatterv(q_buf, topo_c.scount,topo_c.displs, MPI_DOUBLE, q, topo_c.scount[topo_c.myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete []q_buf;

}
void gather_mpi(double *phi){
    double *phi_buf=new double[hostmtx->n];
    memcpy(phi_buf,phi,hostmtx->n*sizeof(double));
    MPI_Gatherv(phi_buf, topo_c.scount[topo_c.myid],MPI_DOUBLE, phi, topo_c.scount, topo_c.displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete []phi_buf;
}
#endif
void xsolver_free(){
    if(hstpre!=NULL){
        hstpre->FreeMatrix();
	delete hstpre;
 	hstpre=NULL;
    }
    if(pam[1]!=0&&hostmtxspmv!=NULL){
    	hostmtxspmv->FreeMatrix();
    	delete hostmtxspmv;
    	hostmtxspmv=NULL; 
    }
    if(pam[2]==1&&devicemtx!=NULL){
        devicemtx->FreeMatrix();
	delete devicemtx;
 	devicemtx=NULL;
	freegpu(precon);
    }
    if(xsol!=NULL){
	delete xsol;
	xsol=NULL;
    }
    if(hostmtx!=NULL){
	delete hostmtx;
	hostmtx=NULL;
    }
}
