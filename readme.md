# compile and install
## intel
- compile for intel with no mpi
```bash
source $HOME/local/intel/oneapi/setvars.sh

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/you/install/dir/xsolver 
make && make install
```
- compile for intel with mpi
```bash
source $HOME/local/intel/oneapi/setvars.sh

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/you/install/dir/xsolver  -DHAVE_MPI=ON
make && make install
``` 

## cuda
- compile for cuda with no mpi
```bash
source $HOME/local/intel/oneapi/setvars.sh

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/you/install/dir/xsolver  -DXSOLVER_BACKEND="CUDA"
make && make install
```
- compile for cuda with mpi
```bash
source $HOME/local/intel/oneapi/setvars.sh

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/you/install/dir/xsolver  -DXSOLVER_BACKEND="CUDA" -DHAVE_MPI=ON
make && make install
```

## hip
- compile for hip with no mpi
```bash
module load compiler/intel/2017.5.239
cmake .. -DCMAKE_INSTALL_PREFIX=/you/install/dir/xsolver -DXSOLVER_BACKEND="HIP" 
make && make install
```

- compile for hip with mpi
```bash
# 环境
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/rocm/dtk/22.10.1
module load compiler/intel/2017.5.239
module load compiler/cmake/3.24.1

cmake .. -DCMAKE_INSTALL_PREFIX=/you/install/dir/xsolver -DXSOLVER_BACKEND="HIP" -DHAVE_MPI=ON
make && make install
```