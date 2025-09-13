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
module load compiler/intel/2017.5.239
cmake .. -DCMAKE_INSTALL_PREFIX=/you/install/dir/xsolver -DXSOLVER_BACKEND="HIP" -DHAVE_MPI
make && make install
```