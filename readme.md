# compile and install
## intel
- compile for intel with no mpi
```bash
source $HOME/local/intel/oneapi/setvars.sh

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/staff/tandongdong/workspace/openfoam/xsolver/build/xsolver
make && make install
```
- compile for intel with mpi
```bash
source $HOME/local/intel/oneapi/setvars.sh

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/staff/tandongdong/workspace/openfoam/xsolver/build/xsolver -DHAVE_MPI=ON
make && make install
``` 

## cuda
- compile for cuda with no mpi
```bash
source $HOME/local/intel/oneapi/setvars.sh

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/staff/tandongdong/workspace/openfoam/xsolver/build/xsolver -DXSOLVER_BACKEND="CUDA"
make && make install
```
- compile for cuda with mpi
```bash
source $HOME/local/intel/oneapi/setvars.sh

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/staff/tandongdong/workspace/openfoam/xsolver/build/xsolver -DXSOLVER_BACKEND="CUDA" -DHAVE_MPI=ON
make && make install
```