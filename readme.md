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
make VERBOSE=1
make && make install
``` 