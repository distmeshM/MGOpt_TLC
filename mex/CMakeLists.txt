cmake_minimum_required(VERSION 3.12)
project(en_grad_hessian)


find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER MX_LIBRARY)


find_package(OpenMP REQUIRED)


include_directories("eigen-3.4.0")
include_directories(include)




matlab_add_mex(
    NAME en_grad_hessian
    SRC en_grad_hessian.cpp
    LINK_TO OpenMP::OpenMP_CXX
)


matlab_add_mex(
    NAME en_grad_hessian2
    SRC en_grad_hessian2.cpp
    LINK_TO OpenMP::OpenMP_CXX
)


matlab_add_mex(
    NAME en_grad_diag
    SRC en_grad_diag.cpp
    LINK_TO OpenMP::OpenMP_CXX
)


matlab_add_mex(
    NAME en_grad_diag2
    SRC en_grad_diag2.cpp
    LINK_TO OpenMP::OpenMP_CXX
)

matlab_add_mex(
    NAME accumarray2
    SRC accumarray2.cpp
    LINK_TO OpenMP::OpenMP_CXX
)

matlab_add_mex(
    NAME en_grad_diag3
    SRC en_grad_diag3.cpp
    LINK_TO OpenMP::OpenMP_CXX
)

matlab_add_mex(
    NAME blkSolve
    SRC blkSolve.cpp
    LINK_TO OpenMP::OpenMP_CXX
)

matlab_add_mex(
    NAME SpMV
    SRC SpMV.cpp
    LINK_TO OpenMP::OpenMP_CXX
)

matlab_add_mex(
    NAME en4d
    SRC en4d.cpp
    LINK_TO OpenMP::OpenMP_CXX
)

matlab_add_mex(
    NAME enPre
    SRC enPre.cpp
    LINK_TO OpenMP::OpenMP_CXX
)

matlab_add_mex(
    NAME en4dwithPre
    SRC en4dwithPre.cpp
    LINK_TO OpenMP::OpenMP_CXX
)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})