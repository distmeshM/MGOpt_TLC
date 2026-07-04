cmake_minimum_required(VERSION 3.18...3.25)
project(IntrinsicProlongation2 LANGUAGES CXX)

# ============================================================
# 1. Resolve the relative extern/ path to an absolute path
# ============================================================
get_filename_component(PROJECT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

set(EXTERN_DIR "${PROJECT_ROOT}/extern")

if(NOT EXISTS "${EXTERN_DIR}")
    message(FATAL_ERROR
        "Cannot find extern/ directory:\n"
        "  ${EXTERN_DIR}\n"
        "Please ensure extern/ exists and contains include/, lib/, bin/."
    )
endif()

# ============================================================
# 2. MATLAB
# ============================================================
find_package(Matlab REQUIRED)

# ========== Boost ==========
set(Boost_ROOT "${EXTERN_DIR}")
set(Boost_NO_SYSTEM_PATHS ON)
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.72 REQUIRED)

# ========== GMP ==========
set(GMP_INCLUDE_DIR "${EXTERN_DIR}/include")
set(GMP_LIBRARY_DIR "${EXTERN_DIR}/lib")
set(GMP_LIBRARIES
    debug "${EXTERN_DIR}/debug/lib/gmp.lib"
    optimized "${EXTERN_DIR}/lib/gmp.lib"
)

# ========== MPFR ==========
set(MPFR_INCLUDE_DIR "${EXTERN_DIR}/include")
set(MPFR_LIBRARY_DIR "${EXTERN_DIR}/lib")
set(MPFR_LIBRARIES
    debug "${EXTERN_DIR}/debug/lib/mpfr.lib"
    optimized "${EXTERN_DIR}/lib/mpfr.lib"
)

# ========== CGAL ==========
set(CGAL_DIR "${EXTERN_DIR}/share/cgal")
find_package(CGAL REQUIRED)

# ============================================================
# 7. libigl
# ============================================================
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

option(LIBIGL_WITH_OPENGL     "Enable OpenGL support in libigl"     ON)
option(LIBIGL_WITH_OPENGL_GLFW "Enable GLFW support in libigl"      ON)

set(LIBIGL_FIND_COMPONENTS core)
if(LIBIGL_WITH_OPENGL)
    list(APPEND LIBIGL_FIND_COMPONENTS opengl)
endif()
if(LIBIGL_WITH_OPENGL_GLFW)
    list(APPEND LIBIGL_FIND_COMPONENTS opengl_glfw)
endif()

# ✅ Disable libigl's automatic Eigen download (strongly recommended)
set(IGL_DOWNLOAD_EIGEN OFF CACHE BOOL "" FORCE)
set(EIGEN_DIR "${EXTERN_DIR}/eigen" CACHE PATH "")

find_package(LIBIGL REQUIRED COMPONENTS ${LIBIGL_FIND_COMPONENTS})

# ============================================================
# 8. MEX source files
# ============================================================
file(GLOB SRCFILES
    *.cpp
    ../src/*.cpp
)

matlab_add_mex(
    NAME ${PROJECT_NAME}
    SRC ${SRCFILES}
)

# ============================================================
# 9. Include directories (modern CMake)
# ============================================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ../src
)

# ============================================================
# 10. Link libraries (fully target-based)
# ============================================================
target_link_libraries(${PROJECT_NAME}
    igl::core
    $<$<BOOL:${LIBIGL_WITH_OPENGL}>:igl::opengl>
    $<$<BOOL:${LIBIGL_WITH_OPENGL_GLFW}>:igl::opengl_glfw>
    CGAL::CGAL
    ${GMP_LIBRARIES}
    ${MPFR_LIBRARIES}
    Boost::headers
)

# ============================================================
# 11. Copy DLLs next to the MEX output (required by MATLAB runtime)
# ============================================================
foreach(dll
    "${EXTERN_DIR}/bin/gmp.dll"
    "${EXTERN_DIR}/bin/mpfr.dll"
)
    if(EXISTS "${dll}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copying ${dll} to MEX output directory"
        )
    else()
        message(WARNING "DLL not found: ${dll}")
    endif()
endforeach()

# ============================================================
# 12. Unified output directory
# ============================================================
set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)