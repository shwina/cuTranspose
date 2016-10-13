#  This file is part of cuTranspose.
#
#   cuTranspose is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   cuTranspose is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with cuTranspose.  If not, see <http://www.gnu.org/licenses/>.
#
#   Copyright 2016 Ibai Gurrutxaga, Javier Muguerza, Jose L. Jodra.

cmake_minimum_required( VERSION 2.8 )
project( cuTranspose )

find_package( CUDA REQUIRED )

set( LIB_NAME "cuTranspose" )

option( CUT_USE_COMPLEX "Use double complex data." ON )
option( CUT_SINGLE_PRECISION "Use single precision real numbers." OFF )
set( CUT_TILE_SIZE 16 CACHE STRING "The tile size." )
set( CUT_BRICK_SIZE 8 CACHE STRING "The brick size." )

if( CUT_USE_COMPLEX )
	set( USE_COMPLEX 1 )
endif( CUT_USE_COMPLEX )
if( CUT_SINGLE_PRECISION )
	set( FLOAT 1 )
endif( CUT_SINGLE_PRECISION )

set( CMAKE_C_FLAGS_DEBUG "-Wall -g -O0 -DDEBUG" )
set( CMAKE_C_FLAGS_RELEASE "-Wall -O3" )
set( CUDA_NVCC_FLAGS_DEBUG -g -G -O0 -DDEBUG -gencode arch=compute_20,code=compute_20 )
set( CUDA_NVCC_FLAGS_RELEASE -O3 -gencode arch=compute_20,code=compute_20 )

set( SRC
     src/transpose3d.cu
     src/kernels_012.cu
     src/kernels_021.cu
     src/kernels_102.cu
     src/kernels_120.cu
     src/kernels_201.cu
     src/kernels_210.cu )

configure_file( ${PROJECT_SOURCE_DIR}/include/cutranspose.h.in cutranspose.h )
include_directories( ${PROJECT_BINARY_DIR} )
cuda_add_library( ${LIB_NAME} ${SRC} )
cuda_add_executable( cuttest test/cuttest.cu test/tools.c )
target_link_libraries( cuttest ${LIB_NAME} )
