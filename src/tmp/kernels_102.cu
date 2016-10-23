/*  This file is part of cuTranspose.

    cuTranspose is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    cuTranspose is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with cuTranspose.  If not, see <http://www.gnu.org/licenses/>.

    Copyright 2016 Ibai Gurrutxaga, Javier Muguerza, Jose L. Jodra.
*/

/********************************************
 * Includes                                 *
 ********************************************/
#include "cutranspose.h"
#include "kernels_102.h"

/********************************************
 * Public functions                         *
 ********************************************/
/**
 * Kernel that performs a 1,0,2 transpose (order inversion) out of place.
 *
 * The grid of threads must be a 3-dimensional grid of 2-dimensional blocks.
 * The x dimension must match the innermost dimension of the input grid, and
 * the y dimension must match the innermost dimension of the transposed grid.
 * \param [out] out The output array.
 * \param [in] in The input array.
 * \param [in] np0 The size of the first dimension of the input array.
 * \param [in] np1 The size of the second dimension of the input array.
 * \param [in] np2 The size of the third dimension of the input array.
 */
 __global__
 void dev_transpose_102_ept1( data_t*       out,
                              const data_t* in,
                              int           np0,
                              int           np1,
                              int           np2 )
{
	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x_in, y_in, z,
	    x_out, y_out,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	y_in = ly + TILE_SIZE * by;

	z = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	y_out = lx + TILE_SIZE * by;


	ind_in = x_in + (y_in + z * np1) * np0;
	ind_out = y_out + (x_out + z * np0) * np1;

	if( x_in < np0 && y_in < np1 )
	{
		tile[lx][ly] = in[ind_in];
	}

	__syncthreads();

	if( x_out < np0 && y_out < np1 )
	{
		out[ind_out] = tile[ly][lx];
	}
}

/**
 * Kernel that performs a 1,0,2 transpose (order inversion) out of place.
 *
 * The grid of threads must be a 3-dimensional grid of 2-dimensional blocks.
 * The x dimension must match the innermost dimension of the input grid, and
 * the y dimension must match the innermost dimension of the transposed grid.
 * \param [out] out The output array.
 * \param [in] in The input array.
 * \param [in] np0 The size of the first dimension of the input array.
 * \param [in] np1 The size of the second dimension of the input array.
 * \param [in] np2 The size of the third dimension of the input array.
 */
__global__
void dev_transpose_102_ept2( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{
	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x_in, y_in, z,
	    x_out, y_out,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	y_in = ly + TILE_SIZE * by;

	z = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	y_out = lx + TILE_SIZE * by;


	ind_in = x_in + (y_in + z * np1) * np0;
	ind_out = y_out + (x_out + z * np0) * np1;

	if( x_in < np0 && y_in < np1 )
	{
		tile[lx][ly] = in[ind_in];
		if( y_in + 8 < np1 )
		{
			tile[lx][ly +  8] = in[ind_in +  8*np0];
		}
	}

	__syncthreads();

	if( x_out < np0 && y_out < np1 )
	{
		out[ind_out] = tile[ly][lx];
		if( x_out + 8 < np0 )
		{
			out[ind_out +  8*np1] = tile[ly + 8][lx];
		}
	}
}

/**
 * Kernel that performs a 1,0,2 transpose (order inversion) out of place.
 *
 * The grid of threads must be a 3-dimensional grid of 2-dimensional blocks.
 * The x dimension must match the innermost dimension of the input grid, and
 * the y dimension must match the innermost dimension of the transposed grid.
 * \param [out] out The output array.
 * \param [in] in The input array.
 * \param [in] np0 The size of the first dimension of the input array.
 * \param [in] np1 The size of the second dimension of the input array.
 * \param [in] np2 The size of the third dimension of the input array.
 */
__global__
void dev_transpose_102_ept4( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{
	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x_in, y_in, z,
	    x_out, y_out,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	y_in = ly + TILE_SIZE * by;

	z = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	y_out = lx + TILE_SIZE * by;


	ind_in = x_in + (y_in + z * np1) * np0;
	ind_out = y_out + (x_out + z * np0) * np1;

	if( x_in < np0 && y_in < np1 )
	{
		tile[lx][ly] = in[ind_in];
		if( y_in + 4 < np1 )
		{
			tile[lx][ly +  4] = in[ind_in +  4*np0];
			if( y_in + 8 < np1 )
			{
				tile[lx][ly +  8] = in[ind_in +  8*np0];
				if( y_in + 12 < np1 )
				{
					tile[lx][ly +  12] = in[ind_in +  12*np0];
				}
			}
		}
	}

	__syncthreads();

	if( x_out < np0 && y_out < np1 )
	{
		out[ind_out] = tile[ly][lx];
		if( x_out + 4 < np0 )
		{
			out[ind_out +  4*np1] = tile[ly + 4][lx];
			if( x_out + 8 < np0 )
			{
				out[ind_out +  8*np1] = tile[ly + 8][lx];
				if( x_out + 12 < np0 )
				{
					out[ind_out +  12*np1] = tile[ly + 12][lx];
				}
			}
		}
	}
}

/**
 * Kernel that performs a 1,0,2 transpose (order inversion) in place.
 *
 * The grid of threads must be a 3-dimensional grid of 2-dimensional blocks.
 * The x dimension must match the innermost dimension of the input grid, and
 * the y dimension must match the innermost dimension of the transposed grid.
 * \param [in, out] in The data array.
 * \param [in] np0 The size of the first and last dimensions of the input array.
 * \param [in] np2 The size of the third dimension of the input array.
 */
__global__
void dev_transpose_102_in_place( data_t* data,
                                 int     np0,
                                 int     np2 )
{
	__shared__ data_t inf_tile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ data_t sup_tile[TILE_SIZE][TILE_SIZE + 1];

	int x_inf, y_inf, z,
	    x_sup, y_sup,
	    ind_inf,
	    ind_sup;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	if( bx > by ) // Block in upper triangle
		return;

	x_inf = lx + TILE_SIZE * bx;
	y_inf = ly + TILE_SIZE * by;
	z = blockIdx.z;

	if( x_inf < np0 && y_inf < np0 )
	{
		ind_inf = x_inf + (y_inf + z * np0) * np0;
		inf_tile[lx][ly] = data[ind_inf];
	}

	x_sup = ly + TILE_SIZE * bx;
	y_sup = lx + TILE_SIZE * by;
	if( bx < by ) // Block in lower triangle
	{
		if( x_sup < np0 && y_sup < np0 )
		{
			ind_sup = y_sup + (x_sup + z * np0) * np0;
			sup_tile[lx][ly] = data[ind_sup];
		}
	}
	else // Block in diagonal
		ind_sup = ind_inf;

	__syncthreads();

	if( x_sup < np0 && y_sup < np0 )
	{
		data[ind_sup] = inf_tile[ly][lx];
	}

	if( bx < by )
	{
		if( x_inf < np0 && y_inf < np0 )
		{
			data[ind_inf] = sup_tile[ly][lx];
		}
	}
}
