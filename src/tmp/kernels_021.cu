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
#include "kernels_021.h"

/********************************************
 * Public functions                         *
 ********************************************/
__global__
void dev_transpose_021_ept1( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{
	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x, y, z,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x = lx + TILE_SIZE * bx;
	y = ly + TILE_SIZE * by;
	z = blockIdx.z;

	ind_in = x + (y + z * np1) * np0;
	ind_out = x + (z + y * np2) * np0;

	if( x < np0 && y < np1 )
	{
		tile[lx][ly] = in[ind_in];
	}

	__syncthreads();

	if( x < np0 && y < np1	 )
	{
		out[ind_out] = tile[lx][ly];
	}
}

__global__
void dev_transpose_021_ept2( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{
	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x, y, z,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x = lx + TILE_SIZE * bx;
	y = ly + TILE_SIZE * by;
	z = blockIdx.z;

	ind_in = x + (y + z * np1) * np0;
	ind_out = x + (z + y * np2) * np0;

	if( x < np0 && y < np1 )
	{
		tile[lx][ly] = in[ind_in];
		if( y + 8 < np1 )
		{
			tile[lx][ly +  8] = in[ind_in +  8*np0];
		}
	}

	__syncthreads();

	if( x < np0 && y < np1 )
	{
		out[ind_out] = tile[lx][ly];
		if( y + 8 < np1 )
		{
			out[ind_out +  8*np0*np2] = tile[lx][ly + 8];
		}
	}
}

__global__
void dev_transpose_021_ept4( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{
__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x, y, z,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x = lx + TILE_SIZE * bx;
	y = ly + TILE_SIZE * by;
	z = blockIdx.z;

	ind_in = x + (y + z * np1) * np0;
	ind_out = x + (z + y * np2) * np0;

	if( x < np0 && y < np1 )
	{
		tile[lx][ly] = in[ind_in];
		if( y + 4 < np1 )
		{
			tile[lx][ly +  4] = in[ind_in +  4*np0];
			if( y + 8 < np1 )
			{
				tile[lx][ly +  8] = in[ind_in +  8*np0];
				if( y + 12 < np1 )
				{
					tile[lx][ly +  12] = in[ind_in +  12*np0];
				}
			}
		}
	}

	__syncthreads();

	if( x < np0 && y < np1 )
	{
		out[ind_out] = tile[lx][ly];
		if( y + 4 < np1 )
		{
			out[ind_out +  4*np0*np2] = tile[lx][ly + 4];
			if( y + 8 < np1 )
			{
				out[ind_out +  8*np0*np2] = tile[lx][ly + 8];
				if( y + 12 < np1 )
				{
					out[ind_out +  12*np0*np2] = tile[lx][ly + 12];
				}
			}
		}
	}
}

__global__
void dev_transpose_021_in_place( data_t* data,
                                 int     np0,
                                 int     np1 )
{
	__shared__ data_t inf_tile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ data_t sup_tile[TILE_SIZE][TILE_SIZE + 1];

	int x, y, z,
	    ind_inf,
	    ind_sup;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	z = blockIdx.z;
	y = ly + TILE_SIZE * by;

	if( z > y ) // Thread in diagonal or upper triangle
		return;

	x = lx + TILE_SIZE * bx;

	if( x < np0 && y < np1 )
	{
		ind_inf = x + (y + z * np1) * np0;
		ind_sup = x + (z + y * np1) * np0;
		inf_tile[lx][ly] = data[ind_inf];
		sup_tile[lx][ly] = data[ind_sup];
	}

	__syncthreads();

	if( x < np0 && y < np1 )
	{
		data[ind_sup] = inf_tile[lx][ly];
		data[ind_inf] = sup_tile[lx][ly];
	}
}
