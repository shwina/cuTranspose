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
#include "kernels_210.h"
 
/********************************************
 * Public functions                         *
 ********************************************/
__global__
void dev_transpose_210_ept1( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{

	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];
	
	int x_in, y, z_in,
	    x_out, z_out,
	    ind_in,
	    ind_out;
	
	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	z_in = ly + TILE_SIZE * by;

	y = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	z_out = lx + TILE_SIZE * by;


	ind_in = x_in + (y + z_in * np1) * np0;
	ind_out = z_out + (y + x_out * np1) * np2;

	if( x_in < np0 && z_in < np2 )
	{
			tile[lx][ly] = in[ind_in];
	}	

	__syncthreads();

	if( z_out < np2 && x_out < np0 )
	{
		out[ind_out] = tile[ly][lx];
	}

}

__global__
void dev_transpose_210_ept2( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{

	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];
	
	int x_in, y, z_in,
	    x_out, z_out,
	    ind_in,
	    ind_out;
	
	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;
		
	x_in = lx + TILE_SIZE * bx;
	z_in = ly + TILE_SIZE * by;

	y = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	z_out = lx + TILE_SIZE * by;

	ind_in = x_in + (y + z_in * np1) * np0;
	ind_out = z_out + (y + x_out * np1) * np2;

	if( x_in < np0 && z_in < np2 )
	{
		tile[lx][ly] = in[ind_in];
		if( z_in + 8 < np2 )
		{
			tile[lx][ly +  8] = in[ind_in +  8*np0*np1];
		}
	}
	
	__syncthreads();

	if( z_out < np2 && x_out < np0 )
	{
		out[ind_out] = tile[ly][lx];
		if( x_out + 8 < np0 )
		{
			out[ind_out +  8*np2*np1] = tile[ly + 8][lx];
		}
	}

}

__global__
void dev_transpose_210_ept4( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{

	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];
	
	int x_in, y, z_in,
	    x_out, z_out,
	    ind_in,
	    ind_out;
	
	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;
		
	x_in = lx + TILE_SIZE * bx;
	z_in = ly + TILE_SIZE * by;

	y = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	z_out = lx + TILE_SIZE * by;

	ind_in = x_in + (y + z_in * np1) * np0;
	ind_out = z_out + (y + x_out * np1) * np2;

	if( x_in < np0 && z_in < np2 )
	{
		tile[lx][ly] = in[ind_in];
		if( z_in + 4 < np2 )
		{
			tile[lx][ly +  4] = in[ind_in +  4*np0*np1];
			if( z_in + 8 < np2 )
			{
				tile[lx][ly +  8] = in[ind_in +  8*np0*np1];
				if( z_in + 12 < np2 )
				{
					tile[lx][ly + 12] = in[ind_in + 12*np0*np1];
				}
			}
		}
	}
	
	__syncthreads();
	
	if( z_out < np2 && x_out < np0 )
	{
		out[ind_out] = tile[ly][lx];
		if( x_out + 4 < np0 )
		{
			out[ind_out +  4*np2*np1] = tile[ly +  4][lx];
			if( x_out + 8 < np0 )
			{
				out[ind_out +  8*np2*np1] = tile[ly +  8][lx];
				if( x_out + 12 < np0 )
				{
					out[ind_out + 12*np2*np1] = tile[ly + 12][lx];
				}
			}
		}
	}
}

__global__
void dev_transpose_210_in_place( data_t* data,
                                 int     np0,
                                 int     np1 )
{
	__shared__ data_t inf_tile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ data_t sup_tile[TILE_SIZE][TILE_SIZE + 1];
	
	int x_inf, y, z_inf,
	    x_sup, z_sup,
	    ind_inf,
	    ind_sup;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	if( bx > by ) // Block in upper triangle
		return;

	x_inf = lx + TILE_SIZE * bx;
	z_inf = ly + TILE_SIZE * by;
	y = blockIdx.z;

	if( x_inf < np0 && z_inf < np0 )
	{
		ind_inf = x_inf + (y + z_inf * np1) * np0;
		inf_tile[lx][ly] = data[ind_inf];
	}

	x_sup = ly + TILE_SIZE * bx;
	z_sup = lx + TILE_SIZE * by;
	if( bx < by ) // Block in lower triangle
	{
		if( x_sup < np0 && z_sup < np0 )
		{
			ind_sup = z_sup + (y + x_sup * np1) * np0;
			sup_tile[lx][ly] = data[ind_sup];
		}
	}
	else // Block in diagonal
		ind_sup = ind_inf;

	__syncthreads();

	if( x_sup < np0 && z_sup < np0 )
	{
		data[ind_sup] = inf_tile[ly][lx];
	}

	if( bx < by )
	{
		if( x_inf < np0 && z_inf < np0 )
		{
			data[ind_inf] = sup_tile[ly][lx];
		}
	}
}
