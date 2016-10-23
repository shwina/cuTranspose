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
#include "kernels_012.h"

/********************************************
 * Public functions                         *
 ********************************************/
__global__
void dev_copy( data_t*       out,
               const data_t* in,
               int           np0,
               int           np1,
               int           elements_per_thread )
{
	int x, y, z,
	    ind,
	    i;

	x = threadIdx.x + TILE_SIZE * blockIdx.x;
	y = threadIdx.y + TILE_SIZE * blockIdx.y;
	z = blockIdx.z;

	if( x >= np0 || y >= np1 )
		return;

	ind = x + (y + z * np1) * np0;

	for( i = 0;
	     i < TILE_SIZE && y + i < np1;
	     i += TILE_SIZE / elements_per_thread )
	{
		out[ind + i*np0] = in[ind + i*np0];
	}
}
