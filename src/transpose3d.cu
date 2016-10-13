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
#include <stdio.h>
#include <stdlib.h>

#include "cutranspose.h"
#include "kernels_012.h"
#include "kernels_021.h"
#include "kernels_102.h"
#include "kernels_120.h"
#include "kernels_201.h"
#include "kernels_210.h"

/********************************************
 * Private function prototypes              *
 ********************************************/
static void set_grid_dims( const int* size,
                           int        d2,
                           dim3*      block_size,
                           dim3*      num_blocks,
                           int        elements_per_thread);
static void set_grid_dims_cube( const int* size,
                                dim3*      block_size,
                                dim3*      num_blocks,
                                int        elements_per_thread );
static int valid_parameters( int        in_place,
                             const int* size,
                             const int* permutation,
                             int        elements_per_thread );

/********************************************
 * Exported functions                         *
 ********************************************/
/**
 * Perform a transposition of a 3D array in a GPU. The first dimension
 * is considered the innermost one. If the \p input and \ output pointers
 * are equal an in-place transposition will be performed.
 *
 * \param[out] output A pointer to the allocated memory space in the device
 * where the transposed array will be stored.
 * \param[in] input A pointer to device memory where the input data is stored.
 * \param[in] size A 3 element array with the size of the input data
 * on each dimension.
 * \param[in] permutation An array with a permutation specifying the
 * transposition to be performed.
 * \param[in] elements_per_thread The number of elements that a GPU thread must transpose.
 * It will be ignored if \p in_place is true.
 * \return 0 on success and -1 otherwise.
 */
int cut_transpose3d( data_t*       output,
                     const data_t* input,
                     const int*    size,
                     const int*    permutation,
                     int           elements_per_thread )

{
	dim3 num_blocks,
	     block_size;
	int in_place = output == input;

	if( !valid_parameters( in_place, size, permutation, elements_per_thread ) )
		return -1;

	if(( permutation[0] == 1 && permutation[1] == 2 && permutation[2] == 0 && in_place ) ||
	   ( permutation[0] == 2 && permutation[1] == 0 && permutation[2] == 1 && in_place ))
		set_grid_dims_cube( size,
		                    &block_size,
		                    &num_blocks,
		                    elements_per_thread );
	else
		set_grid_dims( size,
		               permutation[0],
		               &block_size,
		               &num_blocks,
		               elements_per_thread );

	if( permutation[0] == 0 && permutation[1] == 1 && permutation[2] == 2 )
	{
			dev_copy<<< num_blocks, block_size >>>( output,
			                                        input,
			                                        size[0],
			                                        size[1], 
			                                        elements_per_thread );
	}
	else if( permutation[0] == 0 && permutation[1] == 2 && permutation[2] == 1 )
	{
		if( in_place )
		{
			if( size[1] != size[2] )
			{
				fprintf( stderr, "This in-place transposition requires equal dimensions for "
								 "the second and third dimensions.\n" );
				return -1;
			}
			dev_transpose_021_in_place<<< num_blocks, block_size >>>( output,
			                                                          size[0],
			                                                          size[1] );
		}
		else {
			switch(elements_per_thread)
			{
			case 1:
				dev_transpose_021_ept1<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 2:
				dev_transpose_021_ept2<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 4:
				dev_transpose_021_ept4<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			}
		}
	}
	else if( permutation[0] == 1 && permutation[1] == 0 && permutation[2] == 2 )
	{
		if( in_place )
		{
			if( size[0] != size[1] )
			{
				fprintf( stderr, "The 102 in-place transposition requires equal dimensions for "
								 "the first and second dimensions.\n" );
				return -1;
			}
			dev_transpose_102_in_place<<< num_blocks, block_size >>>( output,
                                                                      size[0],
                                                                      size[2] );
		}
		else {
			switch(elements_per_thread)
			{
			case 1:
				dev_transpose_102_ept1<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 2:
				dev_transpose_102_ept2<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 4:
				dev_transpose_102_ept4<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			}
		}
	}
	else if( permutation[0] == 1 && permutation[1] == 2 && permutation[2] == 0 )
	{
		if( in_place )
		{
			if( size[0] != size[1] || size[0] != size[2] )
			{
				fprintf( stderr, "The 120 in-place transposition requires a cubic input.\n" );
				return -1;
			}
			dev_transpose_120_in_place<<< num_blocks, block_size >>>( output,
			                                                          size[0] );
		}
		else {
			switch(elements_per_thread)
			{
			case 1:
				dev_transpose_120_ept1<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 2:
				dev_transpose_120_ept2<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 4:
				dev_transpose_120_ept4<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			}
		}
	}
	else if( permutation[0] == 2 && permutation[1] == 0 && permutation[2] == 1 )
	{
		if( in_place )
		{
			if( size[0] != size[1] || size[0] != size[2] )
			{
				fprintf( stderr, "The 201 in-place transposition requires a cubic input.\n" );
				return -1;
			}
			dev_transpose_201_in_place<<< num_blocks, block_size >>>( output,
			                                                          size[0] );
		}
		else {
			switch(elements_per_thread)
			{
			case 1:
				dev_transpose_201_ept1<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 2:
				dev_transpose_201_ept2<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 4:
				dev_transpose_201_ept4<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			}
		}
	}
	else if( permutation[0] == 2 && permutation[1] == 1 && permutation[2] == 0 )
	{
		if( in_place )
		{
			if( size[0] != size[2] )
			{
				fprintf( stderr, "The 210 in-place transposition requires equal dimensions for "
								 "the first and third dimensions.\n" );
				return -1;
			}
			dev_transpose_210_in_place<<< num_blocks, block_size >>>( output,
			                                                          size[0],
			                                                          size[1] );
		}
		else {
			switch(elements_per_thread)
			{
			case 1:
				dev_transpose_210_ept1<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 2:
				dev_transpose_210_ept2<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			case 4:
				dev_transpose_210_ept4<<< num_blocks, block_size >>>( output,
				                                                      input,
				                                                      size[0],
				                                                      size[1],
				                                                      size[2]);
				break;
			}
		}
	}
	
	return 0;
}

/********************************************
 * Private functions                        *
 ********************************************/
static void set_grid_dims( const int* size,
                           int        d2,
                           dim3*      block_size,
                           dim3*      num_blocks,
                           int        elements_per_thread)
{
	block_size->x = TILE_SIZE;
	block_size->y = TILE_SIZE / elements_per_thread;
	block_size->z = 1;
	num_blocks->x = size[0] / TILE_SIZE;
	if( size[0] % TILE_SIZE != 0 )
		num_blocks->x++;
	if( d2 == 0 )
		d2 = 1;
	num_blocks->y = size[d2] / TILE_SIZE;
	if( size[d2] % TILE_SIZE != 0 )
		num_blocks->y++;
	num_blocks->z = size[(d2 == 1) ? 2 : 1];
}

static void set_grid_dims_cube( const int* size,
                                dim3*      block_size,
                                dim3*      num_blocks,
                                int        elements_per_thread )
{
	block_size->x = BRICK_SIZE;
	block_size->y = BRICK_SIZE;
	block_size->z = BRICK_SIZE / elements_per_thread;
	num_blocks->x = size[0] / BRICK_SIZE;
	if( size[0] % BRICK_SIZE != 0 )
		num_blocks->x++;
	num_blocks->y = size[1] / BRICK_SIZE;
	if( size[1] % BRICK_SIZE != 0 )
		num_blocks->y++;
	num_blocks->z = size[2] / BRICK_SIZE;
	if( size[2] % BRICK_SIZE != 0 )
		num_blocks->z++;
}

static int valid_parameters( int        in_place,
                             const int* size,
                             const int* permutation,
                             int        elements_per_thread )
{
	int dims[] = { 0, 0, 0 },
        i;

	if( in_place && elements_per_thread != 1 )
		return 0;
	if( size == NULL || permutation == NULL )
		return 0;
	if( size[0] < 2 || size[1] < 2 || size[2] < 2 )
		return 0;

	for( i = 0; i < 3; i++ )
	{
		if( permutation[i] < 0 || permutation[i] > 2 )
			return 0;
		else if( dims[permutation[i]] == 1 )
			return 0;
		else
			dims[permutation[i]] = 1;
	}

	return 1;
}
