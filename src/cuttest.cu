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
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include "sys/time.h"

#include "tools.h"
#include "cutranspose.h"

/********************************************
 * Macros                                   *
 ********************************************/
#define DEFAULT_RUNS 100
#define WARMUP_RUNS    10

/********************************************
 * Private function prototypes             *
 ********************************************/
static int check_transpose3d( const int*    size,
                              const int*    perm,
                              const data_t* a1,
                              const data_t* a2,
                              int           verbose );
static void print_usage( const char* name );

/********************************************
 * Main function                            *
 ********************************************/
int main( int argc, char* argv[] )
{
	int  c,
	     i, i_run,
	     num_runs = DEFAULT_RUNS,
	     elements_per_thread = 1,
	   * size,
	   * perm,
	     np,
	     verbose  = 0,
	     in_place = 0;
	data_t* in_array,
	      * out_array,
	      * dev_in_array,
	      * dev_out_array;

	unsigned long size_bytes;
	char* size_str = NULL,
	    * perm_str = NULL;
	struct dstat kernel_stat;
	struct timeval start, stop;
	float         kernel_time;

	/*
		Process the input arguments.
		The meaning of each argument can be obtained with -h option.
	*/
	opterr = 0;
	while( (c = getopt( argc, argv, "s:p:r:e:ivh" )) != -1 )
		switch( c )
		{
		case 's':
			size_str = optarg;
			break;
		case 'p':
			perm_str = optarg;
			break;
		case 'r':
			num_runs = atoi( optarg );
			break;
		case 'e':
			elements_per_thread = atoi( optarg );
			break;
		case 'v':
			verbose = 1;
			break;
		case 'i':
			in_place = 1;
			break;
		case 'h':
			print_usage( argv[0] );
			return 0;
		case '?':
			if( optopt == 's' || optopt == 'p' || optopt == 'r' || optopt == 'e' )
				fprintf( stderr, "Option -%c requires an argument.\n", optopt );
			else if ( isprint( optopt ) )
				fprintf( stderr, "Unknown option `-%c'.\n", optopt );
			else
				fprintf( stderr,
				         "Unknown option character `\\x%x'.\n",
				         optopt );
			print_usage( argv[0] );
			return 1;
		}

	// Check arguments.
	if( size_str == NULL )
	{
		fprintf( stderr, "The -s option is not set.\n" );
		print_usage( argv[0] );
		return 1;
	}
	if( perm_str == NULL )
	{
		fprintf( stderr, "The -p option is not set.\n" );
		print_usage( argv[0] );
		return 1;
	}

	size = split_ints( 3, size_str, "x" );
	if( size == NULL )
	{
		fprintf( stderr, "Incorrect value for the -s option: %s\n", size_str );
		print_usage( argv[0] );
		return 1;
	}
	if( size[0] < 2 || size[1] < 2 || size[2] < 2 )
	{
		fprintf( stderr, "-s values must be greater than 1.\n" );
		print_usage( argv[0] );
		return 1;
	}

	perm = split_ints( 3, perm_str, "," );
	if( !is_permutation( perm ) )
	{
		fprintf( stderr, "Incorrect value for the -p option: %s\n", perm_str );
		print_usage( argv[0] );
		return 1;
	}

	if( num_runs < 1 )
	{
		fprintf( stderr, "-r option must be set to a positive value.\n" );
		print_usage( argv[0] );
		return 1;
	}

	if( in_place && elements_per_thread != 1 )
		fprintf( stderr, "Warning: -e option is ignored if -i is set.\n" );

	dstat_init( &kernel_stat );
	size_bytes = size[0] * size[1] * size[2] * sizeof( data_t );
	if( verbose )
	{
		printf( "Transposing a %dx%dx%d array (%ld MB) ",
		        size[0],
		        size[1],
		        size[2],
		        size_bytes >> 20 );
		if( in_place )
			printf( "in-place.\n" );
		else
			printf( "out-of-place.\n" );
		printf( "\tElements-per-thread = %d.\n", elements_per_thread );
		printf( "\tTile size = %d.\n", TILE_SIZE );
		printf( "\tBrick size = %d.\n", BRICK_SIZE );
		printf( "\tElement size: %zu bytes.\n", sizeof( data_t ) );
	}

	// Memory allocation (CPU and GPU).
	np = size[0] * size[1] * size[2];
	in_array =  (data_t*) malloc( np * sizeof( data_t ) );
	out_array =  (data_t*) malloc( np * sizeof( data_t ) );
	HANDLE_ERROR( cudaMalloc( &dev_in_array, np * sizeof( data_t ) ) );
	if( in_place )
		dev_out_array = dev_in_array;
	else
	{
		HANDLE_ERROR( cudaMalloc( &dev_out_array, np * sizeof( data_t ) ) );
	}

	// Data initialization.
	for ( i = 0; i < np; i++ )
	{
#ifdef USE_COMPLEX
		in_array[i] = i + i * I;
#else
		in_array[i] = i + 0.0;
#endif
	}
	HANDLE_ERROR( cudaMemcpy( dev_in_array,
	                          in_array,
	                          np * sizeof( data_t ),
	                          cudaMemcpyHostToDevice ) );

	for( i_run = -WARMUP_RUNS; i_run < num_runs; i_run++ )
	{
		gettimeofday( &start, NULL );
		if( cut_transpose3d( dev_out_array,
		                     dev_in_array,
		                     size,
		                     perm,
		                     elements_per_thread ) < 0 )
		{
			fprintf( stderr, "Error while performing transpose.\n" );
			return 1;
		}
		HANDLE_ERROR( cudaDeviceSynchronize() );
		gettimeofday( &stop, NULL );
		kernel_time = TIME_DIFF( start, stop );
		if( i_run >= 0 )
			dstat_add( &kernel_stat, kernel_time );

		if( i_run == -WARMUP_RUNS )
		{
			cudaMemcpy( out_array, dev_out_array, np * sizeof( data_t ), cudaMemcpyDeviceToHost );
			if( !check_transpose3d( size, perm, in_array, out_array, verbose ) )
			{
				fprintf( stderr, "ERROR! The transposition was not correctly done.\n" );
				return -1;
			}
			free( out_array );
		}
	}
	printf("Kernel execution timing:\n\t");
	dstat_print_as_time(kernel_stat, 'u', 0 );
	printf( "\tTranspose speed (GB/s): %.0f/%.0f/%.0f (max/avg/min).\n",
	        2.0 * size_bytes / (1000.0 * kernel_stat.min),
	        2.0 * size_bytes / (1000.0 * kernel_stat.mean),
	        2.0 * size_bytes / (1000.0 * kernel_stat.max) );

	return 0;
}

/**
 * Checks whether an array is the 3D transpose of another one.
 * \param [in] size A vector with the size of every dimension.
 * \param [in] perm A permutation vector that sets the 3D transpose.
 * \param [in] a1 An array.
 * \param [in] a2 The presumed transpose of \p a1.
 * \return 1 if \p a2 is the 3D transpose of \p a1 as specified by \p perm, and
 * 0 otherwise.
 */
int check_transpose3d( const int*    size,
                       const int*    perm,
                       const data_t* a1,
                       const data_t* a2,
                       int           verbose )
{
	int coord[3],
	    psize[3],
	    ind1, ind2,
	    wrong = 0;

	psize[0] = size[perm[0]];
	psize[1] = size[perm[1]];
	psize[2] = size[perm[2]];
	ind1 = 0;
	for( coord[2] = 0; coord[2] < size[2]; coord[2]++ )
	{
		for( coord[1] = 0; coord[1] < size[1]; coord[1]++ )
		{
			for( coord[0] = 0; coord[0] < size[0]; coord[0]++ )
			{
				ind2 = coord[perm[0]] +
				       (coord[perm[1]] + coord[perm[2]] * psize[1] ) *
				       psize[0];
#ifdef USE_COMPLEX
				if( creal(a1[ind1]) != creal(a2[ind2]) ||
				    cimag(a1[ind1]) != cimag(a2[ind2]) )
#else
				if( a1[ind1] != a2[ind2] )
#endif
				{
					if( verbose )
					{
#ifdef USE_COMPLEX
						printf( "(%d, %d, %d) -> Original[%d] = %.f + %.fi"
						        "\tTranspose[%d] = %.f + %.fi\n",
						        coord[0], coord[1], coord[2],
						        ind1,
						        creal(a1[ind1]), cimag(a1[ind1]),
						        ind2,
						        creal(a2[ind2]), cimag(a2[ind2]) );
#else
						printf( "(%d, %d, %d) -> Original[%d] = %.f "
						        "\tTranspose[%d] = %.f\n",
						        coord[0], coord[1], coord[2],
						        ind1,
						        a1[ind1],
						        ind2,
						        a2[ind2] );
#endif
						wrong = 1;
					}
					else
						return 0;
				}
				ind1++;
			}
		}
	}
	if( verbose && wrong )
		return 0;
	return 1;
}

/**
 * Prints the arguments of the program.
 * \param [in] name The program name.
 */
static void print_usage( const char* name )
{
	fprintf( stderr,
	         "Program usage: %s -s <size> -p <permutation> [-r <number_of_runs>] "
	         "[-e elements-per-thread>] [-i] [-v] [-h]\n",
	         name );
	fprintf( stderr,
	         "\t-s <size> The size of the 3D array to be transposed. 3 values "
	         "separated with the 'x' character must be specified. If just one "
	         "value is specified, it is used in the 3 dimensions.\n" );
	fprintf( stderr,
	         "\t-p <permutation> A comma-separated permutation of 0, 1 and 2. "
	         "It defines the transposition to be performed. The dimension that "
	         "must be stored in contiguous positions in memory must be set as "
	         "the first dimension.\n" );
	fprintf( stderr,
	         "\t[-r <num_runs>] The number of times the transposition must be "
	         "repeated. Actually, %d more warm-up runs are executed, whose "
	         "execution times is ignored. The default value is %d.\n",
	         WARMUP_RUNS, DEFAULT_RUNS );
	fprintf( stderr,
	         "\t-e <elements-per-thread> The number of elements per thread. "
	         "It defines the number of elements per thread (default value: 1).\n" );
	fprintf( stderr,
	         "\t-v Verbose output.\n" );
	fprintf( stderr,
	          "\t-i Run the transpose in-place instead of out-of-place.\n" );
	fprintf( stderr, "\t[-h] Show this help.\n" );

	return;
}

