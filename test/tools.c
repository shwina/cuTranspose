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
#include <math.h>

#include "tools.h"

/********************************************
 * Public functions                         *
 ********************************************/
/**
 * Splits a string represented as integers separated by sep. If just an integer is
 * found, it returns an array with \n repetitions of it.
 * \param[in] n The number of values.
 * \param[in] str The string to be split.
 * \return An array with \p n integers.
 */
int* split_ints( int n, const char* str, const char* sep )
{
	int* res,
	     i_token;
	char* token,
	    * _size_str;

	// Copy the input string to a non const string, so it can be used in strtok.
	_size_str = ( char* ) malloc( strlen( str ) );
	strcpy( _size_str, str );

	res = ( int* )malloc( n * sizeof( int ) );
	// Parse size string
	for( i_token = 0, token = strtok( _size_str, sep );
	     token != NULL;
	     token = strtok( NULL, sep ), i_token++ )
	{
		if( i_token == n )
		{
			fprintf( stderr,
			         "Too many size values specified. Just %d expected.\n",
			         n );
			return NULL;
		}
		res[i_token] = atoi( token );

	}
	if( i_token == 1 )
	{
		for( ; i_token < n; i_token++ )
			res[i_token] = res[0];
	}
	else if( i_token < n )
	{
		fprintf( stderr,
		         "%d size values expected and just %d found.\n",
		         n,
		         i_token );
		return NULL;
	}

	return res;
}

/**
 * Checks whether an integer array of size 3 has the 0, 1 and 2 values.
 *
 * \param[in] perm The input array.
 * \return 1 if the array is composed of the 0, 1 and 2 values, and 0 otherwise.
 */
int is_permutation( int* perm )
{
	int dims[] = { 0, 0, 0 },
	    i;

	if( perm == NULL )
		return 0;

	for( i = 0; i < 3; i++ )
	{
		if( perm[i] < 0 || perm[i] > 2 )
			return 0;
		else if( dims[perm[i]] == 1 )
			return 0;
		else
			dims[perm[i]] = 1;
	}

	return 1;
}

/**
 * Initializes a double stat structure. It can be used to compute minimum,
 * maximum, mean and standard deviations.
 * \param[out] stat The structure to be initialized.
 */
void dstat_init(struct dstat* stat)
{
	stat->n = 0;
	stat->min = HUGE_VAL;
	stat->max = -HUGE_VAL;
	stat->mean = stat->stddev = stat->q = 0.0;
}

/**
 * Adds a new element to a double stat structure.
 * \param[in,out] stat The stat structure.
 * \param[in] value The new element.
 */
void dstat_add(struct dstat* stat, double value)
{
	double new_mean;

	stat->n++;
	if( value < stat->min )
		stat->min = value;
	if( value > stat->max )
		stat->max = value;
	new_mean = stat->mean + ( value - stat->mean )/stat->n;
	stat->q += ( value - stat->mean ) * ( value - new_mean );
	stat->mean = new_mean;
	if( stat->n > 1 )
		stat->stddev = stat->q/( stat->n - 1 );
}

/**
 * Prints a set considering it a set of time elapses in microseconds.
 * \param[in] set The set.
 * \param[in] unit The unit to be used for printing: 's' for seconds, 'm' for
 * milliseconds and 'u' for microseconds.
 * \param[in] precision The precision of the printed values.
 */
void dstat_print_as_time( struct dstat stat, char unit, int precision )
{
	char unit_str[3];
	int normalizer;

	if( stat.n == 0 )
	{
		printf( "No time statistics available.\n" );
		return;
	}

	switch( unit )
	{
	case 's':
		strcpy( unit_str, "s" );
		normalizer = 1000000;
		break;
	case 'm':
		strcpy( unit_str, "ms" );
		normalizer = 1000;
		break;
	case 'u':
	default:
		strcpy( unit_str, "us" );
		normalizer = 1;
		break;
	}

	printf( "Elapsed time statistics from %d values (%s.): "
	        "%.*f/%.*f/%.*f +/- %.*f (min/avg/max +/- std_dev).\n",
	        stat.n,
	        unit_str,
	        precision,
	        stat.min / normalizer,
	        precision,
	        stat.mean / normalizer,
	        precision,
	        stat.max / normalizer,
	        precision,
	        sqrt( stat.stddev ) / normalizer );
}
