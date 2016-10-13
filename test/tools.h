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

#ifndef TOOLS_H_
#define TOOLS_H_

/********************************************
 * Macros                                   *
 ********************************************/
#define TIME_DIFF(x,y) ((y.tv_sec - x.tv_sec)*1e6 + (y.tv_usec - x.tv_usec))
#define HANDLE_ERROR(x) \
{ \
	cudaError_t err = x; \
	if( cudaSuccess != err ) \
	{ \
		fprintf( stderr, \
		         "CUDA Error on call \"%s\": %s\n\tLine: %d, File: %s\n", \
		         #x, cudaGetErrorString( err ), __LINE__, __FILE__); \
		fflush( stdout );\
		exit( 1 ); \
	} \
}

/********************************************
 * Data definitions                         *
 ********************************************/
struct dstat
{
	int n;
	double min, max;
	double mean, stddev;
	double q;				// For internal use. To compute the stddev.
};

/********************************************
 * Public functions                         *
 ********************************************/
#ifdef __cplusplus
extern "C"
{
#endif
int is_permutation( int* perm );
int* split_ints( int dim, const char* str, const char* sep );
void dstat_init( struct dstat* stat );
void dstat_add(struct dstat* stat, double value );
void dstat_print_as_time( struct dstat stat, char unit, int precision );
#ifdef __cplusplus
}
#endif
#endif /* TOOLS_H_ */
