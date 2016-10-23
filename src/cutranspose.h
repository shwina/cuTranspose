#ifndef CUTRANSPOSE_H_
#define CUTRANSPOSE_H_

/* #undef USE_COMPLEX */
#define DOUBLE
#define TILE_SIZE 16
#define BRICK_SIZE 8 

#ifdef USE_COMPLEX
#include <complex.h>
#endif

/********************************************
 * Public function prototypes               *
 ********************************************/
#ifdef __cplusplus
extern "C"
{
#endif

#ifdef DOUBLE
#define real double
#else
#define real double
#endif


#ifdef USE_COMPLEX
typedef real _Complex data_t;
#else
typedef real data_t;
#endif

int cut_transpose3d( data_t*       output,
                     const data_t* input,
                     const int*    size,
                     const int*    permutation,
                     int           elements_per_thread );

#ifdef __cplusplus
}
#endif
#endif /* CUTRANSPOSE_H_ */
