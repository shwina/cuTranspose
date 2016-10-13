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

#ifndef KERNELS_012_H_
#define KERNELS_012_H_

/********************************************
 * Public function prototypes               *
 ********************************************/
__global__
void dev_copy( data_t*       out,
               const data_t* in,
               int           np0,
               int           np1,
               int           elements_per_thread );
#endif /* KERNELS_012_H_ */
