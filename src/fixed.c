/* 
 * File:   fixed.c
 * Author: ashwin, zachary
 *
 * Created on September 18, 2018, 5:09 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "fixed.h"

#define floatToFixed(x) (x >= 0 ? ((x * FIXED_MULT_DIVISOR) + 0.5) : ((x * FIXED_MULT_DIVISOR) - 0.5))
#define fixedToFloat(x) (x / (float) FIXED_MULT_DIVISOR)

void to_fixed(float* a, int M, int N, int32_t* a_fixed) {
    for(int iii = 0; iii < M; iii++) {
		#pragma omp parallel for
        for(int jjj = 0; jjj < N; jjj++) {
            a_fixed[iii * N + jjj] = floatToFixed(a[iii * N + jjj]);
			// printf("original: %f, new: %d\n", a[iii * N + jjj], a_fixed[iii * N + jjj]);
        }
    }
	// printf("\n\n");
}

void to_float(int32_t* a, int M, int N, float* a_float) {
    for(int iii = 0; iii < M; iii++) {
		#pragma omp parallel for
        for(int jjj = 0; jjj < N; jjj++) {
            a_float[iii * N + jjj] = fixedToFloat(a[iii * N + jjj]);
			// printf("value:%f\n", a_float[iii * N + jjj]);
        }
    }    
	// printf("\n\n");
}

