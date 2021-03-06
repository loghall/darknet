/* 
 * File:   fixed.h
 * Author: ashwin
 *
 * Created on September 18, 2018, 5:13 PM
 */

#ifndef FIXED_H
#define FIXED_H

#ifdef __cplusplus
extern "C" {
#endif
    #define TOTAL_BITS 24
    #define FRACTION_BITS 16
    #define FIXED_MULT_DIVISOR (1 << FRACTION_BITS)


    void to_fixed(float* a, int h, int w, int32_t* a_fixed);
    void to_float(int32_t* a, int h, int w, float* a_float);


#ifdef __cplusplus
}
#endif

#endif /* FIXED_H */


