/**
 * @file fft.hpp
 *
 * fft based on jjj.de/fxt/fxtbook.pdf (radix-2 fft algorithms)
 *
 * @author Lukasz Wiecaszek <lukasz.wiecaszek@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 */

#ifndef _FFT_
#define _FFT_

/*===========================================================================*\
 * system header files
\*===========================================================================*/
#include <algorithm>

/*===========================================================================*\
 * project header files
\*===========================================================================*/
#include "complex.hpp"
#include "power_of_two.hpp"
#include "ilog2.hpp"

/*===========================================================================*\
 * preprocessor #define constants and macros
\*===========================================================================*/

/*===========================================================================*\
 * global type definitions
\*===========================================================================*/
namespace ymn
{

} /* end of namespace ymn */

/*===========================================================================*\
 * inline function/variable definitions
\*===========================================================================*/
namespace ymn
{

template<typename T>
inline void fft_reorder_samples(complex<T>* iq, const size_t N)
{
    /* decimation in time - re-order samples (in place) */
    for (size_t n = 1, m = 0; n < N; ++n) {
        int l = N;
        do
            l /= 2;
        while (m + l >= N);
        m = (m & (l - 1)) + l; // l is still power of 2
        if (m <= n)
            continue;
        std::swap(iq[n], iq[m]);
    }
}

template<typename T>
inline void fft_reorder_coefficients(complex<T>* iq, const size_t N)
{
    /* re-order coefficients (in place) */
    for (size_t n = 0; n < (N / 2); ++n)
        std::swap(iq[n], iq[n + (N / 2)]);
}

template<typename T>
inline void fft(complex<T>* iq, const complex<T>* e, const size_t N)
{
    fft_reorder_samples(iq, N);

    const int log2_N = ilog2(N);

    for (int log2_n = 0; log2_n < log2_N; ++log2_n) {
        int mh = 1 << log2_n;
        int m = mh * 2;
        for (int j = 0; j < mh; ++j) {
            for (int r = 0; r < static_cast<int>(N); r += m) {
                complex<T> u = iq[r + j];
                complex<T> v = iq[r + j + mh] * e[j];

                iq[r + j] = u + v;
                iq[r + j + mh] = u - v;
            }
        }
    }

    fft_reorder_coefficients(iq, N);
}

template<typename T, std::size_t N>
inline void fft(complex<T> (&iq)[N], const complex<T> (&e)[N])
{
    static_assert(is_power_of_two(N), "N must be power of 2");
    fft(&iq, &e, N);
}

} /* end of namespace ymn */

/*===========================================================================*\
 * global object declarations
\*===========================================================================*/
namespace ymn
{

} /* end of namespace ymn */

/*===========================================================================*\
 * function forward declarations
\*===========================================================================*/
namespace ymn
{

} /* end of namespace ymn */

#endif /* _FFT_ */
