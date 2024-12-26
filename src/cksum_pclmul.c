/* cksum -- calculate and print POSIX checksums and sizes of files
   Copyright (C) 1992-2024 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#include <config.h>

#include <stdio.h>
#include <sys/types.h>
#include <stdint.h>
#include <x86intrin.h>
#include "system.h"

/* Number of bytes to read at once.  */
#define BUFLEN (1 << 16)
#define BUFLEN_WORDS (BUFLEN / sizeof(__m128i))

extern uint_fast32_t const crctab[8][256];

extern bool
cksum_pclmul (FILE * fp, uint_fast32_t * crc_out, uintmax_t * length_out);

/* Calculate CRC32 using PCLMULQDQ CPU instruction found in x86/x64 CPUs */

bool
cksum_pclmul (FILE *fp, uint_fast32_t *crc_out, uintmax_t *length_out)
{
  __m128i buf[BUFLEN_WORDS];
  int next_buf = 0;
  uint_fast32_t crc = 0;
  uintmax_t length = 0;
  size_t bytes_read;
  size_t batch_size_read;
  bool data_available = true;
  __m128i single_mult_constant;
  __m128i four_mult_constant;
  __m128i five_mult_constant;
  __m128i shuffle_constant;

  if (!fp || !crc_out || !length_out)
    return false;

  /* These constants and general algorithms are taken from the Intel whitepaper
     "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction"
     2^(128) mod P = 0xE8A45605
     2^(128+64) mod P = 0xC5B9CD4C
     2^(128*4) mod P = 0xE6228B11
     2^(128*4+64) mod P = 0x8833794C
     2^(128*5) mod P = 0xF91A84E2
     2^(128*5+64) mod P = 0xE2CA9D03
   */
  single_mult_constant = _mm_set_epi64x (0xC5B9CD4C, 0xE8A45605);
  four_mult_constant = _mm_set_epi64x (0x8833794C, 0xE6228B11);
  /* Extra fold for algorithm from https://arxiv.org/abs/2412.16398 */
  five_mult_constant = _mm_set_epi64x (0xE2CA9D03, 0xF91A84E2);

  /* Constant to byteswap a full SSE register */
  shuffle_constant = _mm_set_epi8 (0, 1, 2, 3, 4, 5, 6, 7, 8,
                                   9, 10, 11, 12, 13, 14, 15);

  bytes_read = fread (buf, 1, BUFLEN / 2, fp);
  bytes_read += fread (buf + (BUFLEN_WORDS / 2), 1, BUFLEN / 2, fp);
  if (bytes_read > 0)
    {
      __m128i *datap;
      int data_offset = 0;
      __m128i data;
      __m128i data2;
      __m128i data3;
      __m128i data4;
      __m128i data5;
      __m128i data6;
      __m128i data7;
      __m128i data8;
      __m128i fold_data;
      __m128i xor_crc;
      __m128i chorba1 = _mm_set_epi64x(0, 0);
      __m128i chorba2 = _mm_set_epi64x(0, 0);
      __m128i chorba3 = _mm_set_epi64x(0, 0);
      __m128i chorba4 = _mm_set_epi64x(0, 0);

      if (length + bytes_read < length)
        {
          errno = EOVERFLOW;
          return false;
        }
      length += bytes_read;

      datap = (__m128i *) buf;

      /* Fold in parallel eight 16-byte blocks into four 16-byte blocks */
      if (bytes_read >= 16 * 8)
        {
          data = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
          data = _mm_shuffle_epi8 (data, shuffle_constant);
          /* XOR in initial CRC value (for us 0 so no effect), or CRC value
             calculated for previous BUFLEN buffer from fread */
          xor_crc = _mm_set_epi32 (crc, 0, 0, 0);
          crc = 0;
          data = _mm_xor_si128 (data, xor_crc);
          data3 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS)
                                   + 1);
          data3 = _mm_shuffle_epi8 (data3, shuffle_constant);
          data5 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS)
                                   + 2);
          data5 = _mm_shuffle_epi8 (data5, shuffle_constant);
          data7 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS)
                                   + 3);
          data7 = _mm_shuffle_epi8 (data7, shuffle_constant);

          // use the chorba method to copy 8 vars forward without pclmul
          if (bytes_read >= 512*2 + 64 + 16 * 8)
            {
              while (bytes_read >= 512*2 + 64 + 16 * 8)
                {
                  data_offset += 4;

                  chorba1 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 4);
                  chorba1 = _mm_shuffle_epi8 (chorba1, shuffle_constant) ^ chorba2 ^ chorba4;
                  bytes_read -= 16;
                  data_offset += 1;

                  data2 = _mm_clmulepi64_si128 (data, five_mult_constant, 0x00);
                  data = _mm_clmulepi64_si128 (data, five_mult_constant, 0x11);
                  data4 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x00);
                  data3 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x11);
                  data6 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x00);
                  data5 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x11);
                  data8 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x00);
                  data7 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x11);

                  data = _mm_xor_si128 (data, data2);
                  data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
                  data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba2 ^ chorba4;
                  data = _mm_xor_si128 (data, data2);

                  data3 = _mm_xor_si128 (data3, data4);
                  data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
                  data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba3;
                  data3 = _mm_xor_si128 (data3, data4);

                  data5 = _mm_xor_si128 (data5, data6);
                  data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
                  data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba2 ^ chorba3;
                  data5 = _mm_xor_si128 (data5, data6);

                  data7 = _mm_xor_si128 (data7, data8);
                  data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
                  data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba2 ^ chorba3;
                  data7 = _mm_xor_si128 (data7, data8);

                  bytes_read -= (16 * 4);

                  data_offset += 4;

                  data2 = _mm_clmulepi64_si128 (data, five_mult_constant, 0x00);
                  data = _mm_clmulepi64_si128 (data, five_mult_constant, 0x11);
                  data4 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x00);
                  data3 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x11);
                  data6 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x00);
                  data5 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x11);
                  data8 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x00);
                  data7 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x11);

                  data = _mm_xor_si128 (data, data2);
                  data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
                  data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba2;
                  data = _mm_xor_si128 (data, data2);

                  data3 = _mm_xor_si128 (data3, data4);
                  data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
                  data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba1 ^ chorba3;
                  data3 = _mm_xor_si128 (data3, data4);

                  data5 = _mm_xor_si128 (data5, data6);
                  data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
                  data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba3 ^ chorba4;
                  data5 = _mm_xor_si128 (data5, data6);

                  data7 = _mm_xor_si128 (data7, data8);
                  data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
                  data8 = _mm_shuffle_epi8 (data8, shuffle_constant);
                  data7 = _mm_xor_si128 (data7, data8);

                  bytes_read -= (16 * 4);

                  data_offset += 4;

                  chorba2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 4);
                  chorba2 = _mm_shuffle_epi8 (chorba2, shuffle_constant) ^ chorba3 ^ chorba1;
                  bytes_read -= 16;
                  data_offset += 1;

                  data2 = _mm_clmulepi64_si128 (data, five_mult_constant, 0x00);
                  data = _mm_clmulepi64_si128 (data, five_mult_constant, 0x11);
                  data4 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x00);
                  data3 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x11);
                  data6 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x00);
                  data5 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x11);
                  data8 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x00);
                  data7 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x11);

                  data = _mm_xor_si128 (data, data2);
                  data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
                  data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba3 ^ chorba1;
                  data = _mm_xor_si128 (data, data2);

                  data3 = _mm_xor_si128 (data3, data4);
                  data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
                  data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba4;
                  data3 = _mm_xor_si128 (data3, data4);

                  data5 = _mm_xor_si128 (data5, data6);
                  data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
                  data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba3 ^ chorba4;
                  data5 = _mm_xor_si128 (data5, data6);

                  data7 = _mm_xor_si128 (data7, data8);
                  data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
                  data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba3 ^ chorba4;
                  data7 = _mm_xor_si128 (data7, data8);

                  bytes_read -= (16 * 4);

                  data_offset += 4;

                  data2 = _mm_clmulepi64_si128 (data, five_mult_constant, 0x00);
                  data = _mm_clmulepi64_si128 (data, five_mult_constant, 0x11);
                  data4 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x00);
                  data3 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x11);
                  data6 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x00);
                  data5 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x11);
                  data8 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x00);
                  data7 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x11);

                  data = _mm_xor_si128 (data, data2);
                  data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
                  data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba3;
                  data = _mm_xor_si128 (data, data2);

                  data3 = _mm_xor_si128 (data3, data4);
                  data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
                  data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba2 ^ chorba4;
                  data3 = _mm_xor_si128 (data3, data4);

                  data5 = _mm_xor_si128 (data5, data6);
                  data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
                  data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba4 ^ chorba1;
                  data5 = _mm_xor_si128 (data5, data6);

                  data7 = _mm_xor_si128 (data7, data8);
                  data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
                  data8 = _mm_shuffle_epi8 (data8, shuffle_constant);
                  data7 = _mm_xor_si128 (data7, data8);

                  bytes_read -= (16 * 4);

                  data_offset += 4;

                  chorba3 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 4);
                  chorba3 = _mm_shuffle_epi8 (chorba3, shuffle_constant) ^ chorba4 ^ chorba2;
                  bytes_read -= 16;
                  data_offset += 1;

                  data2 = _mm_clmulepi64_si128 (data, five_mult_constant, 0x00);
                  data = _mm_clmulepi64_si128 (data, five_mult_constant, 0x11);
                  data4 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x00);
                  data3 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x11);
                  data6 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x00);
                  data5 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x11);
                  data8 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x00);
                  data7 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x11);

                  data = _mm_xor_si128 (data, data2);
                  data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
                  data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba4 ^ chorba2;
                  data = _mm_xor_si128 (data, data2);

                  data3 = _mm_xor_si128 (data3, data4);
                  data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
                  data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba1;
                  data3 = _mm_xor_si128 (data3, data4);

                  data5 = _mm_xor_si128 (data5, data6);
                  data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
                  data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba4 ^ chorba1;
                  data5 = _mm_xor_si128 (data5, data6);

                  data7 = _mm_xor_si128 (data7, data8);
                  data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
                  data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba4 ^ chorba1;
                  data7 = _mm_xor_si128 (data7, data8);

                  bytes_read -= (16 * 4);

                  data_offset += 4;

                  data2 = _mm_clmulepi64_si128 (data, five_mult_constant, 0x00);
                  data = _mm_clmulepi64_si128 (data, five_mult_constant, 0x11);
                  data4 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x00);
                  data3 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x11);
                  data6 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x00);
                  data5 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x11);
                  data8 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x00);
                  data7 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x11);

                  data = _mm_xor_si128 (data, data2);
                  data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
                  data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba4;
                  data = _mm_xor_si128 (data, data2);

                  data3 = _mm_xor_si128 (data3, data4);
                  data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
                  data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba3 ^ chorba1;
                  data3 = _mm_xor_si128 (data3, data4);

                  data5 = _mm_xor_si128 (data5, data6);
                  data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
                  data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba1 ^ chorba2;
                  data5 = _mm_xor_si128 (data5, data6);

                  data7 = _mm_xor_si128 (data7, data8);
                  data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
                  data8 = _mm_shuffle_epi8 (data8, shuffle_constant);
                  data7 = _mm_xor_si128 (data7, data8);

                  bytes_read -= (16 * 4);

                  data_offset += 4;

                  chorba4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 4);
                  chorba4 = _mm_shuffle_epi8 (chorba4, shuffle_constant) ^ chorba1 ^ chorba3;
                  bytes_read -= 16;
                  data_offset += 1;

                  data2 = _mm_clmulepi64_si128 (data, five_mult_constant, 0x00);
                  data = _mm_clmulepi64_si128 (data, five_mult_constant, 0x11);
                  data4 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x00);
                  data3 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x11);
                  data6 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x00);
                  data5 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x11);
                  data8 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x00);
                  data7 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x11);

                  data = _mm_xor_si128 (data, data2);
                  data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
                  data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba1 ^ chorba3;
                  data = _mm_xor_si128 (data, data2);

                  data3 = _mm_xor_si128 (data3, data4);
                  data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
                  data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba2;
                  data3 = _mm_xor_si128 (data3, data4);

                  data5 = _mm_xor_si128 (data5, data6);
                  data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
                  data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba1 ^ chorba2;
                  data5 = _mm_xor_si128 (data5, data6);

                  data7 = _mm_xor_si128 (data7, data8);
                  data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
                  data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba1 ^ chorba2;
                  data7 = _mm_xor_si128 (data7, data8);

                  bytes_read -= (16 * 4);

                  data_offset += 4;

                  data2 = _mm_clmulepi64_si128 (data, five_mult_constant, 0x00);
                  data = _mm_clmulepi64_si128 (data, five_mult_constant, 0x11);
                  data4 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x00);
                  data3 = _mm_clmulepi64_si128 (data3, five_mult_constant, 0x11);
                  data6 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x00);
                  data5 = _mm_clmulepi64_si128 (data5, five_mult_constant, 0x11);
                  data8 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x00);
                  data7 = _mm_clmulepi64_si128 (data7, five_mult_constant, 0x11);

                  data = _mm_xor_si128 (data, data2);
                  data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
                  data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba1;
                  data = _mm_xor_si128 (data, data2);

                  data3 = _mm_xor_si128 (data3, data4);
                  data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
                  data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba4 ^ chorba2;
                  data3 = _mm_xor_si128 (data3, data4);

                  data5 = _mm_xor_si128 (data5, data6);
                  data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
                  data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba2 ^ chorba3;
                  data5 = _mm_xor_si128 (data5, data6);

                  data7 = _mm_xor_si128 (data7, data8);
                  data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
                  data8 = _mm_shuffle_epi8 (data8, shuffle_constant);
                  data7 = _mm_xor_si128 (data7, data8);

                  bytes_read -= (16 * 4);


                  /* Refill the buffer if we have used one half */
                  if (bytes_read < (BUFLEN / 2) && data_available)
                    {
                      batch_size_read = fread (buf + next_buf, 1, BUFLEN / 2, fp);
                      next_buf ^= (BUFLEN_WORDS / 2);
                      data_available = batch_size_read != 0;
                      bytes_read += batch_size_read;

                      if (length + batch_size_read < length)
                        {
                          errno = EOVERFLOW;
                          return false;
                        }
                      length += batch_size_read;
                    }
                }
              /* Clean up */
              data_offset += 4;
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba2 ^ chorba4;
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba2 ^ chorba4;
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba3;
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba2 ^ chorba3;
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
              data_offset += 4;
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba2 ^ chorba3;
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba2;
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba3;
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba3 ^ chorba4;
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
              data_offset += 4;
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant);
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba3;
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba3;
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba4;
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
              data_offset += 4;
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba3 ^ chorba4;
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba3 ^ chorba4;
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba3;
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba3;
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
              data_offset += 4;
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant) ^ chorba4;
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant);
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba4;
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba4;
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
              data_offset += 4;
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant);
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant) ^ chorba4;
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant) ^ chorba4;
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant) ^ chorba4;
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
              data_offset += 4;
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant);
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant);
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant);
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant);
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
              data_offset += 4;
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant);
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant);
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant);
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant);
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
              /* Refill the buffer if we have used one half */
              if (bytes_read < (BUFLEN / 2) && data_available)
                {
                  batch_size_read = fread (buf + next_buf, 1, BUFLEN / 2, fp);
                  next_buf ^= (BUFLEN_WORDS / 2);
                  data_available = batch_size_read != 0;
                  bytes_read += batch_size_read;

                  if (length + batch_size_read < length)
                    {
                      errno = EOVERFLOW;
                      return false;
                    }
                  length += batch_size_read;
                }
            }

          while (bytes_read >= 16 * 8)
            {
              data_offset += 4;

              /* Do multiplication here for four consecutive 16 byte blocks */
              data2 = _mm_clmulepi64_si128 (data, four_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, four_mult_constant, 0x11);
              data4 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x00);
              data3 = _mm_clmulepi64_si128 (data3, four_mult_constant, 0x11);
              data6 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x00);
              data5 = _mm_clmulepi64_si128 (data5, four_mult_constant, 0x11);
              data8 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x00);
              data7 = _mm_clmulepi64_si128 (data7, four_mult_constant, 0x11);

              /* Now multiplication results for the four blocks is xor:ed with
                 next four 16 byte blocks from the buffer. This effectively
                 "consumes" the first four blocks from the buffer.
                 Keep xor result in variables for multiplication in next
                 round of loop. */
              data = _mm_xor_si128 (data, data2);
              data2 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
              data2 = _mm_shuffle_epi8 (data2, shuffle_constant);
              data = _mm_xor_si128 (data, data2);

              data3 = _mm_xor_si128 (data3, data4);
              data4 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) +
                                       1);
              data4 = _mm_shuffle_epi8 (data4, shuffle_constant);
              data3 = _mm_xor_si128 (data3, data4);

              data5 = _mm_xor_si128 (data5, data6);
              data6 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) +
                                       2);
              data6 = _mm_shuffle_epi8 (data6, shuffle_constant);
              data5 = _mm_xor_si128 (data5, data6);

              data7 = _mm_xor_si128 (data7, data8);
              data8 = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS) +
                                       3);
              data8 = _mm_shuffle_epi8 (data8, shuffle_constant);
              data7 = _mm_xor_si128 (data7, data8);

              bytes_read -= (16 * 4);
            }
          /* At end of loop we write out results from variables back into
             the buffer, for use in single fold loop */
          data = _mm_shuffle_epi8 (data, shuffle_constant);
          _mm_storeu_si128 (datap + (data_offset % BUFLEN_WORDS),
                            data);
          data3 = _mm_shuffle_epi8 (data3, shuffle_constant);
          _mm_storeu_si128 (datap + (data_offset % BUFLEN_WORDS) + 1,
                            data3);
          data5 = _mm_shuffle_epi8 (data5, shuffle_constant);
          _mm_storeu_si128 (datap + (data_offset % BUFLEN_WORDS) + 2,
                            data5);
          data7 = _mm_shuffle_epi8 (data7, shuffle_constant);
          _mm_storeu_si128 (datap + (data_offset % BUFLEN_WORDS) + 3,
                            data7);
        }

      /* Fold two 16-byte blocks into one 16-byte block */
      if (bytes_read >= 32)
        {
          data = _mm_loadu_si128 (datap + (data_offset % BUFLEN_WORDS));
          data = _mm_shuffle_epi8 (data, shuffle_constant);
          xor_crc = _mm_set_epi32 (crc, 0, 0, 0);
          crc = 0;
          data = _mm_xor_si128 (data, xor_crc);
          while (bytes_read >= 32)
            {
              data_offset++;

              data2 = _mm_clmulepi64_si128 (data, single_mult_constant, 0x00);
              data = _mm_clmulepi64_si128 (data, single_mult_constant, 0x11);
              fold_data =
                _mm_loadu_si128 (datap +
                                 (data_offset %
                                  BUFLEN_WORDS));
              fold_data = _mm_shuffle_epi8 (fold_data, shuffle_constant);
              data = _mm_xor_si128 (data, data2);
              data = _mm_xor_si128 (data, fold_data);
              bytes_read -= 16;
            }
          data = _mm_shuffle_epi8 (data, shuffle_constant);
          _mm_storeu_si128 (datap + (data_offset % BUFLEN_WORDS),
                            data);
        }

      /* And finish up last 0-31 bytes in a byte by byte fashion */
      unsigned char *cp = (unsigned char *) datap
        + (data_offset % BUFLEN_WORDS);
      while (bytes_read--)
        crc = (crc << 8) ^ crctab[0][((crc >> 24) ^ *cp++) & 0xFF];
    }

  *crc_out = crc;
  *length_out = length;

  return !ferror (fp);
}
