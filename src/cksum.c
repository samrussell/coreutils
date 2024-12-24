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

/* Written by Q. Frank Xia, qx@math.columbia.edu.
   Cosmetic changes and reorganization by David MacKenzie, djm@gnu.ai.mit.edu.

  Usage: cksum [file...]

  The code segment between "#ifdef CRCTAB" and "#else" is the code
  which calculates the "crctab". It is included for those who want
  verify the correctness of the "crctab". To recreate the "crctab",
  do something like the following:

      cc -I../lib -DCRCTAB -o crctab cksum.c
      ./crctab > crctab.c

  This software is compatible with neither the System V nor the BSD
  'sum' program.  It is supposed to conform to POSIX, except perhaps
  for foreign language support.  Any inconsistency with the standard
  (other than foreign language support) is a bug.  */

#include <config.h>

#include <stdio.h>
#include <sys/types.h>
#include <stdint.h>
#include <endian.h>
#include "system.h"

#ifdef USE_VMULL_CRC32
# include <sys/auxv.h>
# include <asm/hwcap.h>
#endif

#ifdef CRCTAB

# define BIT(x)	((uint_fast32_t) 1 << (x))
# define SBIT	BIT (31)

/* The generating polynomial is

          32   26   23   22   16   12   11   10   8   7   5   4   2   1
    G(X)=X  + X  + X  + X  + X  + X  + X  + X  + X + X + X + X + X + X + 1

  The i bit in GEN is set if X^i is a summand of G(X) except X^32.  */

# define GEN	(BIT (26) | BIT (23) | BIT (22) | BIT (16) | BIT (12) \
                 | BIT (11) | BIT (10) | BIT (8) | BIT (7) | BIT (5) \
                 | BIT (4) | BIT (2) | BIT (1) | BIT (0))

static uint_fast32_t r[8];

static void
fill_r (void)
{
  r[0] = GEN;
  for (int i = 1; i < 8; i++)
    r[i] = (r[i - 1] << 1) ^ ((r[i - 1] & SBIT) ? GEN : 0);
}

static uint_fast32_t
crc_remainder (int m)
{
  uint_fast32_t rem = 0;

  for (int i = 0; i < 8; i++)
    if (BIT (i) & m)
      rem ^= r[i];

  return rem & 0xFFFFFFFF;	/* Make it run on 64-bit machine.  */
}

int
main (void)
{
  int i;
  static uint_fast32_t crctab[8][256];

  fill_r ();

  for (i = 0; i < 256; i++)
    {
      crctab[0][i] = crc_remainder (i);
    }

  /* CRC(0x11 0x22 0x33 0x44)
     is equal to
     CRC(0x11 0x00 0x00 0x00) XOR CRC(0x22 0x00 0x00) XOR
     CRC(0x33 0x00) XOR CRC(0x44)
     We precompute the CRC values for the offset values into
     separate CRC tables. We can then use them to speed up
     CRC calculation by processing multiple bytes at the time. */
  for (i = 0; i < 256; i++)
    {
      uint32_t crc = 0;

      crc = (crc << 8) ^ crctab[0][((crc >> 24) ^ (i & 0xFF)) & 0xFF];
      for (idx_t offset = 1; offset < 8; offset++)
        {
          crc = (crc << 8) ^ crctab[0][((crc >> 24) ^ 0x00) & 0xFF];
          crctab[offset][i] = crc;
        }
    }

  printf ("#include <config.h>\n");
  printf ("#include <stdint.h>\n");
  printf ("#include <stdio.h>\n");
  printf ("#include \"cksum.h\"\n");
  printf ("\n");
  printf ("uint_fast32_t const crctab[8][256] = {\n");
  for (int y = 0; y < 8; y++)
    {
      printf ("{\n  0x%08x", crctab[y][0]);
      for (i = 0; i < 51; i++)
        {
          printf (",\n  0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x",
                  crctab[y][i * 5 + 1], crctab[y][i * 5 + 2],
                  crctab[y][i * 5 + 3], crctab[y][i * 5 + 4],
                  crctab[y][i * 5 + 5]);
        }
        printf ("\n},\n");
    }
  printf ("};\n");
  return EXIT_SUCCESS;
}

#else /* !CRCTAB */

# include "cksum.h"
# include "crc.h"

/* Number of bytes to read at once.  */
# define BUFLEN (1 << 20)

static bool
pclmul_supported (void)
{
  bool pclmul_enabled = false;
# if USE_PCLMUL_CRC32 || GL_CRC_X86_64_PCLMUL
  pclmul_enabled = (0 < __builtin_cpu_supports ("pclmul")
                    && 0 < __builtin_cpu_supports ("avx"));

  if (cksum_debug)
    error (0, 0, "%s",
           (pclmul_enabled
            ? _("using pclmul hardware support")
            : _("pclmul support not detected")));
# endif

  return pclmul_enabled;
}

static bool
avx2_supported (void)
{
  /* AVX512 processors will not set vpclmulqdq unless they support
     the avx512 version, but it implies that the avx2 version
     is supported  */
  bool avx2_enabled = false;
# if USE_AVX2_CRC32
  avx2_enabled = (0 < __builtin_cpu_supports ("vpclmulqdq")
                  && 0 < __builtin_cpu_supports ("avx2"));

  if (cksum_debug)
    error (0, 0, "%s",
           (avx2_enabled
            ? _("using avx2 hardware support")
            : _("avx2 support not detected")));
# endif

  return avx2_enabled;
}

static bool
avx512_supported (void)
{
  /* vpclmulqdq for multiplication
     mavx512f for most of the avx512 functions we're using
     mavx512bw for byte swapping  */
  bool avx512_enabled = false;
# if USE_AVX512_CRC32
  avx512_enabled = (0 < __builtin_cpu_supports ("vpclmulqdq")
                    && 0 < __builtin_cpu_supports ("avx512bw")
                    && 0 < __builtin_cpu_supports ("avx512f"));

  if (cksum_debug)
    error (0, 0, "%s",
           (avx512_enabled
            ? _("using avx512 hardware support")
            : _("avx512 support not detected")));
# endif

  return avx512_enabled;
}

static bool
vmull_supported (void)
{
  /* vmull for multiplication  */
  bool vmull_enabled = false;
# if USE_VMULL_CRC32

  vmull_enabled = (getauxval (AT_HWCAP) & HWCAP_PMULL) > 0;

  if (cksum_debug)
    error (0, 0, "%s",
           (vmull_enabled
            ? _("using vmull hardware support")
            : _("vmull support not detected")));
# endif

  return vmull_enabled;
}

static
uint32_t chorba_small_nondestructive (uint32_t crc, char* buf, size_t len) {
    char* input = buf;
    unsigned char final[72] = {0};
    uint64_t next1 = crc;
    next1 = next1 << 32;
    crc = 0;
    uint64_t next2 = 0;
    uint64_t next3 = 0;
    uint64_t next4 = 0;
    uint64_t next5 = 0;
    uint64_t next6 = 0;

    int i = 0;
    for(; i + 32 + 40 < len; i += 32) {
        uint64_t in1;
        uint64_t in2;
        uint64_t in3;
        uint64_t in4;
        uint64_t a1, a2, a3, a4;
        uint64_t b1, b2, b3, b4;
        uint64_t c1, c2, c3, c4;
        uint64_t d1, d2, d3, d4;

        uint64_t out1;
        uint64_t out2;
        uint64_t out3;
        uint64_t out4;
        uint64_t out5;

        in1 = htobe64(*((uint64_t*) (input + i))) ^ next1;
        in2 = htobe64(*((uint64_t*) (input + i + (8*1)))) ^ next2;

        a1 = (in1 >> 17) ^ (in1 >> 55);
        a2 = (in1 << 47) ^ (in1 << 9) ^ (in1 >> 19);
        a3 = (in1 << 45) ^ (in1 >> 44);
        a4 = (in1 << 20);

        b1 = (in2 >> 17) ^ (in2 >> 55);
        b2 = (in2 << 47) ^ (in2 << 9) ^ (in2 >> 19);
        b3 = (in2 << 45) ^ (in2 >> 44);
        b4 = (in2 << 20);

        in3 = htobe64(*((uint64_t*) (input + i + (8*2)))) ^ next3 ^ a1;
        in4 = htobe64(*((uint64_t*) (input + i + (8*3)))) ^ next4 ^ a2 ^ b1;

        c1 = (in3 >> 17) ^ (in3 >> 55);
        c2 = (in3 << 47) ^ (in3 << 9) ^ (in3 >> 19);
        c3 = (in3 << 45) ^ (in3 >> 44);
        c4 = (in3 << 20);

        d1 = (in4 >> 17) ^ (in4 >> 55);
        d2 = (in4 << 47) ^ (in4 << 9) ^ (in4 >> 19);
        d3 = (in4 << 45) ^ (in4 >> 44);
        d4 = (in4 << 20);

        out1 = a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;

    }

    memcpy(final, input+i, len-i);
    *((uint64_t*) (final + (0*8))) ^= be64toh(next1);
    *((uint64_t*) (final + (1*8))) ^= be64toh(next2);
    *((uint64_t*) (final + (2*8))) ^= be64toh(next3);
    *((uint64_t*) (final + (3*8))) ^= be64toh(next4);
    *((uint64_t*) (final + (4*8))) ^= be64toh(next5);

    // up to 72 bytes left

    // RFC 1952
    for(int j = 0; j<len-i; j++) {
        crc = crctab[0][(crc >> 24) ^ final[j]] ^ (crc << 8);
    }

    return crc;
}

#define bitbuffersizebytes (128 * 1024)
#define bitbuffersizeqwords (bitbuffersizebytes / sizeof(uint64_t))

uint32_t chorba_118960_nondestructive (uint32_t crc, char* buf, size_t len) {
    if ((118960*2) + 512 > len)
        return chorba_small_nondestructive(crc, buf, len);

    unsigned char* input = buf;
    uint64_t* bitbuffer = malloc(bitbuffersizebytes);
    unsigned char* bitbufferbytes = (unsigned char*) bitbuffer;
    int i = 0;

    uint64_t next1 = crc;
    next1 = next1 << 32;
    uint64_t next2 = 0;
    uint64_t next3 = 0;
    uint64_t next4 = 0;
    uint64_t next5 = 0;
    uint64_t next6 = 0;
    uint64_t next7 = 0;
    uint64_t next8 = 0;
    uint64_t next9 = 0;
    uint64_t next10 = 0;
    uint64_t next11 = 0;
    uint64_t next12 = 0;
    uint64_t next13 = 0;
    uint64_t next14 = 0;
    uint64_t next15 = 0;
    uint64_t next16 = 0;
    uint64_t next17 = 0;
    uint64_t next18 = 0;
    uint64_t next19 = 0;
    uint64_t next20 = 0;
    uint64_t next21 = 0;
    uint64_t next22 = 0;
    crc = 0;

    // do a first pass to zero out bitbuffer
    for(; i < 118784; i += 256) {
        uint64_t in1, in2, in3, in4, in5, in6, in7, in8;
        uint64_t in9, in10, in11, in12, in13, in14, in15, in16;
        uint64_t in17, in18, in19, in20, in21, in22, in23, in24;
        uint64_t in25, in26, in27, in28, in29, in30, in31, in32;
        int outoffset1 = ((i + 118784)/8) % bitbuffersizeqwords;
        int outoffset2 = ((i + 119040)/8) % bitbuffersizeqwords;

        in1 = *((uint64_t*) (input + i + (0*8))) ^ next1;
        in2 = *((uint64_t*) (input + i + (1*8))) ^ next2;
        in3 = *((uint64_t*) (input + i + (2*8))) ^ next3;
        in4 = *((uint64_t*) (input + i + (3*8))) ^ next4;
        in5 = *((uint64_t*) (input + i + (4*8))) ^ next5;
        in6 = *((uint64_t*) (input + i + (5*8))) ^ next6;
        in7 = *((uint64_t*) (input + i + (6*8))) ^ next7;
        in8 = *((uint64_t*) (input + i + (7*8))) ^ next8 ^ in1;
        in9 = *((uint64_t*) (input + i + (8*8))) ^ next9 ^ in2;
        in10 = *((uint64_t*) (input + i + (9*8))) ^ next10 ^ in3;
        in11 = *((uint64_t*) (input + i + (10*8))) ^ next11 ^ in4;
        in12 = *((uint64_t*) (input + i + (11*8))) ^ next12 ^ in1 ^ in5;
        in13 = *((uint64_t*) (input + i + (12*8))) ^ next13 ^ in2 ^ in6;
        in14 = *((uint64_t*) (input + i + (13*8))) ^ next14 ^ in3 ^ in7;
        in15 = *((uint64_t*) (input + i + (14*8))) ^ next15 ^ in4 ^ in8;
        in16 = *((uint64_t*) (input + i + (15*8))) ^ next16 ^ in5 ^ in9;
        in17 = *((uint64_t*) (input + i + (16*8))) ^ next17 ^ in6 ^ in10;
        in18 = *((uint64_t*) (input + i + (17*8))) ^ next18 ^ in7 ^ in11;
        in19 = *((uint64_t*) (input + i + (18*8))) ^ next19 ^ in8 ^ in12;
        in20 = *((uint64_t*) (input + i + (19*8))) ^ next20 ^ in9 ^ in13;
        in21 = *((uint64_t*) (input + i + (20*8))) ^ next21 ^ in10 ^ in14;
        in22 = *((uint64_t*) (input + i + (21*8))) ^ next22 ^ in11 ^ in15;
        in23 = *((uint64_t*) (input + i + (22*8))) ^ in1 ^ in12 ^ in16;
        in24 = *((uint64_t*) (input + i + (23*8))) ^ in2 ^ in13 ^ in17;
        in25 = *((uint64_t*) (input + i + (24*8))) ^ in3 ^ in14 ^ in18;
        in26 = *((uint64_t*) (input + i + (25*8))) ^ in4 ^ in15 ^ in19;
        in27 = *((uint64_t*) (input + i + (26*8))) ^ in5 ^ in16 ^ in20;
        in28 = *((uint64_t*) (input + i + (27*8))) ^ in6 ^ in17 ^ in21;
        in29 = *((uint64_t*) (input + i + (28*8))) ^ in7 ^ in18 ^ in22;
        in30 = *((uint64_t*) (input + i + (29*8))) ^ in8 ^ in19 ^ in23;
        in31 = *((uint64_t*) (input + i + (30*8))) ^ in9 ^ in20 ^ in24;
        in32 = *((uint64_t*) (input + i + (31*8))) ^ in10 ^ in21 ^ in25;

        next1 = in11 ^ in22 ^ in26;
        next2 = in12 ^ in23 ^ in27;
        next3 = in13 ^ in24 ^ in28;
        next4 = in14 ^ in25 ^ in29;
        next5 = in15 ^ in26 ^ in30;
        next6 = in16 ^ in27 ^ in31;
        next7 = in17 ^ in28 ^ in32;
        next8 = in18 ^ in29;
        next9 = in19 ^ in30;
        next10 = in20 ^ in31;
        next11 = in21 ^ in32;
        next12 = in22;
        next13 = in23;
        next14 = in24;
        next15 = in25;
        next16 = in26;
        next17 = in27;
        next18 = in28;
        next19 = in29;
        next20 = in30;
        next21 = in31;
        next22 = in32;

        bitbuffer[outoffset1 + 22] = in1;
        bitbuffer[outoffset1 + 23] = in2;
        bitbuffer[outoffset1 + 24] = in3;
        bitbuffer[outoffset1 + 25] = in4;
        bitbuffer[outoffset1 + 26] = in5;
        bitbuffer[outoffset1 + 27] = in6;
        bitbuffer[outoffset1 + 28] = in7;
        bitbuffer[outoffset1 + 29] = in8;
        bitbuffer[outoffset1 + 30] = in9;
        bitbuffer[outoffset1 + 31] = in10;
        bitbuffer[outoffset2 + 0] = in11;
        bitbuffer[outoffset2 + 1] = in12;
        bitbuffer[outoffset2 + 2] = in13;
        bitbuffer[outoffset2 + 3] = in14;
        bitbuffer[outoffset2 + 4] = in15;
        bitbuffer[outoffset2 + 5] = in16;
        bitbuffer[outoffset2 + 6] = in17;
        bitbuffer[outoffset2 + 7] = in18;
        bitbuffer[outoffset2 + 8] = in19;
        bitbuffer[outoffset2 + 9] = in20;
        bitbuffer[outoffset2 + 10] = in21;
        bitbuffer[outoffset2 + 11] = in22;
        bitbuffer[outoffset2 + 12] = in23;
        bitbuffer[outoffset2 + 13] = in24;
        bitbuffer[outoffset2 + 14] = in25;
        bitbuffer[outoffset2 + 15] = in26;
        bitbuffer[outoffset2 + 16] = in27;
        bitbuffer[outoffset2 + 17] = in28;
        bitbuffer[outoffset2 + 18] = in29;
        bitbuffer[outoffset2 + 19] = in30;
        bitbuffer[outoffset2 + 20] = in31;
        bitbuffer[outoffset2 + 21] = in32;
    }

    // one intermediate pass where we pull half the values
    for(; i < 119040; i += 256) {
        uint64_t in1, in2, in3, in4, in5, in6, in7, in8;
        uint64_t in9, in10, in11, in12, in13, in14, in15, in16;
        uint64_t in17, in18, in19, in20, in21, in22, in23, in24;
        uint64_t in25, in26, in27, in28, in29, in30, in31, in32;
        int inoffset = (i/8) % bitbuffersizeqwords;
        int outoffset1 = ((i + 118784)/8) % bitbuffersizeqwords;
        int outoffset2 = ((i + 119040)/8) % bitbuffersizeqwords;

        in1 = *((uint64_t*) (input + i + (0*8))) ^ next1;
        in2 = *((uint64_t*) (input + i + (1*8))) ^ next2;
        in3 = *((uint64_t*) (input + i + (2*8))) ^ next3;
        in4 = *((uint64_t*) (input + i + (3*8))) ^ next4;
        in5 = *((uint64_t*) (input + i + (4*8))) ^ next5;
        in6 = *((uint64_t*) (input + i + (5*8))) ^ next6;
        in7 = *((uint64_t*) (input + i + (6*8))) ^ next7;
        in8 = *((uint64_t*) (input + i + (7*8))) ^ next8 ^ in1;
        in9 = *((uint64_t*) (input + i + (8*8))) ^ next9 ^ in2;
        in10 = *((uint64_t*) (input + i + (9*8))) ^ next10 ^ in3;
        in11 = *((uint64_t*) (input + i + (10*8))) ^ next11 ^ in4;
        in12 = *((uint64_t*) (input + i + (11*8))) ^ next12 ^ in1 ^ in5;
        in13 = *((uint64_t*) (input + i + (12*8))) ^ next13 ^ in2 ^ in6;
        in14 = *((uint64_t*) (input + i + (13*8))) ^ next14 ^ in3 ^ in7;
        in15 = *((uint64_t*) (input + i + (14*8))) ^ next15 ^ in4 ^ in8;
        in16 = *((uint64_t*) (input + i + (15*8))) ^ next16 ^ in5 ^ in9;
        in17 = *((uint64_t*) (input + i + (16*8))) ^ next17 ^ in6 ^ in10;
        in18 = *((uint64_t*) (input + i + (17*8))) ^ next18 ^ in7 ^ in11;
        in19 = *((uint64_t*) (input + i + (18*8))) ^ next19 ^ in8 ^ in12;
        in20 = *((uint64_t*) (input + i + (19*8))) ^ next20 ^ in9 ^ in13;
        in21 = *((uint64_t*) (input + i + (20*8))) ^ next21 ^ in10 ^ in14;
        in22 = *((uint64_t*) (input + i + (21*8))) ^ next22 ^ in11 ^ in15;
        in23 = *((uint64_t*) (input + i + (22*8))) ^ in1 ^ in12 ^ in16 ^ bitbuffer[inoffset + 22];
        in24 = *((uint64_t*) (input + i + (23*8))) ^ in2 ^ in13 ^ in17 ^ bitbuffer[inoffset + 23];
        in25 = *((uint64_t*) (input + i + (24*8))) ^ in3 ^ in14 ^ in18 ^ bitbuffer[inoffset + 24];
        in26 = *((uint64_t*) (input + i + (25*8))) ^ in4 ^ in15 ^ in19 ^ bitbuffer[inoffset + 25];
        in27 = *((uint64_t*) (input + i + (26*8))) ^ in5 ^ in16 ^ in20 ^ bitbuffer[inoffset + 26];
        in28 = *((uint64_t*) (input + i + (27*8))) ^ in6 ^ in17 ^ in21 ^ bitbuffer[inoffset + 27];
        in29 = *((uint64_t*) (input + i + (28*8))) ^ in7 ^ in18 ^ in22 ^ bitbuffer[inoffset + 28];
        in30 = *((uint64_t*) (input + i + (29*8))) ^ in8 ^ in19 ^ in23 ^ bitbuffer[inoffset + 29];
        in31 = *((uint64_t*) (input + i + (30*8))) ^ in9 ^ in20 ^ in24 ^ bitbuffer[inoffset + 30];
        in32 = *((uint64_t*) (input + i + (31*8))) ^ in10 ^ in21 ^ in25 ^ bitbuffer[inoffset + 31];

        next1 = in11 ^ in22 ^ in26;
        next2 = in12 ^ in23 ^ in27;
        next3 = in13 ^ in24 ^ in28;
        next4 = in14 ^ in25 ^ in29;
        next5 = in15 ^ in26 ^ in30;
        next6 = in16 ^ in27 ^ in31;
        next7 = in17 ^ in28 ^ in32;
        next8 = in18 ^ in29;
        next9 = in19 ^ in30;
        next10 = in20 ^ in31;
        next11 = in21 ^ in32;
        next12 = in22;
        next13 = in23;
        next14 = in24;
        next15 = in25;
        next16 = in26;
        next17 = in27;
        next18 = in28;
        next19 = in29;
        next20 = in30;
        next21 = in31;
        next22 = in32;

        bitbuffer[outoffset1 + 22] = in1;
        bitbuffer[outoffset1 + 23] = in2;
        bitbuffer[outoffset1 + 24] = in3;
        bitbuffer[outoffset1 + 25] = in4;
        bitbuffer[outoffset1 + 26] = in5;
        bitbuffer[outoffset1 + 27] = in6;
        bitbuffer[outoffset1 + 28] = in7;
        bitbuffer[outoffset1 + 29] = in8;
        bitbuffer[outoffset1 + 30] = in9;
        bitbuffer[outoffset1 + 31] = in10;
        bitbuffer[outoffset2 + 0] = in11;
        bitbuffer[outoffset2 + 1] = in12;
        bitbuffer[outoffset2 + 2] = in13;
        bitbuffer[outoffset2 + 3] = in14;
        bitbuffer[outoffset2 + 4] = in15;
        bitbuffer[outoffset2 + 5] = in16;
        bitbuffer[outoffset2 + 6] = in17;
        bitbuffer[outoffset2 + 7] = in18;
        bitbuffer[outoffset2 + 8] = in19;
        bitbuffer[outoffset2 + 9] = in20;
        bitbuffer[outoffset2 + 10] = in21;
        bitbuffer[outoffset2 + 11] = in22;
        bitbuffer[outoffset2 + 12] = in23;
        bitbuffer[outoffset2 + 13] = in24;
        bitbuffer[outoffset2 + 14] = in25;
        bitbuffer[outoffset2 + 15] = in26;
        bitbuffer[outoffset2 + 16] = in27;
        bitbuffer[outoffset2 + 17] = in28;
        bitbuffer[outoffset2 + 18] = in29;
        bitbuffer[outoffset2 + 19] = in30;
        bitbuffer[outoffset2 + 20] = in31;
        bitbuffer[outoffset2 + 21] = in32;
    }

    for(; i + 118960 + 512 < len; i += 256) {
        uint64_t in1, in2, in3, in4, in5, in6, in7, in8;
        uint64_t in9, in10, in11, in12, in13, in14, in15, in16;
        uint64_t in17, in18, in19, in20, in21, in22, in23, in24;
        uint64_t in25, in26, in27, in28, in29, in30, in31, in32;
        int inoffset = (i/8) % bitbuffersizeqwords;
        int outoffset1 = ((i + 118784)/8) % bitbuffersizeqwords;
        int outoffset2 = ((i + 119040)/8) % bitbuffersizeqwords;

        in1 = *((uint64_t*) (input + i + (0*8))) ^ next1 ^ bitbuffer[inoffset + 0];
        in2 = *((uint64_t*) (input + i + (1*8))) ^ next2 ^ bitbuffer[inoffset + 1];
        in3 = *((uint64_t*) (input + i + (2*8))) ^ next3 ^ bitbuffer[inoffset + 2];
        in4 = *((uint64_t*) (input + i + (3*8))) ^ next4 ^ bitbuffer[inoffset + 3];
        in5 = *((uint64_t*) (input + i + (4*8))) ^ next5 ^ bitbuffer[inoffset + 4];
        in6 = *((uint64_t*) (input + i + (5*8))) ^ next6 ^ bitbuffer[inoffset + 5];
        in7 = *((uint64_t*) (input + i + (6*8))) ^ next7 ^ bitbuffer[inoffset + 6];
        in8 = *((uint64_t*) (input + i + (7*8))) ^ next8 ^ in1 ^ bitbuffer[inoffset + 7];
        in9 = *((uint64_t*) (input + i + (8*8))) ^ next9 ^ in2 ^ bitbuffer[inoffset + 8];
        in10 = *((uint64_t*) (input + i + (9*8))) ^ next10 ^ in3 ^ bitbuffer[inoffset + 9];
        in11 = *((uint64_t*) (input + i + (10*8))) ^ next11 ^ in4 ^ bitbuffer[inoffset + 10];
        in12 = *((uint64_t*) (input + i + (11*8))) ^ next12 ^ in1 ^ in5 ^ bitbuffer[inoffset + 11];
        in13 = *((uint64_t*) (input + i + (12*8))) ^ next13 ^ in2 ^ in6 ^ bitbuffer[inoffset + 12];
        in14 = *((uint64_t*) (input + i + (13*8))) ^ next14 ^ in3 ^ in7 ^ bitbuffer[inoffset + 13];
        in15 = *((uint64_t*) (input + i + (14*8))) ^ next15 ^ in4 ^ in8 ^ bitbuffer[inoffset + 14];
        in16 = *((uint64_t*) (input + i + (15*8))) ^ next16 ^ in5 ^ in9 ^ bitbuffer[inoffset + 15];
        in17 = *((uint64_t*) (input + i + (16*8))) ^ next17 ^ in6 ^ in10 ^ bitbuffer[inoffset + 16];
        in18 = *((uint64_t*) (input + i + (17*8))) ^ next18 ^ in7 ^ in11 ^ bitbuffer[inoffset + 17];
        in19 = *((uint64_t*) (input + i + (18*8))) ^ next19 ^ in8 ^ in12 ^ bitbuffer[inoffset + 18];
        in20 = *((uint64_t*) (input + i + (19*8))) ^ next20 ^ in9 ^ in13 ^ bitbuffer[inoffset + 19];
        in21 = *((uint64_t*) (input + i + (20*8))) ^ next21 ^ in10 ^ in14 ^ bitbuffer[inoffset + 20];
        in22 = *((uint64_t*) (input + i + (21*8))) ^ next22 ^ in11 ^ in15 ^ bitbuffer[inoffset + 21];
        in23 = *((uint64_t*) (input + i + (22*8))) ^ in1 ^ in12 ^ in16 ^ bitbuffer[inoffset + 22];
        in24 = *((uint64_t*) (input + i + (23*8))) ^ in2 ^ in13 ^ in17 ^ bitbuffer[inoffset + 23];
        in25 = *((uint64_t*) (input + i + (24*8))) ^ in3 ^ in14 ^ in18 ^ bitbuffer[inoffset + 24];
        in26 = *((uint64_t*) (input + i + (25*8))) ^ in4 ^ in15 ^ in19 ^ bitbuffer[inoffset + 25];
        in27 = *((uint64_t*) (input + i + (26*8))) ^ in5 ^ in16 ^ in20 ^ bitbuffer[inoffset + 26];
        in28 = *((uint64_t*) (input + i + (27*8))) ^ in6 ^ in17 ^ in21 ^ bitbuffer[inoffset + 27];
        in29 = *((uint64_t*) (input + i + (28*8))) ^ in7 ^ in18 ^ in22 ^ bitbuffer[inoffset + 28];
        in30 = *((uint64_t*) (input + i + (29*8))) ^ in8 ^ in19 ^ in23 ^ bitbuffer[inoffset + 29];
        in31 = *((uint64_t*) (input + i + (30*8))) ^ in9 ^ in20 ^ in24 ^ bitbuffer[inoffset + 30];
        in32 = *((uint64_t*) (input + i + (31*8))) ^ in10 ^ in21 ^ in25 ^ bitbuffer[inoffset + 31];

        next1 = in11 ^ in22 ^ in26;
        next2 = in12 ^ in23 ^ in27;
        next3 = in13 ^ in24 ^ in28;
        next4 = in14 ^ in25 ^ in29;
        next5 = in15 ^ in26 ^ in30;
        next6 = in16 ^ in27 ^ in31;
        next7 = in17 ^ in28 ^ in32;
        next8 = in18 ^ in29;
        next9 = in19 ^ in30;
        next10 = in20 ^ in31;
        next11 = in21 ^ in32;
        next12 = in22;
        next13 = in23;
        next14 = in24;
        next15 = in25;
        next16 = in26;
        next17 = in27;
        next18 = in28;
        next19 = in29;
        next20 = in30;
        next21 = in31;
        next22 = in32;

        bitbuffer[outoffset1 + 22] = in1;
        bitbuffer[outoffset1 + 23] = in2;
        bitbuffer[outoffset1 + 24] = in3;
        bitbuffer[outoffset1 + 25] = in4;
        bitbuffer[outoffset1 + 26] = in5;
        bitbuffer[outoffset1 + 27] = in6;
        bitbuffer[outoffset1 + 28] = in7;
        bitbuffer[outoffset1 + 29] = in8;
        bitbuffer[outoffset1 + 30] = in9;
        bitbuffer[outoffset1 + 31] = in10;
        bitbuffer[outoffset2 + 0] = in11;
        bitbuffer[outoffset2 + 1] = in12;
        bitbuffer[outoffset2 + 2] = in13;
        bitbuffer[outoffset2 + 3] = in14;
        bitbuffer[outoffset2 + 4] = in15;
        bitbuffer[outoffset2 + 5] = in16;
        bitbuffer[outoffset2 + 6] = in17;
        bitbuffer[outoffset2 + 7] = in18;
        bitbuffer[outoffset2 + 8] = in19;
        bitbuffer[outoffset2 + 9] = in20;
        bitbuffer[outoffset2 + 10] = in21;
        bitbuffer[outoffset2 + 11] = in22;
        bitbuffer[outoffset2 + 12] = in23;
        bitbuffer[outoffset2 + 13] = in24;
        bitbuffer[outoffset2 + 14] = in25;
        bitbuffer[outoffset2 + 15] = in26;
        bitbuffer[outoffset2 + 16] = in27;
        bitbuffer[outoffset2 + 17] = in28;
        bitbuffer[outoffset2 + 18] = in29;
        bitbuffer[outoffset2 + 19] = in30;
        bitbuffer[outoffset2 + 20] = in31;
        bitbuffer[outoffset2 + 21] = in32;
    }

    bitbuffer[((i + (0*8)) / 8) % bitbuffersizeqwords] ^= next1;
    bitbuffer[((i + (1*8)) / 8) % bitbuffersizeqwords] ^= next2;
    bitbuffer[((i + (2*8)) / 8) % bitbuffersizeqwords] ^= next3;
    bitbuffer[((i + (3*8)) / 8) % bitbuffersizeqwords] ^= next4;
    bitbuffer[((i + (4*8)) / 8) % bitbuffersizeqwords] ^= next5;
    bitbuffer[((i + (5*8)) / 8) % bitbuffersizeqwords] ^= next6;
    bitbuffer[((i + (6*8)) / 8) % bitbuffersizeqwords] ^= next7;
    bitbuffer[((i + (7*8)) / 8) % bitbuffersizeqwords] ^= next8;
    bitbuffer[((i + (8*8)) / 8) % bitbuffersizeqwords] ^= next9;
    bitbuffer[((i + (9*8)) / 8) % bitbuffersizeqwords] ^= next10;
    bitbuffer[((i + (10*8)) / 8) % bitbuffersizeqwords] ^= next11;
    bitbuffer[((i + (11*8)) / 8) % bitbuffersizeqwords] ^= next12;
    bitbuffer[((i + (12*8)) / 8) % bitbuffersizeqwords] ^= next13;
    bitbuffer[((i + (13*8)) / 8) % bitbuffersizeqwords] ^= next14;
    bitbuffer[((i + (14*8)) / 8) % bitbuffersizeqwords] ^= next15;
    bitbuffer[((i + (15*8)) / 8) % bitbuffersizeqwords] ^= next16;
    bitbuffer[((i + (16*8)) / 8) % bitbuffersizeqwords] ^= next17;
    bitbuffer[((i + (17*8)) / 8) % bitbuffersizeqwords] ^= next18;
    bitbuffer[((i + (18*8)) / 8) % bitbuffersizeqwords] ^= next19;
    bitbuffer[((i + (19*8)) / 8) % bitbuffersizeqwords] ^= next20;
    bitbuffer[((i + (20*8)) / 8) % bitbuffersizeqwords] ^= next21;
    bitbuffer[((i + (21*8)) / 8) % bitbuffersizeqwords] ^= next22;

    for (int j = 118960 / 8; j < (118960 / 8) + 60; j++) {
        bitbuffer[(j + (i / sizeof(uint64_t))) % bitbuffersizeqwords] = 0;
    }

    next1 = 0;
    next2 = 0;
    next3 = 0;
    next4 = 0;
    next5 = 0;
    next6 = 0;
    unsigned char final[72] = {0};
    for(; i + 32 + 40 < len; i += 32) {
        uint64_t in1;
        uint64_t in2;
        uint64_t in3;
        uint64_t in4;
        uint64_t a1, a2, a3, a4;
        uint64_t b1, b2, b3, b4;
        uint64_t c1, c2, c3, c4;
        uint64_t d1, d2, d3, d4;

        uint64_t out1;
        uint64_t out2;
        uint64_t out3;
        uint64_t out4;
        uint64_t out5;

        in1 = htobe64(*((uint64_t*) (input + i))) ^ next1 ^ htobe64(bitbuffer[(i/8) % bitbuffersizeqwords]);
        in2 = htobe64(*((uint64_t*) (input + i + (8*1)))) ^ next2 ^ htobe64(bitbuffer[(i/8 + 1) % bitbuffersizeqwords]);

        a1 = (in1 >> 17) ^ (in1 >> 55);
        a2 = (in1 << 47) ^ (in1 << 9) ^ (in1 >> 19);
        a3 = (in1 << 45) ^ (in1 >> 44);
        a4 = (in1 << 20);

        b1 = (in2 >> 17) ^ (in2 >> 55);
        b2 = (in2 << 47) ^ (in2 << 9) ^ (in2 >> 19);
        b3 = (in2 << 45) ^ (in2 >> 44);
        b4 = (in2 << 20);

        in3 = htobe64(*((uint64_t*) (input + i + (8*2)))) ^ next3 ^ a1 ^ htobe64(bitbuffer[(i/8 + 2) % bitbuffersizeqwords]);
        in4 = htobe64(*((uint64_t*) (input + i + (8*3)))) ^ next4 ^ a2 ^ b1 ^ htobe64(bitbuffer[(i/8 + 3) % bitbuffersizeqwords]);

        c1 = (in3 >> 17) ^ (in3 >> 55);
        c2 = (in3 << 47) ^ (in3 << 9) ^ (in3 >> 19);
        c3 = (in3 << 45) ^ (in3 >> 44);
        c4 = (in3 << 20);

        d1 = (in4 >> 17) ^ (in4 >> 55);
        d2 = (in4 << 47) ^ (in4 << 9) ^ (in4 >> 19);
        d3 = (in4 << 45) ^ (in4 >> 44);
        d4 = (in4 << 20);

        out1 = a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;

    }

    memcpy(final, input+i, len-i);
    *((uint64_t*) (final + (0*8))) ^= be64toh(next1);
    *((uint64_t*) (final + (1*8))) ^= be64toh(next2);
    *((uint64_t*) (final + (2*8))) ^= be64toh(next3);
    *((uint64_t*) (final + (3*8))) ^= be64toh(next4);
    *((uint64_t*) (final + (4*8))) ^= be64toh(next5);

    // up to 72 bytes left
    // RFC 1952
    for(int j = 0; j<len-i; j++) {
        crc = crctab[0][(crc >> 24) ^ final[j] ^ bitbufferbytes[(j+i) % bitbuffersizebytes]] ^ (crc << 8);
    }

    free(bitbuffer);
    return crc;
}

int
cksum_chorba_file (FILE *stream, void *resstream, uintmax_t *reslen)
{
  uint32_t buf[BUFLEN / sizeof (uint32_t)];
  uint32_t crc = 0;
  uintmax_t len = 0;
  size_t bytes_read;

  if (!stream || !resstream || !reslen)
    return -1;

# if GL_CRC_X86_64_PCLMUL
  if (cksum_debug)
    (void) pclmul_supported ();
# endif

  while ((bytes_read = fread (buf, 1, BUFLEN, stream)) > 0)
    {
      if (len + bytes_read < len)
        {
          errno = EOVERFLOW;
          return -1;
        }
      len += bytes_read;

      crc = chorba_118960_nondestructive (crc, (char const *)buf, bytes_read);

      if (feof (stream))
        break;
    }

  unsigned int crc_out = crc;
  memcpy (resstream, &crc_out, sizeof crc_out);

  *reslen = len;

  return ferror (stream) ? -1 : 0;
}

//static bool
bool
cksum_slice8 (FILE *fp, uint_fast32_t *crc_out, uintmax_t *length_out)
{
  uint32_t buf[BUFLEN / sizeof (uint32_t)];
  uint_fast32_t crc = 0;
  uintmax_t length = 0;
  size_t bytes_read;

  if (!fp || !crc_out || !length_out)
    return false;

  while ((bytes_read = fread (buf, 1, BUFLEN, fp)) > 0)
    {
      uint32_t *datap;

      if (length + bytes_read < length)
        {
          errno = EOVERFLOW;
          return false;
        }
      length += bytes_read;

      /* Process multiples of 8 bytes */
      datap = (uint32_t *)buf;
      while (bytes_read >= 8)
        {
          uint32_t first = *datap++, second = *datap++;
          crc ^= htobe32 (first);
          second = htobe32 (second);
          crc = (crctab[7][(crc >> 24) & 0xFF]
                 ^ crctab[6][(crc >> 16) & 0xFF]
                 ^ crctab[5][(crc >> 8) & 0xFF]
                 ^ crctab[4][(crc) & 0xFF]
                 ^ crctab[3][(second >> 24) & 0xFF]
                 ^ crctab[2][(second >> 16) & 0xFF]
                 ^ crctab[1][(second >> 8) & 0xFF]
                 ^ crctab[0][(second) & 0xFF]);
          bytes_read -= 8;
        }

      /* And finish up last 0-7 bytes in a byte by byte fashion */
      unsigned char *cp = (unsigned char *)datap;
      while (bytes_read--)
        crc = (crc << 8) ^ crctab[0][((crc >> 24) ^ *cp++) & 0xFF];
      if (feof (fp))
        break;
    }

  *crc_out = crc;
  *length_out = length;

  return !ferror (fp);
}

/* Calculate the checksum and length in bytes of stream STREAM.
   Return -1 on error, 0 on success.  */

int
crc_sum_stream (FILE *stream, void *resstream, uintmax_t *length)
{
  uintmax_t total_bytes = 0;
  uint_fast32_t crc = 0;

  static bool (*cksum_fp) (FILE *, uint_fast32_t *, uintmax_t *);
  if (! cksum_fp)
    {
      // if (avx512_supported ())
      //   cksum_fp = cksum_avx512;
      // else if (avx2_supported ())
      //   cksum_fp = cksum_avx2;
      // else if (pclmul_supported ())
      //   cksum_fp = cksum_pclmul;
      // else if (vmull_supported ())
      //   cksum_fp = cksum_vmull;
      // else
        cksum_fp = cksum_slice8;
    }

  if (! cksum_fp (stream, &crc, &total_bytes))
    return -1;

  *length = total_bytes;

  for (; total_bytes; total_bytes >>= 8)
    crc = (crc << 8) ^ crctab[0][((crc >> 24) ^ total_bytes) & 0xFF];
  crc = ~crc & 0xFFFFFFFF;

  unsigned int crc_out = crc;
  memcpy (resstream, &crc_out, sizeof crc_out);

  return 0;
}

/* Calculate the crc32b checksum and length in bytes of stream STREAM.
   Return -1 on error, 0 on success.  */

int
crc32b_sum_stream (FILE *stream, void *resstream, uintmax_t *reslen)
{
  uint32_t buf[BUFLEN / sizeof (uint32_t)];
  uint32_t crc = 0;
  uintmax_t len = 0;
  size_t bytes_read;

  if (!stream || !resstream || !reslen)
    return -1;

# if GL_CRC_X86_64_PCLMUL
  if (cksum_debug)
    (void) pclmul_supported ();
# endif

  while ((bytes_read = fread (buf, 1, BUFLEN, stream)) > 0)
    {
      if (len + bytes_read < len)
        {
          errno = EOVERFLOW;
          return -1;
        }
      len += bytes_read;

      //crc = crc32_update (crc, (char const *)buf, bytes_read);

      if (feof (stream))
        break;
    }

  unsigned int crc_out = crc;
  memcpy (resstream, &crc_out, sizeof crc_out);

  *reslen = len;

  return ferror (stream) ? -1 : 0;
}

/* Print the checksum and size to stdout.
   If ARGS is true, also print the FILE name.  */

void
output_crc (char const *file, int binary_file, void const *digest, bool raw,
            bool tagged, unsigned char delim, bool args, uintmax_t length)
{
  if (raw)
    {
      /* Output in network byte order (big endian).  */
      uint32_t out_int = htobe32 (*(uint32_t *)digest);
      fwrite (&out_int, 1, 32/8, stdout);
      return;
    }

  printf ("%u %ju", *(unsigned int *)digest, length);
  if (args)
    printf (" %s", file);
  putchar (delim);
}

#endif /* !CRCTAB */
