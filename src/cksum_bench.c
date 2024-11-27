#include "config.h"

#include <stdio.h>
#include <stdlib.h>

#include "cksum.h"

void xorshift_populate(char* buffer, size_t len) {
    size_t i;
    unsigned int state = 0x123;

    for (i = 0; i < len; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        buffer[i] = (char) state;
    }
}

int main(int argc, char* argv[]) {
    uint_fast32_t hash;
    uintmax_t length;
    size_t iterations, i;
    FILE* fp;
    size_t buffer_len;
    char* buffer;

    if (argc != 3) {
        printf("Usage: %s length iterations\n", argv[0]);
        return -1;
    }

    buffer_len = atoi(argv[1]);
    iterations = atoi(argv[2]);
    buffer = calloc(1, buffer_len);
    xorshift_populate(buffer, buffer_len);
    for (i = 0; i < iterations; i++) {
        fp = fmemopen(buffer, buffer_len, "r");
        cksum_pclmul(fp, &hash, &length);
    }
    free(buffer);

    printf("Hash: %08X, length: %d\n", (unsigned int) hash, (int) length);
    return 0;
}
