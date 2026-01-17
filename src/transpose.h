#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Dimension {
  uint64_t size_px;
  uint64_t tile_size_px;
};

struct Layout {
  uint64_t shape[64];
  int64_t strides[64];
  uint8_t rank;
};

struct Layout tiled(const struct Dimension* beg, const struct Dimension* end);

#ifdef __cplusplus
}
#endif

#endif // TRANSPOSE_H
