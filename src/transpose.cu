#include "transpose.h"

template<typename T>
struct span {
  const T* beg_;
  const T* end_;

  auto begin()->const T*;
  auto end()->const T*;
};

template<typename T> auto span<T>::begin()->const T* { return this->beg_; }
template<typename T> auto span<T>::end()->const T* { return this->end_; }

extern "C" Layout tiled(const Dimension* beg, const Dimension* end) {
  Layout out={};
  uint8_t i=0;
  for(const Dimension* d = beg; d != end; ++d) {
    if((2*i+1)<64) {
      out.shape[2*i]=d->size_px%d->tile_size_px;
      out.shape[2*i+1]=d->size_px/d->tile_size_px;
    }
    ++i;
  }
  out.rank=2*i;
  return out;
}
