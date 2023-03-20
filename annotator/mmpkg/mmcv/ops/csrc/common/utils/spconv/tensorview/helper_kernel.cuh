#pragma once
namespace tv {
namespace detail {

template <typename scalar_t>
class KernelLoop {
  struct Iterator {
    __forceinline__ __device__ Iterator(scalar_t index, scalar_t delta)
        : index_(index), delta_(delta) {}
    __forceinline__ __device__ scalar_t operator*() const { return index_; }
    __forceinline__ __device__ Iterator &operator++() {
      index_ += delta_;
      return *this;
    }
    __forceinline__ __device__ bool operator!=(const Iterator &other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    scalar_t index_;
    const scalar_t delta_;
  };

 public:
  __forceinline__ __device__ KernelLoop(scalar_t begin, scalar_t delta,
                                        scalar_t end)
      : begin_(begin), delta_(delta), end_(end) {}

  __forceinline__ __device__ Iterator begin() const {
    return Iterator{begin_, delta_};
  }
  __forceinline__ __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  scalar_t begin_;
  scalar_t delta_;
  scalar_t end_;
};

}  // namespace detail

template <typename scalar_t, int NumILP = 1>
__forceinline__ __device__ detail::KernelLoop<scalar_t> KernelLoopX(
    scalar_t count) {
  return detail::KernelLoop<scalar_t>(blockIdx.x * blockDim.x + threadIdx.x,
                                      gridDim.x * blockDim.x * NumILP, count);
}

// Helper to visit indices in the range 0 <= i < count using the y-coordinate.
// Usage: for(int i : KernelLoopY(count)) { visit(i); }
template <typename scalar_t, int NumILP = 1>
__forceinline__ __device__ detail::KernelLoop<scalar_t> KernelLoopY(
    scalar_t count) {
  return detail::KernelLoop<scalar_t>(blockIdx.y * blockDim.y + threadIdx.y,
                                      gridDim.y * blockDim.y * NumILP, count);
}

// Helper to visit indices in the range 0 <= i < count using the z-coordinate.
// Usage: for(int i : KernelLoopZ(count)) { visit(i); }
template <typename scalar_t, int NumILP = 1>
__forceinline__ __device__ detail::KernelLoop<scalar_t> KernelLoopZ(
    scalar_t count) {
  return detail::KernelLoop<scalar_t>(blockIdx.z * blockDim.z + threadIdx.z,
                                      gridDim.z * blockDim.z * NumILP, count);
}

}  // namespace tv
