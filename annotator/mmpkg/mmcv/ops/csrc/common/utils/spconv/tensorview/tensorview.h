// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#include "pytorch_cpp_helper.hpp"

namespace tv {

#if defined(__NVCC__) || defined(__HIP__)
#define TV_HOST_DEVICE_INLINE __forceinline__ __device__ __host__
#define TV_DEVICE_INLINE __forceinline__ __device__
#define TV_HOST_DEVICE __device__ __host__
#define TV_ASSERT(expr) assert(expr)
#elif defined(__CUDACC_RTC__)
#define TV_ASSERT(expr) assert(expr)
#define TV_HOST_DEVICE_INLINE __forceinline__ __device__
#define TV_DEVICE_INLINE __forceinline__ __device__
#define TV_HOST_DEVICE __device__ __host__
#else
#define TV_ASSERT(x) assert(x)
#define TV_HOST_DEVICE_INLINE inline
#define TV_HOST_DEVICE
#endif

#define TV_REQUIRE(expr, ...) \
  {                           \
    if (!(expr)) {            \
      printf(__VA_ARGS__);    \
      assert(expr);           \
    }                         \
  }

#define TV_DEVICE_REQUIRE(expr, ...)                      \
  {                                                       \
    if (!(expr) && threadIdx.x == 0) printf(__VA_ARGS__); \
    assert(expr);                                         \
  }

template <class SStream, class T>
void sstream_print(SStream &ss, T val) {
  ss << val;
}

template <class SStream, class T, class... TArgs>
void sstream_print(SStream &ss, T val, TArgs... args) {
  ss << val << " ";
  sstream_print(ss, args...);
}

#define TV_ASSERT_RT_ERR(expr, ...)                     \
  {                                                     \
    if (!(expr)) {                                      \
      std::stringstream __macro_s;                      \
      __macro_s << __FILE__ << " " << __LINE__ << "\n"; \
      __macro_s << #expr << " assert failed. ";         \
      tv::sstream_print(__macro_s, __VA_ARGS__);        \
      throw std::runtime_error(__macro_s.str());        \
    }                                                   \
  }

#define TV_ASSERT_INVALID_ARG(expr, ...)                \
  {                                                     \
    if (!(expr)) {                                      \
      std::stringstream __macro_s;                      \
      __macro_s << __FILE__ << " " << __LINE__ << "\n"; \
      __macro_s << #expr << " assert failed. ";         \
      tv::sstream_print(__macro_s, __VA_ARGS__);        \
      throw std::invalid_argument(__macro_s.str());     \
    }                                                   \
  }

#define TV_CHECK_CUDA_ERR()                                    \
  {                                                            \
    auto err = cudaGetLastError();                             \
    if (err != cudaSuccess) {                                  \
      std::stringstream __macro_s;                             \
      __macro_s << __FILE__ << " " << __LINE__ << "\n";        \
      __macro_s << "cuda execution failed with error " << err; \
      throw std::runtime_error(__macro_s.str());               \
    }                                                          \
  }

struct CPU {};

#define TV_MAX_DIM 6

template <typename scalar_t, size_t MaxDim = TV_MAX_DIM>
struct SimpleVector {
 public:
  TV_HOST_DEVICE_INLINE SimpleVector(){};
  TV_HOST_DEVICE_INLINE SimpleVector(std::initializer_list<scalar_t> q) {
    TV_ASSERT(q.size() <= MaxDim);
    mSize = 0;
    for (scalar_t s : q) {
      mArray[mSize++] = s;
    }
    mSize = q.size();
  }
  SimpleVector(const std::vector<scalar_t> &arr) {
    TV_ASSERT(arr.size() <= MaxDim);
    for (size_t i = 0; i < arr.size(); ++i) {
      mArray[i] = arr[i];
    }
    mSize = arr.size();
  }
  TV_HOST_DEVICE_INLINE SimpleVector(
      const SimpleVector<scalar_t, MaxDim> &arr) {
    TV_ASSERT(arr.size() <= MaxDim);
    for (size_t i = 0; i < arr.size(); ++i) {
      mArray[i] = arr[i];
    }
    mSize = arr.size();
  }
  TV_HOST_DEVICE_INLINE scalar_t &operator[](int idx) {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < mSize);
#endif
    return mArray[idx];
  }
  TV_HOST_DEVICE_INLINE const scalar_t &operator[](int idx) const {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < mSize);
#endif
    return mArray[idx];
  }
  TV_HOST_DEVICE_INLINE void push_back(scalar_t s) {
#ifdef TV_DEBUG
    TV_ASSERT(mSize < MaxDim);
#endif
    mArray[mSize] = s;
    mSize++;
  }
  TV_HOST_DEVICE_INLINE void pop_back() {
#ifdef TV_DEBUG
    TV_ASSERT(mSize > 0);
#endif
    mSize--;
  }

  TV_HOST_DEVICE_INLINE size_t size() const { return mSize; }
  TV_HOST_DEVICE_INLINE const scalar_t *data() const { return mArray; }
  TV_HOST_DEVICE_INLINE size_t empty() const { return mSize == 0; }

  typedef size_t size_type;

  class iterator {
   public:
    typedef iterator self_type;
    typedef scalar_t value_type;
    typedef scalar_t &reference;
    typedef scalar_t *pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef std::ptrdiff_t difference_type;
    TV_HOST_DEVICE_INLINE iterator(pointer ptr) : ptr_(ptr) {}
    TV_HOST_DEVICE_INLINE self_type operator++(int junk) {
      self_type i = *this;
      ptr_++;
      return i;
    }
    TV_HOST_DEVICE_INLINE self_type operator++() {
      ptr_++;
      return *this;
    }
    TV_HOST_DEVICE_INLINE reference operator*() { return *ptr_; }
    TV_HOST_DEVICE_INLINE pointer operator->() { return ptr_; }
    TV_HOST_DEVICE_INLINE bool operator==(const self_type &rhs) {
      return ptr_ == rhs.ptr_;
    }
    TV_HOST_DEVICE_INLINE bool operator!=(const self_type &rhs) {
      return ptr_ != rhs.ptr_;
    }

   private:
    pointer ptr_;
  };

  class const_iterator {
   public:
    typedef const_iterator self_type;
    typedef scalar_t value_type;
    typedef const scalar_t &reference;
    typedef const scalar_t *pointer;
    typedef std::ptrdiff_t difference_type;
    typedef std::forward_iterator_tag iterator_category;
    TV_HOST_DEVICE_INLINE const_iterator(pointer ptr) : ptr_(ptr) {}
    TV_HOST_DEVICE_INLINE self_type operator++(int junk) {
      self_type i = *this;
      ptr_++;
      return i;
    }
    TV_HOST_DEVICE_INLINE self_type operator++() {
      ptr_++;
      return *this;
    }
    TV_HOST_DEVICE_INLINE reference operator*() { return *ptr_; }
    TV_HOST_DEVICE_INLINE pointer operator->() { return ptr_; }
    TV_HOST_DEVICE_INLINE bool operator==(const self_type &rhs) {
      return ptr_ == rhs.ptr_;
    }
    TV_HOST_DEVICE_INLINE bool operator!=(const self_type &rhs) {
      return ptr_ != rhs.ptr_;
    }

   private:
    pointer ptr_;
  };

  TV_HOST_DEVICE_INLINE iterator begin() { return iterator(mArray); }

  TV_HOST_DEVICE_INLINE iterator end() { return iterator(mArray + mSize); }

  TV_HOST_DEVICE_INLINE const_iterator begin() const {
    return const_iterator(mArray);
  }

  TV_HOST_DEVICE_INLINE const_iterator end() const {
    return const_iterator(mArray + mSize);
  }
  TV_HOST_DEVICE_INLINE const_iterator cbegin() const {
    return const_iterator(mArray);
  }

  TV_HOST_DEVICE_INLINE const_iterator cend() const {
    return const_iterator(mArray + mSize);
  }

 protected:
  scalar_t mArray[MaxDim];
  size_t mSize = 0;
};

template <typename scalar_t, size_t MaxDim>
bool operator==(const SimpleVector<scalar_t, MaxDim> &lfs,
                const SimpleVector<scalar_t, MaxDim> &rfs) {
  if (lfs.size() != rfs.size()) return false;
  for (size_t i = 0; i < lfs.size(); ++i) {
    if (lfs[i] != rfs[i]) return false;
  }
  return true;
}

template <typename scalar_t, size_t MaxDim>
bool operator!=(const SimpleVector<scalar_t, MaxDim> &lfs,
                const SimpleVector<scalar_t, MaxDim> &rfs) {
  return !(lfs == rfs);
}

struct Slice {
  template <class... Integers>
  TV_HOST_DEVICE_INLINE Slice(Integers... ints) {
    static_assert(sizeof...(ints) <= 3, "slice init must smaller than 3");
    SimpleVector<int, 3> slices{int(ints)...};
    mSlices[0] = -1;
    mSlices[1] = -1;
    mSlices[2] = -1;
    for (size_t i = 0; i < slices.size(); ++i) {
      mSlices[i] = slices[i];
    }
  }

  TV_HOST_DEVICE_INLINE Slice() {
    mSlices[0] = -1;
    mSlices[1] = -1;
    mSlices[2] = -1;
  }
  template <typename scalar_t>
  TV_HOST_DEVICE_INLINE Slice(std::initializer_list<scalar_t> slice) {
    mSlices[0] = -1;
    mSlices[1] = -1;
    mSlices[2] = -1;
    TV_ASSERT(slice.size() <= 3);
    int idx = 0;
    for (scalar_t s : slice) {
      mSlices[idx] = int(s);
      ++idx;
    }
  }
  TV_HOST_DEVICE_INLINE int &operator[](int idx) {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < 3);
#endif
    return mSlices[idx];
  }
  TV_HOST_DEVICE_INLINE const int &operator[](int idx) const {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < 3);
#endif
    return mSlices[idx];
  }

 protected:
  int mSlices[3];
};

template <size_t MaxDim = TV_MAX_DIM>
struct ShapeBase : public SimpleVector<int, MaxDim> {
  TV_HOST_DEVICE_INLINE ShapeBase() : SimpleVector<int, MaxDim>(){};
  TV_HOST_DEVICE_INLINE ShapeBase(std::initializer_list<int> shape)
      : SimpleVector<int, MaxDim>(shape) {}

  template <typename scalar_t, template <class...> class Container>
  ShapeBase(Container<scalar_t> shape) : SimpleVector<int, MaxDim>(shape) {}
  TV_HOST_DEVICE_INLINE ShapeBase(const ShapeBase<MaxDim> &shape)
      : SimpleVector<int, MaxDim>(shape) {}
  ShapeBase(const std::vector<int> &arr) : SimpleVector<int, MaxDim>(arr) {}

  ShapeBase<MaxDim> &operator=(const ShapeBase<MaxDim> &shape) = default;
  TV_HOST_DEVICE_INLINE ShapeBase<MaxDim> subshape(int start, int end) const {
#ifdef TV_DEBUG
    TV_ASSERT(start >= 0 && end < this->mSize && end > start);
#endif
    ShapeBase<MaxDim> shape;
    for (int i = start; i < end; ++i) {
      shape.push_back(this->mArray[i]);
    }
    return shape;
  }
  TV_HOST_DEVICE_INLINE ShapeBase<MaxDim> subshape(int start) const {
#ifdef TV_DEBUG
    TV_ASSERT(start >= 0 && start <= this->mSize);
#endif
    ShapeBase<MaxDim> shape;
    for (int i = start; i < this->mSize; ++i) {
      shape.push_back(this->mArray[i]);
    }
    return shape;
  }

  TV_HOST_DEVICE_INLINE size_t size() const {
    if (this->mSize == 0) return 0;
    size_t s = 1;
    for (int i = 0; i < int(this->mSize); ++i) {
      s *= this->mArray[i];
    }
    return s;
  }
  TV_HOST_DEVICE_INLINE size_t ndim() const { return this->mSize; }
  TV_HOST_DEVICE_INLINE ShapeBase<MaxDim> squeeze() const {
    ShapeBase<MaxDim> shape;
    for (int i = 0; i < this->mSize; ++i) {
      if (this->mArray[i] != 1) shape.push_back(this->mArray[i]);
    }
    return shape;
  }
  TV_HOST_DEVICE_INLINE ShapeBase<MaxDim> squeeze(int dim) const {
    ShapeBase<MaxDim> shape;
    for (int i = 0; i < this->mSize; ++i) {
      if (i != dim || this->mArray[i] != 1) shape.push_back(this->mArray[i]);
    }
    return shape;
  }
};

using Shape = ShapeBase<TV_MAX_DIM>;

template <class... Inds>
TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(std::vector<int> &shape,
                                           Inds... indexes) {
  unsigned offset = 0;
  unsigned m = 1;
  int indexes_vec[sizeof...(indexes)] = {indexes...};
#ifdef TV_DEBUG
  TV_ASSERT(sizeof...(indexes) == shape.size());
#endif
#pragma unroll
  for (int i = sizeof...(indexes) - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(std::vector<int> &shape,
                                           std::vector<int> &indexes_vec) {
  unsigned offset = 0;
  unsigned m = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

template <class... Inds>
TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(const Shape &shape,
                                           Inds... indexes) {
  unsigned offset = 0;
  unsigned m = 1;
  int indexes_vec[sizeof...(indexes)] = {indexes...};
#pragma unroll
  for (int i = sizeof...(indexes) - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(const Shape &shape,
                                           const Shape &indexes_vec) {
  unsigned offset = 0;
  unsigned m = 1;
  for (int i = indexes_vec.ndim() - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(const Index *indexes,
                                           const Index *shape) {
  unsigned offset = 0;
  unsigned m = 1;
#pragma unroll
  for (int i = NDim - 1; i >= 0; --i) {
    offset += m * indexes[i];
    m *= shape[i];
  }
  return offset;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE_INLINE Index rowArrayIdxInv(Index index, Index *output,
                                           const Index *shape) {
#pragma unroll
  for (int i = NDim - 1; i >= 0; --i) {
    output[i] = index % shape[i];
    index -= output[i];
    index /= shape[i];
  }
  return index;
}

template <int N>
struct ArrayIndexRowMajor {
  TV_HOST_DEVICE_INLINE static unsigned run(const Shape &shape,
                                            const Shape &indexes) {
    return indexes[N - 1] +
           shape[N - 1] * ArrayIndexRowMajor<N - 1>::run(shape, indexes);
  }
};

template <>
struct ArrayIndexRowMajor<0> {
  TV_HOST_DEVICE_INLINE static unsigned run(const Shape &shape,
                                            const Shape &indexes) {
    return 0;
  }
};

namespace detail {
template <typename scalar_t>
constexpr const char *simpleTypeName(scalar_t val = scalar_t());
template <>
constexpr const char *simpleTypeName(float val) {
  return "float32";
}
template <>
constexpr const char *simpleTypeName(double val) {
  return "float64";
}
template <>
constexpr const char *simpleTypeName(int val) {
  return "int32";
}
template <>
constexpr const char *simpleTypeName(unsigned val) {
  return "uint32";
}
template <>
constexpr const char *simpleTypeName(long val) {
  return "int64";
}
template <>
constexpr const char *simpleTypeName(unsigned long val) {
  return "uint64";
}
};  // namespace detail

template <typename scalar_t, int Rank = -1>
struct TensorView {
  TV_HOST_DEVICE_INLINE TensorView() {}
  explicit TV_HOST_DEVICE_INLINE TensorView(scalar_t *ptr, Shape shape)
      : mPtr(ptr), mShape(shape) {}
  template <class... Integers>
  explicit TV_HOST_DEVICE_INLINE TensorView(scalar_t *ptr, Integers... shapes)
      : mPtr(ptr) {
    mShape = {int(shapes)...};
  }

  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> &assign(
      const TensorView<scalar_t, Rank> &tensor) {
    TV_REQUIRE(tensor.shape() == shape(), "you must provide same input size%s",
               "\n");
    scalar_t *ptr = mPtr;
    const scalar_t *other_ptr = tensor.data();
    for (size_t i = 0; i < size(); ++i) *(ptr++) = *(other_ptr++);
    return *this;
  }

  template <typename T1>
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> &assign(
      std::initializer_list<T1> seq) {
    TV_REQUIRE(seq.size() == size(), "you must provide same input size%s",
               "\n");
    scalar_t *ptr = mPtr;
    for (const T1 &s : seq) *(ptr++) = scalar_t(s);
    return *this;
  }

  template <class... Inds>
  TV_HOST_DEVICE_INLINE scalar_t &operator()(Inds... inds) {
#ifdef TV_DEBUG
    int idxes[sizeof...(Inds)]{int(inds)...};
    TV_REQUIRE(sizeof...(inds) == mShape.ndim(),
               "you provide %d indexes, but dim is %d\n", sizeof...(inds),
               mShape.ndim());
    for (int i = 0; i < sizeof...(inds); ++i) {
      TV_REQUIRE(idxes[i] >= 0 && idxes[i] < mShape[i],
                 "index-%d(%d) out-of-range: [0, %d)\n", i, idxes[i],
                 mShape[i]);
    }
#endif
    return mPtr[rowArrayIdx(mShape, int(inds)...)];
  }
  template <class... Inds>
  TV_HOST_DEVICE_INLINE const scalar_t &operator()(Inds... inds) const {
#ifdef TV_DEBUG
    int idxes[sizeof...(Inds)]{int(inds)...};
    TV_REQUIRE(sizeof...(inds) == mShape.ndim(),
               "you provide %d indexes, but dim is %d\n", sizeof...(inds),
               mShape.ndim());
    for (int i = 0; i < sizeof...(inds); ++i) {
      TV_REQUIRE(idxes[i] >= 0 && idxes[i] < mShape[i],
                 "index-%d(%d) out-of-range: [0, %d)\n", i, idxes[i],
                 mShape[i]);
    }
#endif
    return mPtr[rowArrayIdx(mShape, int(inds)...)];
  }
  TV_HOST_DEVICE_INLINE scalar_t &operator()() {
#if defined TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mPtr != nullptr,
                      "you want get value but the view is empty.%s", "\n");
    TV_DEVICE_REQUIRE(mShape.ndim() == 0,
                      "you provide 0 indexes, but dim is %ld\n", mShape.ndim());
#else
    TV_REQUIRE(mPtr != nullptr, "you want get value but the view is empty.%s",
               "\n");
    TV_REQUIRE(mShape.ndim() == 0, "you provide 0 indexes, but dim is %ld\n",
               mShape.ndim());
#endif
#endif
    return mPtr[0];
  }
  TV_HOST_DEVICE_INLINE const scalar_t &operator()() const {
#if defined TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mPtr != nullptr,
                      "you want get value but the view is empty.%s", "\n");
    TV_DEVICE_REQUIRE(mShape.ndim() == 0,
                      "you provide 0 indexes, but dim is %ld\n", mShape.ndim());
#else
    TV_REQUIRE(mPtr != nullptr, "you want get value but the view is empty.%s",
               "\n");
    TV_REQUIRE(mShape.ndim() == 0, "you provide 0 indexes, but dim is %ld\n",
               mShape.ndim());
#endif
#endif
    return mPtr[0];
  }

  template <class T1>
  TV_HOST_DEVICE_INLINE scalar_t &operator()(T1 i1) {
#if defined TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mShape.ndim() == 1,
                      "you provide 1 indexes, but dim is %ld\n", mShape.ndim());
    TV_DEVICE_REQUIRE(i1 >= 0 && i1 < mShape[0],
                      "index-%d(%d) out-of-range: [0, %d)\n", 0, i1, mShape[0]);
#else
    TV_REQUIRE(mShape.ndim() == 1, "you provide 1 indexes, but dim is %ld\n",
               mShape.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < mShape[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, i1, mShape[0]);
#endif
#endif
    return mPtr[i1];
  }
  template <class T1, class T2>
  TV_HOST_DEVICE_INLINE scalar_t &operator()(T1 i1, T2 i2) {
#ifdef TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mShape.ndim() == 2,
                      "you provide 2 indexes, but dim is %ld\n", mShape.ndim());
    TV_DEVICE_REQUIRE(i1 >= 0 && i1 < mShape[0],
                      "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1),
                      mShape[0]);
    TV_DEVICE_REQUIRE(i2 >= 0 && i2 < mShape[1],
                      "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2),
                      mShape[1]);
#else
    TV_REQUIRE(mShape.ndim() == 2, "you provide 2 indexes, but dim is %ld\n",
               mShape.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < mShape[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), mShape[0]);
    TV_REQUIRE(i2 >= 0 && i2 < mShape[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), mShape[1]);
#endif
#endif
    return mPtr[i1 * mShape[1] + i2];
  }
  template <class T1, class T2, class T3>
  TV_HOST_DEVICE_INLINE scalar_t &operator()(T1 i1, T2 i2, T3 i3) {
#ifdef TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mShape.ndim() == 3,
                      "you provide 3 indexes, but dim is %ld\n", mShape.ndim());
    TV_DEVICE_REQUIRE(i1 >= 0 && i1 < mShape[0],
                      "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1),
                      mShape[0]);
    TV_DEVICE_REQUIRE(i2 >= 0 && i2 < mShape[1],
                      "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2),
                      mShape[1]);
    TV_DEVICE_REQUIRE(i3 >= 0 && i3 < mShape[2],
                      "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3),
                      mShape[2]);
#else
    TV_REQUIRE(mShape.ndim() == 3, "you provide 3 indexes, but dim is %ld\n",
               mShape.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < mShape[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), mShape[0]);
    TV_REQUIRE(i2 >= 0 && i2 < mShape[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), mShape[1]);
    TV_REQUIRE(i3 >= 0 && i3 < mShape[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), mShape[2]);
#endif
#endif
    return mPtr[(i1 * mShape[1] + i2) * mShape[2] + i3];
  }
  template <class T1, class T2, class T3, class T4>
  TV_HOST_DEVICE_INLINE scalar_t &operator()(T1 i1, T2 i2, T3 i3, T4 i4) {
#ifdef TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mShape.ndim() == 4,
                      "you provide 4 indexes, but dim is %ld\n", mShape.ndim());
    TV_DEVICE_REQUIRE(i1 >= 0 && i1 < mShape[0],
                      "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1),
                      mShape[0]);
    TV_DEVICE_REQUIRE(i2 >= 0 && i2 < mShape[1],
                      "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2),
                      mShape[1]);
    TV_DEVICE_REQUIRE(i3 >= 0 && i3 < mShape[2],
                      "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3),
                      mShape[2]);
    TV_DEVICE_REQUIRE(i4 >= 0 && i4 < mShape[3],
                      "index-%d(%d) out-of-range: [0, %d)\n", 3, int(i4),
                      mShape[3]);
#else
    TV_REQUIRE(mShape.ndim() == 4, "you provide 4 indexes, but dim is %ld\n",
               mShape.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < mShape[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), mShape[0]);
    TV_REQUIRE(i2 >= 0 && i2 < mShape[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), mShape[1]);
    TV_REQUIRE(i3 >= 0 && i3 < mShape[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), mShape[2]);
    TV_REQUIRE(i4 >= 0 && i4 < mShape[3],
               "index-%d(%d) out-of-range: [0, %d)\n", 3, int(i4), mShape[3]);
#endif
#endif
    return mPtr[((i1 * mShape[1] + i2) * mShape[2] + i3) * mShape[3] + i4];
  }

  template <class T1>
  TV_HOST_DEVICE_INLINE const scalar_t &operator()(T1 i1) const {
#ifdef TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mShape.ndim() == 1,
                      "you provide 1 indexes, but dim is %ld\n", mShape.ndim());
    TV_DEVICE_REQUIRE(i1 >= 0 && i1 < mShape[0],
                      "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1),
                      mShape[0]);
#else
    TV_REQUIRE(mShape.ndim() == 1, "you provide 1 indexes, but dim is %ld\n",
               mShape.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < mShape[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), mShape[0]);
#endif
#endif
    return mPtr[i1];
  }
  template <class T1, class T2>
  TV_HOST_DEVICE_INLINE const scalar_t &operator()(T1 i1, T2 i2) const {
#ifdef TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mShape.ndim() == 2,
                      "you provide 2 indexes, but dim is %ld\n", mShape.ndim());
    TV_DEVICE_REQUIRE(i1 >= 0 && i1 < mShape[0],
                      "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1),
                      mShape[0]);
    TV_DEVICE_REQUIRE(i2 >= 0 && i2 < mShape[1],
                      "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2),
                      mShape[1]);
#else
    TV_REQUIRE(mShape.ndim() == 2, "you provide 2 indexes, but dim is %ld\n",
               mShape.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < mShape[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), mShape[0]);
    TV_REQUIRE(i2 >= 0 && i2 < mShape[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), mShape[1]);

#endif
#endif
    return mPtr[i1 * mShape[1] + i2];
  }
  template <class T1, class T2, class T3>
  TV_HOST_DEVICE_INLINE const scalar_t &operator()(T1 i1, T2 i2, T3 i3) const {
#ifdef TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mShape.ndim() == 3,
                      "you provide 3 indexes, but dim is %ld\n", mShape.ndim());
    TV_DEVICE_REQUIRE(i1 >= 0 && i1 < mShape[0],
                      "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1),
                      mShape[0]);
    TV_DEVICE_REQUIRE(i2 >= 0 && i2 < mShape[1],
                      "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2),
                      mShape[1]);
    TV_DEVICE_REQUIRE(i3 >= 0 && i3 < mShape[2],
                      "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3),
                      mShape[2]);
#else
    TV_REQUIRE(mShape.ndim() == 3, "you provide 3 indexes, but dim is %ld\n",
               mShape.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < mShape[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), mShape[0]);
    TV_REQUIRE(i2 >= 0 && i2 < mShape[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), mShape[1]);
    TV_REQUIRE(i3 >= 0 && i3 < mShape[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), mShape[2]);
#endif
#endif
    return mPtr[(i1 * mShape[1] + i2) * mShape[2] + i3];
  }
  template <class T1, class T2, class T3, class T4>
  TV_HOST_DEVICE_INLINE const scalar_t &operator()(T1 i1, T2 i2, T3 i3,
                                                   T4 i4) const {
#ifdef TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(mShape.ndim() == 4,
                      "you provide 4 indexes, but dim is %ld\n", mShape.ndim());
    TV_DEVICE_REQUIRE(i1 >= 0 && i1 < mShape[0],
                      "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1),
                      mShape[0]);
    TV_DEVICE_REQUIRE(i2 >= 0 && i2 < mShape[1],
                      "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2),
                      mShape[1]);
    TV_DEVICE_REQUIRE(i3 >= 0 && i3 < mShape[2],
                      "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3),
                      mShape[2]);
    TV_DEVICE_REQUIRE(i4 >= 0 && i4 < mShape[3],
                      "index-%d(%d) out-of-range: [0, %d)\n", 3, int(i4),
                      mShape[3]);
#else
    TV_REQUIRE(mShape.ndim() == 4, "you provide 4 indexes, but dim is %ld\n",
               mShape.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < mShape[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), mShape[0]);
    TV_REQUIRE(i2 >= 0 && i2 < mShape[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), mShape[1]);
    TV_REQUIRE(i3 >= 0 && i3 < mShape[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), mShape[2]);
    TV_REQUIRE(i4 >= 0 && i4 < mShape[3],
               "index-%d(%d) out-of-range: [0, %d)\n", 3, int(i4), mShape[3]);
#endif
#endif
    return mPtr[((i1 * mShape[1] + i2) * mShape[2] + i3) * mShape[3] + i4];
  }

  TV_HOST_DEVICE_INLINE scalar_t &operator[](int idx) {
#ifdef TV_DEBUG
#if defined(__CUDA_ARCH__)
    TV_DEVICE_REQUIRE(idx >= 0 && idx < size(),
                      "index(%d) out-of-range: [0, %ld)\n", int(idx), size());
#else
    TV_REQUIRE(idx >= 0 && idx < size(), "index(%d) out-of-range: [0, %ld)\n",
               int(idx), size());
#endif
#endif
    return mPtr[idx];
  }
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> operator[](
      SimpleVector<Slice> slice_vec) {
    return _subview(slice_vec);
  }
  TV_HOST_DEVICE_INLINE const TensorView<scalar_t, Rank> operator[](
      SimpleVector<Slice> slice_vec) const {
    return _subview(slice_vec);
  }
  TV_HOST_DEVICE_INLINE bool empty() const { return mPtr == nullptr; }
  TV_HOST_DEVICE_INLINE scalar_t *data() { return mPtr; }
  TV_HOST_DEVICE_INLINE const scalar_t *data() const { return mPtr; }
  TV_HOST_DEVICE_INLINE const Shape &shape() const { return mShape; }
  TV_HOST_DEVICE_INLINE int dim(int idx) const { return mShape[idx]; }
  TV_HOST_DEVICE_INLINE int ndim() const { return mShape.ndim(); }
  template <class... Inds>
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> &reshape(Inds... newShapes) {
    Shape shapes{int(newShapes)...};
    TV_ASSERT(shapes.size() == size());
    mShape = shapes;
    return *this;
  }
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> &reshape(Shape shapes) {
    TV_ASSERT(shapes.size() == size());
    mShape = shapes;
    return *this;
  }
  template <class... Inds>
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> view(
      Inds... newShapes) const {
    Shape shapes{int(newShapes)...};
    for (size_t i = 0; i < shapes.ndim(); ++i) {
      if (shapes[i] == -1) {
        shapes[i] = 1;
        shapes[i] = size() / shapes.size();
        break;
      }
    }
    TV_ASSERT(shapes.size() == size());
    return TensorView<scalar_t, Rank>(mPtr, shapes);
  }
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> view(Shape shapes) const {
    TV_ASSERT(shapes.size() == size());
    return TensorView<scalar_t, Rank>(mPtr, shapes);
  }
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> squeeze() const {
    return TensorView<scalar_t, Rank>(mPtr, mShape.squeeze());
  }
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> squeeze(int dim) const {
    return TensorView<scalar_t, Rank>(mPtr, mShape.squeeze(dim));
  }
  TV_HOST_DEVICE_INLINE size_t size() const { return mShape.size(); }

  template <class... Slices>
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> subview(
      Slice slice, Slices... slices) const {
    return subview<float, Slice, Slices...>(slice, slices...);
  }
  template <class T2 = float, class... Slices>
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> subview(
      Slices... slices) const {
    Slice slice_vec[sizeof...(Slices)] = {to_slice(slices)...};
    Shape new_shape{to_slice(slices)[0]...};
    Shape start{to_slice(slices)[0]...};
    TV_ASSERT(new_shape.ndim() <= mShape.ndim());
    TV_ASSERT(new_shape.ndim() != 0);
    size_t idxsize = new_shape.ndim();
    for (size_t i = idxsize; i < mShape.ndim(); ++i) {
      new_shape.push_back(0);
      start.push_back(0);
    }
#pragma unroll
    for (size_t i = 0; i < sizeof...(Slices); ++i) {
      if (slice_vec[i][1] != -1) {
        new_shape[i] = slice_vec[i][1] - slice_vec[i][0];
        TV_ASSERT(new_shape[i] >= 0);
      } else {
        new_shape[i] = 1;
      }
    }
    auto offset = rowArrayIdx(mShape, start);
#pragma unroll
    for (size_t i = sizeof...(Slices); i < mShape.ndim(); ++i) {
      new_shape[i] = mShape[i];
      TV_ASSERT(new_shape[i] >= 0);
    }
    Shape reduced_shape;
#pragma unroll
    for (size_t i = 0; i < sizeof...(Slices); ++i) {
      if (slice_vec[i][1] != -1) {
        reduced_shape.push_back(new_shape[i]);
      }
    }
#pragma unroll
    for (size_t i = sizeof...(Slices); i < mShape.ndim(); ++i) {
      reduced_shape.push_back(new_shape[i]);
    }
    return TensorView<scalar_t, Rank>(mPtr + offset, reduced_shape);
  }

  template <class... Integers>
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> subview(int id,
                                                           Integers... ints) {
    Shape start = {id, ints...};
    for (int i = 1 + sizeof...(ints); i < ndim(); ++i) {
      start.push_back(0);
    }
    return TensorView<scalar_t, Rank>(mPtr + rowArrayIdx(mShape, start),
                                      mShape.subshape(sizeof...(ints) + 1));
  }

  std::string repr() const {
    std::ostringstream ss;
    if (empty()) return "";
    if (mShape.ndim() == 0) {
      ss << *mPtr;
      ss << "Tensor: dtype=" << detail::simpleTypeName<scalar_t>();
      return ss.str();
    }
    Shape counter = mShape;
    auto tensor_flat = this->view(-1);
    for (int i = 0; i < counter.ndim(); ++i) {
      counter[i] = 0;
      ss << "[";
    }
    for (size_t i = 0; i < this->size(); ++i) {
      ss << tensor_flat(rowArrayIdx(mShape, counter));
      counter[counter.ndim() - 1] += 1;
      int inc_count = 0;
      bool print_comma = true;
      for (int c = counter.ndim() - 1; c >= 0; --c) {
        if (counter[c] == this->dim(c) && c > 0) {
          ++inc_count;
          counter[c - 1] += 1;
          counter[c] = 0;
          print_comma = false;
        }
      }
      if (print_comma && i != this->size() - 1) ss << ", ";
      for (int j = 0; j < inc_count; ++j) {
        ss << "]";
      }
      if (i != this->size() - 1) {
        if (inc_count != 0) ss << "\n";
        for (int j = 0; j < inc_count; ++j) {
          ss << "[";
        }
      }
    }
    ss << "]";
    ss << "Tensor: dtype=" << detail::simpleTypeName<scalar_t>();
    return ss.str();
  }

 protected:
  // TODO: make this function public.
  // currently this function is called unexpectedly when using subview({0, 0}).
  TV_HOST_DEVICE_INLINE TensorView<scalar_t, Rank> _subview(
      SimpleVector<Slice> slice_vec) {
    Shape new_shape;
    for (int i = 0; i < slice_vec.size(); ++i) {
      new_shape.push_back(slice_vec[i][0]);
    }
    Shape start = new_shape;
    TV_ASSERT(new_shape.ndim() <= mShape.ndim());
    TV_ASSERT(new_shape.ndim() != 0);
    size_t idxsize = new_shape.ndim();
    for (size_t i = idxsize; i < mShape.ndim(); ++i) {
      new_shape.push_back(0);
      start.push_back(0);
    }
    for (size_t i = 0; i < slice_vec.size(); ++i) {
      if (slice_vec[i][1] != -1) {
        new_shape[i] = slice_vec[i][1] - slice_vec[i][0];
        TV_ASSERT(new_shape[i] >= 0);
      } else {
        new_shape[i] = 1;  // reduce dim
      }
    }
    auto offset = rowArrayIdx(mShape, start);
    for (size_t i = slice_vec.size(); i < mShape.ndim(); ++i) {
      new_shape[i] = mShape[i];
      TV_ASSERT(new_shape[i] >= 0);
    }
    Shape reduced_shape;
    for (size_t i = 0; i < slice_vec.size(); ++i) {
      if (slice_vec[i][1] != -1) {
        reduced_shape.push_back(new_shape[i]);
      }
    }
    for (size_t i = slice_vec.size(); i < mShape.ndim(); ++i) {
      reduced_shape.push_back(new_shape[i]);
    }
    return TensorView<scalar_t, Rank>(mPtr + offset, reduced_shape);
  }
  template <typename T1>
  TV_HOST_DEVICE_INLINE Slice to_slice(T1 s) const {
    return Slice{int(s), -1, -1};
  }

  TV_HOST_DEVICE_INLINE Slice to_slice(Slice s) const { return Slice(s); }

  scalar_t *mPtr = nullptr;
  Shape mShape;
};

template <typename Os, typename scalar_t, int Rank>
Os &operator<<(Os &os, const TensorView<scalar_t, Rank> &dt) {
  os << dt.repr();
  return os;
}

template <typename Os, typename scalar_t, int Rank>
Os &operator<<(Os &os, const TensorView<const scalar_t, Rank> &dt) {
  os << dt.repr();
  return os;
}

namespace detail {
template <typename scalar_t>
constexpr const char *printfTypeFormat(scalar_t val = scalar_t());
template <>
constexpr const char *printfTypeFormat(float val) {
  return "%.2f";
}
template <>
constexpr const char *printfTypeFormat(double val) {
  return "%.2f";
}
template <>
constexpr const char *printfTypeFormat(int val) {
  return "%d";
}
template <>
constexpr const char *printfTypeFormat(unsigned val) {
  return "%u";
}
template <>
constexpr const char *printfTypeFormat(long val) {
  return "%ld";
}
template <>
constexpr const char *printfTypeFormat(unsigned long val) {
  return "%lu";
}
};  // namespace detail

template <typename scalar_t>
TV_HOST_DEVICE void printTensorView(const TensorView<scalar_t> tensor,
                                    const char *format) {
  if (tensor.empty()) return;
  if (tensor.ndim() == 0) {
    printf(format, tensor());
    printf("\n");
    return;
  }
  Shape counter = tensor.shape();
  auto tensor_flat = tensor.view(-1);
  for (int i = 0; i < counter.ndim(); ++i) {
    counter[i] = 0;
    printf("[");
  }
  for (size_t i = 0; i < tensor.size(); ++i) {
    printf(format, tensor_flat(rowArrayIdx(tensor.shape(), counter)));
    counter[counter.ndim() - 1] += 1;
    int inc_count = 0;
    bool print_comma = true;
    for (int c = counter.ndim() - 1; c >= 0; --c) {
      if (counter[c] == tensor.dim(c) && c > 0) {
        ++inc_count;
        counter[c - 1] += 1;
        counter[c] = 0;
        print_comma = false;
      }
    }
    if (print_comma && i != tensor.size() - 1) printf(", ");
    for (int j = 0; j < inc_count; ++j) {
      printf("]");
    }
    if (i != tensor.size() - 1) {
      if (inc_count != 0) printf("\n");
      for (int j = 0; j < inc_count; ++j) {
        printf("[");
      }
    }
  }
  printf("]\n");
}

template <typename scalar_t>
TV_HOST_DEVICE void printTensorView(TensorView<scalar_t> tensor) {
  using Traw = typename std::remove_const<scalar_t>::type;
  return printTensorView(tensor, detail::printfTypeFormat<Traw>());
}
template <typename scalar_t>
TV_HOST_DEVICE void printTensorView(const scalar_t *ptr, Shape shape) {
  using Traw = typename std::remove_const<scalar_t>::type;
  return printTensorView(TensorView<const scalar_t>(ptr, shape),
                         detail::printfTypeFormat<Traw>());
}
template <typename scalar_t>
TV_HOST_DEVICE void printTensorView(const scalar_t *ptr, Shape shape,
                                    const char *format) {
  return printTensorView(TensorView<const scalar_t>(ptr, shape), format);
}

}  // namespace tv
