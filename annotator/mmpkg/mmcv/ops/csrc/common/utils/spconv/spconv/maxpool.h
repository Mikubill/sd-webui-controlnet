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

#ifndef SPARSE_MAXPOOL_FUNCTOR_H_
#define SPARSE_MAXPOOL_FUNCTOR_H_
#include <utils/spconv/tensorview/tensorview.h>

namespace functor {
template <typename Device, typename scalar_t, typename Index>
struct SparseMaxPoolForwardFunctor {
  void operator()(const Device& d, tv::TensorView<scalar_t> outFeatures,
                  tv::TensorView<const scalar_t> inFeatures,
                  tv::TensorView<const Index> indices, int size);
};

template <typename Device, typename scalar_t, typename Index>
struct SparseMaxPoolBackwardFunctor {
  void operator()(const Device& d, tv::TensorView<const scalar_t> outFeatures,
                  tv::TensorView<const scalar_t> inFeatures,
                  tv::TensorView<const scalar_t> fout,
                  tv::TensorView<scalar_t> fin,
                  tv::TensorView<const Index> indices, int size);
};
}  // namespace functor

#endif
