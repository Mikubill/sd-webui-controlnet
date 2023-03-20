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

#ifndef PARAMS_GRID_H_
#define PARAMS_GRID_H_
#include <tuple>
#include <vector>

namespace detail {
template <class scalar_t>
int getTotalSize(std::vector<scalar_t> arg) {
  return arg.size();
}

template <class scalar_t, class... TArgs>
int getTotalSize(std::vector<scalar_t> arg, std::vector<TArgs>... args) {
  return arg.size() * getTotalSize(args...);
}

template <typename scalar_t>
int getSize(std::vector<scalar_t> arg) {
  return arg.size();
}

template <int Idx, class TT, class scalar_t>
void assigner(TT &src, std::vector<int> counter, std::vector<scalar_t> &arg) {
  std::get<Idx>(src) = arg[counter[Idx]];
}

template <int Idx, class TT, class scalar_t, class... TArgs>
void assigner(TT &src, std::vector<int> counter, std::vector<scalar_t> &arg,
              std::vector<TArgs> &... args) {
  std::get<Idx>(src) = arg[counter[Idx]];
  assigner<Idx + 1>(src, counter, args...);
}
}  // namespace detail

template <class... TArgs>
std::vector<std::tuple<TArgs...>> paramsGrid(std::vector<TArgs>... args) {
  int length = detail::getTotalSize(args...);
  std::vector<int> sizes = {detail::getSize(args)...};
  int size = sizes.size();

  std::vector<std::tuple<TArgs...>> params(length);
  std::vector<int> counter(size);
  for (int i = 0; i < length; ++i) {
    detail::assigner<0>(params[i], counter, args...);
    counter[size - 1] += 1;
    for (int c = size - 1; c >= 0; --c) {
      if (counter[c] == sizes[c] && c > 0) {
        counter[c - 1] += 1;
        counter[c] = 0;
      }
    }
  }
  return params;
}

#endif
