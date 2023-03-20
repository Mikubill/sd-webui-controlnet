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
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spconv/tensorview/tensorview.h>

#include <algorithm>
#include <iostream>

namespace py = pybind11;

template <typename scalar_t, typename TPyObject>
std::vector<scalar_t> array2Vector(TPyObject arr) {
  py::array arr_np = arr;
  size_t size = arr.attr("size").template cast<size_t>();
  py::array_t<scalar_t> arr_cc = arr_np;
  std::vector<scalar_t> data(arr_cc.data(), arr_cc.data() + size);
  return data;
}

template <typename scalar_t>
std::vector<scalar_t> arrayT2Vector(py::array_t<scalar_t> arr) {
  std::vector<scalar_t> data(arr.data(), arr.data() + arr.size());
  return data;
}

template <typename scalar_t, typename TPyObject>
tv::TensorView<scalar_t> array2TensorView(TPyObject arr) {
  py::array arr_np = arr;
  py::array_t<scalar_t> arr_cc = arr_np;
  tv::Shape shape;
  for (int i = 0; i < arr_cc.ndim(); ++i) {
    shape.push_back(arr_cc.shape(i));
  }
  return tv::TensorView<scalar_t>(arr_cc.mutable_data(), shape);
}
template <typename scalar_t>
tv::TensorView<scalar_t> arrayT2TensorView(py::array_t<scalar_t> arr) {
  tv::Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  return tv::TensorView<scalar_t>(arr.mutable_data(), shape);
}
