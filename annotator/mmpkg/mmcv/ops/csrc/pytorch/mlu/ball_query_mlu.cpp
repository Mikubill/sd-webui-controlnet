/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "mlu_common_helper.h"

void ball_query_forward_mlu(int b, int n, int m, float min_radius,
                            float max_radius, int nsample, const Tensor new_xyz,
                            const Tensor xyz, Tensor idx) {
  auto new_xyz_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      new_xyz, new_xyz.suggest_memory_format());
  auto xyz_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      xyz, new_xyz.suggest_memory_format());
  auto idx_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      idx, new_xyz.suggest_memory_format());

  MluOpTensorDescriptor new_xyz_desc, xyz_desc, idx_desc;
  new_xyz_desc.set(new_xyz_contiguous);
  xyz_desc.set(xyz_contiguous);
  idx_desc.set(idx_contiguous);

  auto new_xyz_impl = torch_mlu::getMluTensorImpl(new_xyz_contiguous);
  auto xyz_impl = torch_mlu::getMluTensorImpl(xyz_contiguous);
  auto idx_impl = torch_mlu::getMluTensorImpl(idx_contiguous);
  auto new_xyz_ptr = new_xyz_impl->cnnlMalloc();
  auto xyz_ptr = xyz_impl->cnnlMalloc();
  auto idx_ptr = idx_impl->cnnlMalloc();

  auto handle = mluOpGetCurrentHandle();
  mluOpBallQuery(handle, new_xyz_desc.desc(), new_xyz_ptr, xyz_desc.desc(),
                 xyz_ptr, min_radius, max_radius, nsample, idx_desc.desc(),
                 idx_ptr);
}

void ball_query_forward_impl(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx);

REGISTER_DEVICE_IMPL(ball_query_forward_impl, MLU, ball_query_forward_mlu);
