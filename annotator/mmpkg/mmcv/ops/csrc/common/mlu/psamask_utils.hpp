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
#ifndef PSAMASK_UTILS_HPP_
#define PSAMASK_UTILS_HPP_

typedef enum {
  COLLECT = 0,
  DISTRIBUTE = 1,
} PsamaskType;

typedef enum {
  PARTITION_N = 0,
  PARTITION_H = 1,
} DimPartitionType;

struct PartitionSeg {
  int h_per_cluster;
  int n_per_cluster;
  int h_per_core;
  int n_per_core;
  DimPartitionType cluster_partition;
  DimPartitionType core_partition;
};

struct Shape {
  int n;
  int h;
  int w;
  int c;
};

struct LimitParam {
  int n;
  int h;
  int w;
};

struct PositionInCore {
  int n_start;
  int n_end;
  int h_start;
  int h_end;
  int w_start;
  int w_end;
};
#endif  // PSAMASK_UTILS_HPP_
