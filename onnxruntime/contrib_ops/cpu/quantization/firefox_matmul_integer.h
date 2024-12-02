// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/quantization/matmul_integer_base.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

class FirefoxMatMulInteger8 final : public MatMulIntegerBase {
 public:
  FirefoxMatMulInteger8(const OpKernelInfo& info) : MatMulIntegerBase(info) {}
  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_ZERO_POINT = 2,
    IN_B_ZERO_POINT = 3
  };

  enum OutputTensors : int { OUT_Y = 0 };

 protected:
  int GetBIdx() const override { return IN_B; }
};

}  // namespace contrib
}  // namespace onnxruntime
