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

/** 
 * Headers for the gemmology functions 
 */
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>



#include <cstdint>

#if 0
extern "C" void
    __attribute__((import_module("wasm_gemm"), import_name("int8_multiply")))
    int8Multiply(const uint8_t* input_A,
                 float zero_point_A,
                 const int8_t* input_B,
                 const uint8_t* zero_point_B,
                 float rows_A,
                 float width,
                 float cols_B,
                 float* output);
#endif

#endif

}  // namespace contrib
}  // namespace onnxruntime
