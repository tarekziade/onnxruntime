// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#ifndef __EMSCRIPTEN__
#include "gemmology.h"
#endif

#include "firefox_matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

using Index = std::size_t;

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FirefoxMatMulInteger8,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    FirefoxMatMulInteger8);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FirefoxMatMulInteger8,
    kMSDomain,
    1,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    FirefoxMatMulInteger8);



/** Typical Call 

Input Tensor A shape: {1,171,1024}
Input Tensor B shape: {1024,1024}
A Zero Point shape: {} 
A Zero Point value: 123
B Zero Point shape: {1024}
B Zero Point is per-column: 1
Computing helper with A and B shapes.
Output Tensor Y shape: {1,171,1024}
GEMM Shape - M: 171, N: 1024, K: 1024, AIsSigned: 0, BIsSigned: 1 
Batch size: 1 

*/ 
Status FirefoxMatMulInteger8::Compute(OpKernelContext* ctx) const {
  std::cout << "FirefoxMatMulInteger8::Compute started" << std::endl;
  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  // Validate zero points
  uint8_t a_offset = 0;
  const auto* a_zero_point = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  if (a_zero_point != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *(static_cast<const uint8_t*>(a_zero_point->DataRaw()));
  }

  bool is_b_zp_per_column = false;
  uint8_t b_default_offset = 0;
  const uint8_t* b_offset_ptr = &b_default_offset;
  const auto* b_zero_point = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  if (b_zero_point != nullptr) {
    ORT_ENFORCE(IsBQuantParamSupported(b_zero_point->Shape(), b ? b->Shape() : b_shape_),
                "MatmulInteger : B zero point is not valid");
    is_b_zp_per_column = !IsScalarOr1ElementVector(b_zero_point);
    b_offset_ptr = static_cast<const uint8_t*>(b_zero_point->DataRaw());
  }

  MatMulComputeHelper helper;
  const uint8_t* b_data;
  bool b_is_signed;
  if (nullptr != b) {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), nullptr, b_zero_point ? &b_zero_point->Shape() : nullptr));
    b_data = static_cast<const uint8_t*>(b->DataRaw());
    b_is_signed = b->IsDataType<int8_t>();
  } else {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_, nullptr, b_zero_point ? &b_zero_point->Shape() : nullptr));
    b_data = static_cast<const uint8_t*>(packed_b_.get());
    b_is_signed = b_is_signed_;
  }

  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
  if (y->Shape().Size() == 0) {
    return Status::OK();
  }
  const uint8_t* a_data = static_cast<const uint8_t*>(a->DataRaw());
  auto* y_data = y->MutableData<int32_t>();

  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.AIsSigned = a->IsDataType<int8_t>();
  gemm_shape.BIsSigned = b_is_signed;

  const size_t batch_size = helper.OutputOffsets().size();

  std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(batch_size);

  for (size_t batch = 0; batch < batch_size; batch++) {
    auto& gemm_params = gemm_data_vec[batch];
    gemm_params.lda = gemm_shape.K;
    gemm_params.ZeroPointA = a_offset;
    gemm_params.ldb = gemm_shape.N;
    gemm_params.ZeroPointB = b_offset_ptr + helper.RightZeroPointOffsets()[batch];
    gemm_params.PerColumnZeroPoints = is_b_zp_per_column;
    gemm_params.ldc = gemm_shape.N;
    gemm_params.BIsPacked = bool(packed_b_);
    gemm_params.A = a_data + helper.LeftOffsets()[batch];
    gemm_params.B = b_data + helper.RightOffsets()[batch];
    gemm_params.C = y_data + helper.OutputOffsets()[batch];
  }
 
  #ifdef __EMSCRIPTEN__
  //MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());

    // Prepare output buffer
    std::vector<float> float_output(helper.M() * helper.N(), 0.0f);

    // Call the function
    // matix A (M x K) * matrix B (K x N)
    // matrix C (M x N)
    size_t rows_a = static_cast<size_t>(helper.M());
    size_t cols_b = static_cast<size_t>(helper.N());
    size_t width = static_cast<size_t>(helper.K());

    int8MultiplyAndAddBias(reinterpret_cast<const int8_t*>(a_data),
                           1.0f,  // scale factor for A
                           a_offset,
                           reinterpret_cast<const int8_t*>(b_data),
                           1.0f,  // scale factor for B
                           0, // b_zero_point
                           0,  // we don't have any bias
                           1.0f, // quantization multiplier
                           rows_a,  // rows A
                           width,  // width
                           cols_b,  // col B
                           float_output.data());

    // temporarily convert to int32
    size_t num_elements = rows_a * cols_b;

    for (size_t i = 0; i < num_elements; ++i) {
        // Convert and assign: round and cast the float to int32_t
        y_data[i] = static_cast<int32_t>(std::round(float_output[i]));

        // Optional: Clamp to int32 range (unlikely needed if input floats are reasonable)
        y_data[i] = std::clamp(
            y_data[i],
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max()
        );
    }

    // Print the output
    
    std::cout << "Output matrix:\n";
    for (Index i = 0; i < rows_a; ++i) {
        for (Index j = 0; j < cols_b; ++j) {
            std::cout << y_data[i * cols_b + j] << " ";
        }
        std::cout << "\n";
    }

  #else 

  std::cout << "Calling J'aime l'euneology" << std::endl;
  std::cout << "A shape: " << a->Shape() << std::endl;
  std::cout << "B shape: " << b->Shape() << std::endl;
  size_t a_1 = static_cast<size_t>(a->Shape()[0]);
  size_t a_2 = static_cast<size_t>(a->Shape()[1]);  // <----
  size_t b_1 = static_cast<size_t>(b->Shape()[1]);

  const int8_t* casted_b_data = reinterpret_cast<const int8_t*>(b_data);  
  const int8_t* casted_a_data = static_cast<const int8_t*>(a->DataRaw());

  // Print A data
  std::cout << "Input Tensor A (casted_a_data):" << std::endl;
  for (size_t i = 0; i < a_1; ++i) {
    for (size_t j = 0; j < a_2; ++j) {
      std::cout << static_cast<int>(casted_a_data[i * a_2 + j]) << " ";
    }
    std::cout << std::endl; // Move to the next row
  }

  // Print casted B data
  std::cout << "Input Tensor B (casted_b_data):" << std::endl;
  for (size_t i = 0; i < a_2; ++i) { // Rows of B
    for (size_t j = 0; j < b_1; ++j) {
      std::cout << static_cast<int>(casted_b_data[i * b_1 + j]) << " ";
    }
    std::cout << std::endl; // Move to the next row
  }

  // the result is dequantized to float...
  gemmology::Shift::Multiply(
      reinterpret_cast<const uint8_t*>(casted_a_data), 
      casted_b_data,  
      a_1,
      a_2,
      b_1,
      gemmology::callbacks::Write(reinterpret_cast<float*>(y_data))
  );

  // and we want int32..
  //
  // Get the shape of the tensor
  std::cout << "y data result:" << std::endl;

  size_t M = helper.M();
  size_t N = helper.N();
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      // Access the element at row i and column j
      std::cout << y_data[i * N + j] << " ";
    }
    std::cout << std::endl; // Move to the next row
  }

   #endif
  std::cout << "Exiting FirefoxMatMulInteger8::Compute" << std::endl;
  return Status::OK();
}


}  // namespace contrib
}  // namespace onnxruntime
