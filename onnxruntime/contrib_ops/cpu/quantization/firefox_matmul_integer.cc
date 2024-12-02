// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iostream>
#include <vector>
#include <cassert>
#include "firefox_matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include <chrono> // For time measurement


// Define aliases for convenience
using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;
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


Status FirefoxMatMulInteger8::Compute(OpKernelContext* ctx) const {
  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);
  uint8_t a_offset = 0;
  const auto* a_zero_point = ctx->Input<Tensor>(IN_A_ZERO_POINT);

  if (a_zero_point != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *(static_cast<const uint8_t*>(a_zero_point->DataRaw()));
  }  

  uint8_t b_default_offset = 0;
  const auto* b_zero_point = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  bool b_is_signed;
  const uint8_t* b_offset_ptr = &b_default_offset;
  bool is_b_zp_per_column = false;

  if (b_zero_point != nullptr) {
    ORT_ENFORCE(IsBQuantParamSupported(b_zero_point->Shape(), b ? b->Shape() : b_shape_),
                "MatmulInteger : B zero point is not valid");
    is_b_zp_per_column = !IsScalarOr1ElementVector(b_zero_point);
    b_offset_ptr = static_cast<const uint8_t*>(b_zero_point->DataRaw());
  }

  MatMulComputeHelper helper;

  const uint8_t* b_data;
  if (nullptr != b) {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), nullptr, b_zero_point ? &b_zero_point->Shape() : nullptr));
    b_data = static_cast<const uint8_t*>(b->DataRaw());
    b_is_signed = b->IsDataType<int8_t>();
  } else {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_, nullptr, b_zero_point ? &b_zero_point->Shape() : nullptr));
    b_data = static_cast<const uint8_t*>(packed_b_.get());
    b_is_signed = b_is_signed_;
  }


  size_t M = static_cast<size_t>(helper.M());
  size_t K = static_cast<size_t>(helper.K());
  size_t N = static_cast<size_t>(helper.N());

  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
  if (y->Shape().Size() == 0) {
    return Status::OK();
  }

  const uint8_t* a_data = static_cast<const uint8_t*>(a->DataRaw());
  auto* y_data = y->MutableData<int32_t>();

  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = M;
  gemm_shape.N = N;
  gemm_shape.K = K;
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
  #if 0
  std::cout << "Matrix A (sample):\n";
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      std::cout << static_cast<unsigned int>(a_data[i * helper.K() + j]) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  std::cout << "Matrix B (sample):\n";
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      std::cout << static_cast<unsigned int>(b_data[i * helper.N() + j]) << " ";
    }
    std::cout << "\n";
  }

  std::cout << "b_zero_point content: \n";
  if (b_zero_point != nullptr) {
    size_t b_zero_point_size = static_cast<size_t>(b_zero_point->Shape()[0]);
    const uint8_t* b_zp_data = static_cast<const uint8_t*>(b_zero_point->DataRaw());
    for (size_t i = 0; i < b_zero_point_size; ++i) {
      std::cout << static_cast<unsigned int>(b_zp_data[i]) << " ";
    }
    std::cout << "\n";
  } else {
    std::cout << "b_zero_point is null\n";
  }
  #endif
  //auto start_matmul = Clock::now();
  /*
  int8Multiply(
             reinterpret_cast<const uint8_t*>(a->DataRaw()),
             a_offset,
             reinterpret_cast<const int8_t*>(b->DataRaw()),
             reinterpret_cast<const uint8_t*>(b_zero_point->DataRaw()),
             M, 
             K, 
             N,
             reinterpret_cast<float*>(y_data)
             );
  */
  //auto end_matmul = Clock::now();
  //auto matmul_time = std::chrono::duration_cast<Microseconds>(end_matmul - start_matmul).count();

  // rowsA = M
  // width = K
  // colsB = N
  #if 0
  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    const uint8_t* aRow = inputMatrixAPtr + rowIndex * width;  // Start of row in A
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
      int32_t tempResult = 0;

      for (size_t k = 0; k < width; ++k) {
        // Row-major access
        uint8_t aValue = aRow[k];

        // Column-major access for B
        int8_t bValue = inputMatrixBPtr[k * colsB + colIndex];

        // Adjust for zero-point offsets
        int32_t adjustedA = static_cast<int32_t>(aValue) - static_cast<int32_t>(a_offset);
        int32_t adjustedB = static_cast<int32_t>(bValue); // - static_cast<int32_t>(b_offset_ptr[colIndex]);

        // Accumulate product
        tempResult += adjustedA * adjustedB;
      }

      // Write result to the output array
      outputPtr[rowIndex * colsB + colIndex] =  tempResult;
    }
  }

  // Mlas (will fallback if we don't meet requirements) 
  auto start_mblas = Clock::now();
  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());
  auto end_mblas = Clock::now();
  auto mblas_time = std::chrono::duration_cast<Microseconds>(end_mblas - start_mblas).count();
  // Output timing results
  std::cout << "Timing (microseconds):\n";
  std::cout << "MatMulFull: " << matmul_time << "\n";
  std::cout << "MlasGemmBatch: " << mblas_time << "\n";

#endif
  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
