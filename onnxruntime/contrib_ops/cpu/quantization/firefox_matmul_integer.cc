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


std::vector<uint32_t> MatMulFull(const uint8_t* a_data, const int8_t* b_data,
                                 size_t M, size_t K, size_t N,
                                 int8_t a_offset, const uint8_t* b_offset_ptr) {
    std::vector<uint32_t> output(M * N, 0);

    for (size_t row_idx = 0; row_idx < M; ++row_idx) {
        const uint8_t* a_row = a_data + row_idx * K;  // Start of row in A
        for (size_t col_idx = 0; col_idx < N; ++col_idx) {
            int64_t temp_result = 0;  // Use int64_t for intermediate accumulation

            for (size_t k = 0; k < K; ++k) {
                // Row-major access
                uint8_t a_value = a_row[k];
                int8_t b_value = b_data[k * N + col_idx];

                // Adjust for zero-point offsets
                int32_t adjusted_a = static_cast<int32_t>(a_value) - static_cast<int32_t>(a_offset);
                int32_t adjusted_b = static_cast<int32_t>(b_value) - static_cast<int32_t>(b_offset_ptr[col_idx]);

                // Accumulate product
                temp_result += static_cast<int64_t>(adjusted_a) * static_cast<int64_t>(adjusted_b);
            }


            int64_t index = row_idx * N + col_idx;
            if (index < 10) {
              std::cout << " Result for index " << index <<" " << temp_result << "\n";
            }
            // Convert to uint32_t, allowing wraparound for negative values
            output[row_idx * N + col_idx] = static_cast<uint32_t>(temp_result);
        }
    }

    return output;
}

void DisplayMatrixSample(const uint32_t* matrix, size_t rows, size_t cols, const std::string& name) {
    std::cout << "Sample of " << name << ":\n";
    size_t sample_rows = std::min(rows, static_cast<size_t>(5));
    size_t sample_cols = std::min(cols, static_cast<size_t>(5));

    for (size_t i = 0; i < sample_rows; ++i) {
        for (size_t j = 0; j < sample_cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

void CompareMatrices(const uint32_t* matrix1, const uint32_t* matrix2, size_t rows, size_t cols, const std::string& matrix1_name, const std::string& matrix2_name) {
    std::cout << "Comparing " << matrix1_name << " and " << matrix2_name << "\n";

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          if (matrix1[i * cols + j] != matrix2[i * cols + j]) {
            std::cout << "Mismatch between " << matrix1_name << " and " << matrix2_name << " at row " << i << ", col " << j << "\n";
            throw std::runtime_error(
                "Mismatch between " + matrix1_name + " and " + 
                matrix2_name + " at row " + std::to_string(i) + ", col " + std::to_string(j));
          }
        }
    }
    std::cout << "Matrices match\n";
}

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

  std::vector<uint32_t> gemmology_output(helper.M() * helper.N(), 0);

  // Manual MatMul
  auto start_matmul = Clock::now();
  std::vector<uint32_t> matmul_output = MatMulFull(a_data, reinterpret_cast<const int8_t*>(b_data), M, K, N, a_offset, b_offset_ptr);
  auto end_matmul = Clock::now();
  auto matmul_time = std::chrono::duration_cast<Microseconds>(end_matmul - start_matmul).count();

  // Gemmology
  auto start_gemmology = Clock::now();
  int8Multiply(
             reinterpret_cast<const uint8_t*>(a_data),
             a_offset,
             reinterpret_cast<const int8_t*>(b_data),
             b_offset_ptr[0],
             M, 
             N, 
             K, 
             reinterpret_cast<float*>(gemmology_output.data()));

  auto end_gemmology = Clock::now();
  auto gemmology_time = std::chrono::duration_cast<Microseconds>(end_gemmology - start_gemmology).count();

  // Mlas 
  auto start_mblas = Clock::now();
  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());
  auto end_mblas = Clock::now();
  auto mblas_time = std::chrono::duration_cast<Microseconds>(end_mblas - start_mblas).count();

  // Display samples
  DisplayMatrixSample(matmul_output.data(), M, N, "MatMulFull Output");
  DisplayMatrixSample(gemmology_output.data(), M, N, "gemmology Output");
  DisplayMatrixSample(reinterpret_cast<const uint32_t*>(y_data), M, N, "MLas Output");

  // make sure the three implementations return the same data
  CompareMatrices(matmul_output.data(), reinterpret_cast<const uint32_t*>(y_data), M, N, "MatMulFull", "MLas");
  CompareMatrices(matmul_output.data(), gemmology_output.data(), M, N, "MatMulFull", "gemmology");

  // Output timing results
  std::cout << "Timing (microseconds):\n";
  std::cout << "MatMulFull: " << matmul_time << "\n";
  std::cout << "Mlas: " << mblas_time << "\n";
  std::cout << "Gemmology: " << gemmology_time << "\n";

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
