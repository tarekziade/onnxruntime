// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iostream>
#include <vector>
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

    std::vector<int32_t> int32_output(helper.M() * helper.N(), 0);

  #ifdef __EMSCRIPTEN__
    uint8_t zero_point_b = *(b_offset_ptr + helper.RightZeroPointOffsets()[0]);

    // Output all inputs before the call
    std::cout << "Matrix A:\n";
    for (size_t i = 0; i < static_cast<size_t>(helper.M()); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(helper.K()); ++j) {
        std::cout << static_cast<int>(a_data[i * helper.K() + j]) << " ";
      }
      std::cout << "\n";
    }

    std::cout << "Matrix B:\n";
    for (size_t i = 0; i < static_cast<size_t>(helper.K()); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(helper.N()); ++j) {
        std::cout << static_cast<int>(b_data[i * helper.N() + j]) << " ";
      }
      std::cout << "\n";
    }

    std::cout << "A Zero point: " << static_cast<int>(a_offset) << "\n";
    std::cout << "B zero_point: " << static_cast<int>(zero_point_b) << "\n";
    std::cout << "rows A: " << helper.M() << ", width: " << helper.K() << ", Cols B: " << helper.N() << "\n";
    std::cout << "B is packed: " << (packed_b_ ? "true" : "false") << "\n";
    std::cout << "B is signed: " << (b_is_signed ? "true" : "false") << "\n";

    // Gemmology call 
    int8Multiply(reinterpret_cast<const int8_t*>(a_data),
                 a_offset,
                 reinterpret_cast<const int8_t*>(b_data),
                 zero_point_b, 
                 static_cast<size_t>(helper.M()),  // rows A
                 static_cast<size_t>(helper.K()),  // width
                 static_cast<size_t>(helper.N()),  // col B
                 reinterpret_cast<float*>(int32_output.data()));
  #endif

  // Original MatmulInteger call
  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());

  // Compare the outputs
  std::cout << "Comparing Outputs:\n";
  std::cout << "Gemmology:\n";
  for (size_t i = 0; i < static_cast<size_t>(helper.M()); ++i) {
    for (size_t j = 0; j < static_cast<size_t>(helper.N()); ++j) {
      std::cout << static_cast<int>(int32_output[i * helper.N() + j]) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "MBLas:\n";
  for (size_t i = 0; i < static_cast<size_t>(helper.M()); ++i) {
    for (size_t j = 0; j < static_cast<size_t>(helper.N()); ++j) {
      std::cout << static_cast<int>(y_data[i * helper.N() + j]) << " ";
    }
    std::cout << "\n";
  }

for (size_t i = 0; i < static_cast<size_t>(helper.M()); ++i) {
  for (size_t j = 0; j < static_cast<size_t>(helper.N()); ++j) {
    size_t index = i * helper.N() + j;
    if (int32_output[index] != static_cast<float>(y_data[index])) {
      ORT_ENFORCE(false, "Mismatch at Row ", i, ", Col ", j, ": int8Multiply = ", int32_output[index],
                  ", MlasGemmBatch = ", static_cast<float>(y_data[index]));
    }
  }
}

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
