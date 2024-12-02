// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "firefox_matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

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


#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>

extern "C" void __attribute__((import_name("int8MultiplyAndAddBias")))
  int8MultiplyAndAddBias(
      size_t ARowSize, 
      size_t BColSize, 
      size_t AColSizeBRowSize, 
      bool AIsSigned, 
      bool BIsSigned, 
      MLAS_GEMM_QUANT_DATA_PARAMS* gemm_data
  );

#endif

Status FirefoxMatMulInteger8::Compute(OpKernelContext* ctx) const {



  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  std::cout << "Input Tensor A shape: " << (a ? a->Shape().ToString() : "null") << std::endl;
  if (b) {
    std::cout << "Input Tensor B shape: " << b->Shape().ToString() << std::endl;
  } else {
    std::cout << "Using packed B." << std::endl;
  }

  // Validate zero points
  uint8_t a_offset = 0;
  const auto* a_zero_point = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  if (a_zero_point != nullptr) {
    std::cout << "A Zero Point shape: " << a_zero_point->Shape().ToString() << std::endl;
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *(static_cast<const uint8_t*>(a_zero_point->DataRaw()));
    std::cout << "A Zero Point value: " << static_cast<int>(a_offset) << std::endl;
  }

  bool is_b_zp_per_column = false;
  uint8_t b_default_offset = 0;
  const uint8_t* b_offset_ptr = &b_default_offset;
  const auto* b_zero_point = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  if (b_zero_point != nullptr) {
    std::cout << "B Zero Point shape: " << b_zero_point->Shape().ToString() << std::endl;
    ORT_ENFORCE(IsBQuantParamSupported(b_zero_point->Shape(), b ? b->Shape() : b_shape_),
                "MatmulInteger : B zero point is not valid");
    is_b_zp_per_column = !IsScalarOr1ElementVector(b_zero_point);
    b_offset_ptr = static_cast<const uint8_t*>(b_zero_point->DataRaw());
    std::cout << "B Zero Point is per-column: " << is_b_zp_per_column << std::endl;
  }

  MatMulComputeHelper helper;
  const uint8_t* b_data;
  bool b_is_signed;
  if (nullptr != b) {
    std::cout << "Computing helper with A and B shapes." << std::endl;
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), nullptr, b_zero_point ? &b_zero_point->Shape() : nullptr));
    b_data = static_cast<const uint8_t*>(b->DataRaw());
    b_is_signed = b->IsDataType<int8_t>();
  } else {
    std::cout << "Computing helper with A shape and packed B." << std::endl;
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_, nullptr, b_zero_point ? &b_zero_point->Shape() : nullptr));
    b_data = static_cast<const uint8_t*>(packed_b_.get());
    b_is_signed = b_is_signed_;
  }

  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
  std::cout << "Output Tensor Y shape: " << y->Shape().ToString() << std::endl;

  if (y->Shape().Size() == 0) {
    std::cout << "Output Tensor is empty. Exiting early." << std::endl;
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

  std::cout << "GEMM Shape - M: " << gemm_shape.M << ", N: " << gemm_shape.N
            << ", K: " << gemm_shape.K << ", AIsSigned: " << gemm_shape.AIsSigned
            << ", BIsSigned: " << gemm_shape.BIsSigned << std::endl;

  const size_t batch_size = helper.OutputOffsets().size();
  std::cout << "Batch size: " << batch_size << std::endl;

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

    std::cout << "Batch " << batch << " - A offset: " << helper.LeftOffsets()[batch]
              << ", B offset: " << helper.RightOffsets()[batch]
              << ", C offset: " << helper.OutputOffsets()[batch] << std::endl;
  }

  std::cout << "Calling MlasGemmBatch." << std::endl;

#ifdef __EMSCRIPTEN__
  onnxruntime::contrib::int8MultiplyAndAddBias(
      gemm_shape.M,
      gemm_shape.N, 
      gemm_shape.K, 
      gemm_shape.AIsSigned, 
      gemm_shape.BIsSigned, 
      gemm_data_vec.data()
  );
#endif

  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());

  std::cout << "Exiting FirefoxMatMulInteger8::Compute" << std::endl;
  return Status::OK();
}


}  // namespace contrib
}  // namespace onnxruntime
