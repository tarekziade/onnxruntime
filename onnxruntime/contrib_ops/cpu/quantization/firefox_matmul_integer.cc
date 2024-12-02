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


#ifdef __EMSCRIPTEN__

 float alpha = 25;
 float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f);
 //float quant_mult = 127/alpha;

/*
 // prepate the two matrices
 // we need to import `intgemm/aligned.h`
 // XXX speeds up because the memory start is a multiple of 8
 AlignedVector<Integer> A_prepared(a_data.size());
 AlignedVector<Integer> B_prepared(b_data.size());

 int8PrepareA(a_data, A_prepared, quant_mult, A_rows, width);
 int8PrepareB(b_data, B_prepared, quant_mult, width, B_cols);
 */
 // XXX I don't think I need to add a bias here
 // Multiply 
 // take unit8 for a and int8 for b
 std::cout << "int8MultiplyAndAddBias" << std::endl;

 int8MultiplyAndAddBias(
     // XXX this will work as long as a_data is aligned
     // look at the memory address printf %p multiple of 8
     reinterpret_cast<const int8_t*>(a_data), 
     1.0,  // scale_A, 
     a_offset, 
     reinterpret_cast<const int8_t*>(b_data), 
     1.0,  //scale_B, 
     0, // b_offset, // XXX 
     0, // input_bias_prepared, 
     unquant_mult_forprep, 
     1, //A_rows, 
     static_cast<size_t>(helper.K()), // width, 
     1, // B_cols, 
     reinterpret_cast<float*>(y_data)
  );

  std::cout << "int8MultiplyAndAddBias done" << std::endl;

  return Status::OK();
 #endif

  // MLas fallback.
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

  std::cout << "Calling MlasGemmBatch." << std::endl;
  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());
  std::cout << "Exiting FirefoxMatMulInteger8::Compute" << std::endl;
  return Status::OK();
}


}  // namespace contrib
}  // namespace onnxruntime
