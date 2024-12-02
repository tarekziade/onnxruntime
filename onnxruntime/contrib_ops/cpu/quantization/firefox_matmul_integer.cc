// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "firefox_matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FirefoxMatMulInteger8,
    kMSDomain,
    11,
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
    11,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    FirefoxMatMulInteger8);


Status FirefoxMatMulInteger8::Compute(OpKernelContext* ctx) const {
  auto A = ctx->Input<Tensor>(0);
  auto B = ctx->Input<Tensor>(1);
  ORT_ENFORCE(A != nullptr && B != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  for (int i = 0; i < static_cast<int>(helper.OutputOffsets().size()); i++) {
    EigenCastGEMM<int16_t, int16_t, int32_t>(
        A->Data<int16_t>() + helper.LeftOffsets()[i],
        B->Data<int16_t>() + helper.RightOffsets()[i],
        Y->MutableData<int32_t>() + helper.OutputOffsets()[i],
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()));
  }

  printf("I was called\n");
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
