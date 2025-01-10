// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/quantization/matmul_integer_base.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include <cassert>

#include <chrono>
#include <algorithm>

namespace onnxruntime {
namespace contrib {

namespace {


using Index = uint32_t;

extern "C" void
    __attribute__((import_module("wasm_gemm"), import_name("onnx_matmul_integer_to_float")))
    GeckoMatmulIntegerToFloat(
      const uint8_t* a_data,
      float zero_point_A,
      const int8_t* input_B,
      const uint8_t* zero_point_B,
      uint32_t rows_A,
      uint32_t width,
      uint32_t cols_B,
      const float* b_scale_data,
      float is_b_scale_per_column,
      float* output
);


void ScaleOutput(const Tensor& scale, Tensor& output) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.ScalarInput0<float>() * per_iter_bh.EigenInput1<float>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>().array() * per_iter_bh.ScalarInput1<float>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>().cwiseProduct(per_iter_bh.EigenInput1<float>());
      }};

  InputBroadcaster input_broadcaster(scale, output);
  OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(),
                                       output);
  BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster);

  BroadcastLooper(broadcast_helper, funcs);
}
}  // namespace

class MatMulIntegerToFloatBase : public MatMulIntegerBase {
 public:
  MatMulIntegerToFloatBase(const OpKernelInfo& info) : MatMulIntegerBase(info) {
  }

  enum OutputTensors : int { OUT_Y = 0 };

 protected:
  Status ComputeCommon(OpKernelContext* ctx,
                       const uint8_t* a_data,
                       const TensorShape& a_shape,
                       float a_scale,
                       uint8_t a_zp,
                       bool a_is_signed,
const Tensor* b_tensor,
                       const Tensor* b_scale,
                       const Tensor* b_zp,
                       const Tensor* bias_tensor) const;
};

void MatMulFull(const uint8_t* inputMatrixA,
                const int8_t* inputMatrixB,
                float* output,
                size_t rowsA,
                size_t width,
                size_t colsB,
                uint8_t zeroPointA,
                const uint8_t* zeroPointB,
                const float* b_scale_data,
                bool is_b_scale_per_column) {

  float matrixScale = is_b_scale_per_column ? 0.0f : b_scale_data[0];
  int32_t matrixZeroPointB = is_b_scale_per_column ? 0 : static_cast<int32_t>(zeroPointB[0]);

  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    const uint8_t* aRow = inputMatrixA + rowIndex * width;  // Start of row in A
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
      int32_t tempResult = 0;

      for (size_t k = 0; k < width; ++k) {
        // Row-major access
        uint8_t aValue = aRow[k];

        // Column-major access for B
        int8_t bValue = inputMatrixB[k * colsB + colIndex];

        // Adjust for zero-point offsets
        int32_t adjustedA = static_cast<int32_t>(aValue) - static_cast<int32_t>(zeroPointA);
        int32_t adjustedB = static_cast<int32_t>(bValue);

        if (is_b_scale_per_column) {
          adjustedB -= static_cast<int32_t>(zeroPointB[colIndex]);
        } else {
          adjustedB -= matrixZeroPointB;
        }
        // Accumulate product
        tempResult += adjustedA * adjustedB;
      }

      float scaledResult = tempResult;
      if (is_b_scale_per_column) {
        scaledResult *= b_scale_data[colIndex]; 
      }
      else {
        scaledResult *= matrixScale;
      }

      // Store the scaled result in y_data
      output[rowIndex * colsB + colIndex] = scaledResult;
    }
  }
}

Status MatMulIntegerToFloatBase::ComputeCommon(OpKernelContext* ctx,
                                               const uint8_t* a_data,
                                               const TensorShape& a_shape,
                                               float a_scale,
                                               uint8_t a_zp,
                                               bool a_is_signed,
                                               const Tensor* b_tensor,
                                               const Tensor* b_scale_tensor,
                                               const Tensor* b_zp_tensor,
                                               const Tensor* bias_tensor) const {
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape,
                                     b_tensor ? b_tensor->Shape() : b_shape_,
                                     b_scale_tensor ? &b_scale_tensor->Shape() : nullptr,
                                     b_zp_tensor ? &b_zp_tensor->Shape() : nullptr));
  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();
  const auto* bias_data = bias_tensor != nullptr ? bias_tensor->Data<float>() : nullptr;

  // process zero point of b
  bool is_b_zp_per_column = false;
  uint8_t b_zp_default = 0;
  const uint8_t* b_zp_ptr = &b_zp_default;
  if (nullptr != b_zp_tensor) {
    ORT_ENFORCE(IsBQuantParamSupported(b_zp_tensor->Shape(), b_tensor ? b_tensor->Shape() : b_shape_),
                "MatmulInteger : b zero point is not valid");

    is_b_zp_per_column = !IsScalarOr1ElementVector(b_zp_tensor);
    b_zp_ptr = static_cast<const uint8_t*>(b_zp_tensor->DataRaw());
  }

  // process scale of b
  bool is_b_scale_per_column = false;
  float multiplier_per_tensor = a_scale;
  const float* b_scale_data = &multiplier_per_tensor;
  std::vector<float> multipliers_per_column;
  if (nullptr != b_scale_tensor) {
    is_b_scale_per_column = !IsScalarOr1ElementVector(b_scale_tensor);
    const float* b_scale_tensor_data = b_scale_tensor->Data<float>();

    if (is_b_scale_per_column) {
      multipliers_per_column.reserve(narrow<size_t>(b_scale_tensor->Shape().Size()));
      std::transform(b_scale_tensor_data,
                     b_scale_tensor_data + b_scale_tensor->Shape().Size(),
                     std::back_inserter(multipliers_per_column),
                     [&a_scale](float b_scale) {
                       return a_scale * b_scale;
                     });
      b_scale_data = multipliers_per_column.data();
    } else {
      multiplier_per_tensor *= *b_scale_tensor_data;
    }
  }

  // batch gemm
  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.AIsSigned = a_is_signed;
  gemm_shape.BIsSigned = b_tensor ? b_tensor->IsDataType<int8_t>() : b_is_signed_;

  const size_t num_gemms = helper.OutputOffsets().size();
  std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> gemm_scale_procs;
  gemm_scale_procs.reserve(num_gemms);
  std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(num_gemms);

  for (size_t gemm_idx = 0; gemm_idx < num_gemms; gemm_idx++) {
    gemm_scale_procs.emplace_back(y_data + helper.OutputOffsets()[gemm_idx],
                                  gemm_shape.N,
                                  b_scale_data + helper.RightScaleOffsets()[gemm_idx],
                                  bias_data,
                                  MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                  is_b_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
    auto& params = gemm_data_vec[gemm_idx];
    params.OutputProcessor = &(gemm_scale_procs[gemm_idx]);
    params.A = a_data + helper.LeftOffsets()[gemm_idx];
    params.lda = gemm_shape.K;
    params.ZeroPointA = a_zp;
    params.BIsPacked = bool(packed_b_);
    params.B = b_tensor ? static_cast<const uint8_t*>(b_tensor->DataRaw()) + helper.RightOffsets()[gemm_idx] : packed_b_.get();
    params.ldb = gemm_shape.N;
    params.ZeroPointB = b_zp_ptr + helper.RightZeroPointOffsets()[gemm_idx];
    params.PerColumnZeroPoints = is_b_zp_per_column;
    params.C = reinterpret_cast<int32_t*>(y_data + helper.OutputOffsets()[gemm_idx]);
    params.ldc = gemm_shape.N;
  }

    #if 0 
  std::vector<float> y_data_2(rowsA * colsB, 0.0f);
  if (rowsA > 1) {
  std::cout << "rowsA: " << rowsA << ", width: " << width << ", colsB: " << colsB << "\n";
  std::cout << "a_zp: " << static_cast<int>(a_zp) << "\n";
  std::cout << "is_b_scale_per_column: " << is_b_scale_per_column << "\n";
  std::cout << "multiplier_per_tensor: " << multiplier_per_tensor << "\n";
  std::cout << "b_scale_data sample: [";
  for (size_t i = 0; i < 25; ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << b_scale_data[i];
  }
  std::cout << "]\n";
  std::cout << "b_zero point sample: [";
  for (size_t i = 0; i < 25; ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << static_cast<int>(b_zp_ptr[i]) << ", ";
  }
  std::cout << "]\n";

  if (bias_data != nullptr) {
    size_t bias_size = static_cast<size_t>(bias_tensor->Shape().Size()); // Get the total size of bias_data
    size_t display_limit = std::min(bias_size, static_cast<size_t>(100));
    std::cout << "First " << display_limit << " elements of bias_data: [";
    for (size_t i = 0; i < display_limit; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << bias_data[i];
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "multiplier_per_tensor: " << multiplier_per_tensor << std::endl;
  std::cout << "b_scale_data[0]: " << b_scale_data[0] << std::endl;
  }
  #endif 
  //auto start = std::chrono::steady_clock::now();
  //std::cout << "Calling f32Multiply\n";
  // should split in parts and call ctx.ParallelFor just on the rows part

  // rowsA = M
  // width = K
  // colsB = N
#if 0
  size_t rowsA = static_cast<size_t>(helper.M());

  if (rowsA > 1) {
  size_t width = static_cast<size_t>(helper.K());
  size_t colsB = static_cast<size_t>(helper.N());
  
  const int8_t* b_data = static_cast<const int8_t*>(b_tensor->DataRaw());

  GeckoMatmulIntegerToFloat(a_data, 
              a_zp,  
              b_data,
              b_zp_ptr,
              rowsA,
              width,
              colsB,
              b_scale_data,
              is_b_scale_per_column,
              y_data
  );
  }
  else {
#endif
    MlasGemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms, ctx->GetOperatorThreadPool());
  
  //}

 //
  /* 
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Done calling f32Multiply. Duration: " << duration << " nano\n";

  std::cout << "Calling MlasGemmBatch\n";
  auto start2 = std::chrono::steady_clock::now();
  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms, ctx->GetOperatorThreadPool());
  auto end2 = std::chrono::steady_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
  std::cout << "Done calling MlasGemmBatch. Duration: " << duration2 << " nano\n";
  */
  /* 

  // Compare y_data and y_data_2

  size_t total_elements = rowsA * colsB;
  size_t display_limit = std::min(total_elements, static_cast<size_t>(100));
  bool mismatch_found = false;
  for (size_t i = 0; i < total_elements; ++i) {
      if (std::fabs(y_data[i] - y_data_2[i]) > 1e-6) { // Tolerance for floating-point comparison
          std::cerr << "Mismatch at index " << i << ": y_data=" << y_data[i] << ", y_data_2=" << y_data_2[i] << std::endl;
          mismatch_found = true;
          break;
      }
  }

  if (mismatch_found) {
    std::cerr << "Displaying the first 100 elements of y_data and y_data_2:" << std::endl;
    std::cerr << "[";
    for (size_t i = 0; i < display_limit; ++i) {
        std::cerr << "(Index " << i << ": y_data=" << y_data[i] << ", y_data_2=" << y_data_2[i] << ")";
        if (i != display_limit - 1) {
            std::cerr << ", ";
        }
    }
    std::cerr << "]" << std::endl;
    std::cerr << "Mismatch found between y_data and y_data_2!" << std::endl;
    assert(false && "Validation failed: y_data and y_data_2 are not equal.");
  }
  */
  return Status::OK();
}

class DynamicQuantizeMatMul final : public MatMulIntegerToFloatBase {
 public:
  DynamicQuantizeMatMul(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_B_SCALE = 2,
    IN_B_ZERO_POINT = 3,
    IN_BIAS = 4
  };

 protected:
  int GetBIdx() const override { return IN_B; }
};

class MatMulIntegerToFloat final : public MatMulIntegerToFloatBase {
 public:
  MatMulIntegerToFloat(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_SCALE = 2,
    IN_B_SCALE = 3,
    IN_A_ZERO_POINT = 4,
    IN_B_ZERO_POINT = 5,
    IN_BIAS = 6
  };

 protected:
  int GetBIdx() const override { return IN_B; }

 private:
  // a scale and b scale may be switched in fusion stage because of lack of shape information.
  // Fix them up before computation.
  static void FixupScaleTensor(const Tensor*& a_scale_tensor, const Tensor*& b_scale_tensor);
};

Status DynamicQuantizeMatMul::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  const Tensor* b_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);

  // calculate quantization parameter of a
  const float* a_data = a->Data<float>();
  int64_t num_of_elements = a->Shape().Size();

  float a_scale;
  uint8_t a_zero_point;
  GetQuantizationParameter(a_data, num_of_elements, a_scale, a_zero_point, ctx->GetOperatorThreadPool());

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  uint8_t* a_data_quant = static_cast<uint8_t*>(allocator->Alloc(SafeInt<size_t>(num_of_elements) * sizeof(uint8_t)));
  BufferUniquePtr a_buffer_quant_holder(a_data_quant, BufferDeleter(std::move(allocator)));

  ParQuantizeLinearStd(a_data, a_data_quant, narrow<size_t>(num_of_elements), a_scale, a_zero_point, ctx->GetOperatorThreadPool());

  bool is_b_scale_supported = IsBQuantParamSupported(b_scale_tensor->Shape(), b ? b->Shape() : b_shape_);
  
  //std::cout << "dynamic quantize matmul calling ComputeCommon" << std::endl;


  ORT_RETURN_IF_ERROR(ComputeCommon(
      ctx,
      a_data_quant,
      a->Shape(),
      a_scale,
      a_zero_point,
      false /*a_is_signed*/,
      b,
      is_b_scale_supported ? b_scale_tensor : nullptr,
      b_zp_tensor,
      ctx->Input<Tensor>(IN_BIAS)));

  if (!is_b_scale_supported) {
    //std::cout << "dynamic quantize matmul: b scale is not supported\n";
    ScaleOutput(*b_scale_tensor, *ctx->Output<Tensor>(0));
  }

  return Status::OK();
}

void MatMulIntegerToFloat::FixupScaleTensor(const Tensor*& a_scale_tensor, const Tensor*& b_scale_tensor) {
  const TensorShape a_scale_shape = a_scale_tensor->Shape();
  const TensorShape b_scale_shape = b_scale_tensor->Shape();
  if (!IsScalarOr1ElementVector(a_scale_tensor)) {
    size_t a_scale_rank = a_scale_shape.NumDimensions();
    if (a_scale_rank == 1 || a_scale_shape[a_scale_rank - 1] != 1) {
      std::swap(a_scale_tensor, b_scale_tensor);
    }
  } else if (!IsScalarOr1ElementVector(b_scale_tensor)) {
    size_t b_scale_rank = b_scale_shape.NumDimensions();
    if (b_scale_rank > 1 && b_scale_shape[b_scale_rank - 2] != 1) {
      std::swap(a_scale_tensor, b_scale_tensor);
    }
  }
}

Status MatMulIntegerToFloat::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  const Tensor* a_scale_tensor = ctx->Input<Tensor>(IN_A_SCALE);
  const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  FixupScaleTensor(a_scale_tensor, b_scale_tensor);
  bool is_a_scale_scalar = IsScalarOr1ElementVector(a_scale_tensor);
  bool is_b_scale_supported = IsBQuantParamSupported(b_scale_tensor->Shape(), nullptr != b ? b->Shape() : b_shape_);

  // validate zero point of a
  uint8_t a_zero_point = 0;
  const Tensor* a_zero_point_tensor = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  if (a_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point_tensor),
                "MatMulIntegerToFloat : input a zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    a_zero_point = *(static_cast<const uint8_t*>(a_zero_point_tensor->DataRaw()));
  }

  //std::cout << "matmul integer float calling ComputeCommon" << std::endl;
  const Tensor* b_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  ORT_RETURN_IF_ERROR(ComputeCommon(
      ctx,
      static_cast<const uint8_t*>(a->DataRaw()),
      a->Shape(),
      is_a_scale_scalar ? *a_scale_tensor->Data<float>() : 1.f,
      a_zero_point,
      a->IsDataType<int8_t>(),
      b,
      is_b_scale_supported ? b_scale_tensor : nullptr,
      b_zp_tensor,
      ctx->Input<Tensor>(IN_BIAS)));

  if (!is_a_scale_scalar) {
    //std::cout << "dynamic quantize matmul: a scale is not scalar\n";
    ScaleOutput(*a_scale_tensor, *ctx->Output<Tensor>(0));
  }
  if (!is_b_scale_supported) {
    //std::cout << "dynamic quantize matmul: b scale is not supported\n";
    ScaleOutput(*b_scale_tensor, *ctx->Output<Tensor>(0));
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DynamicQuantizeMatMul,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
    DynamicQuantizeMatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

}  // namespace contrib
}  // namespace onnxruntime
