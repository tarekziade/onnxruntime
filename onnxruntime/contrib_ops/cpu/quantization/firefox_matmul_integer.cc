// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iostream>
#include <vector>
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

    std::vector<uint32_t> gemmology_output(helper.M() * helper.N(), 0);

  #ifdef __EMSCRIPTEN__
    uint8_t zero_point_b = *(b_offset_ptr + helper.RightZeroPointOffsets()[0]);

    std::cout << "A Zero point: " << static_cast<int>(a_offset) << "\n";
    std::cout << "B zero_point: " << static_cast<int>(zero_point_b) << "\n";
    std::cout << "rows A: " << helper.M() << ", width: " << helper.K() << ", Cols B: " << helper.N() << "\n";
    std::cout << "B is packed: " << (packed_b_ ? "true" : "false") << "\n";
    std::cout << "B is signed: " << (b_is_signed ? "true" : "false") << "\n";


std::cout << "Zero Points Debug:\n";
std::cout << "A Zero Point: " << static_cast<int>(a_offset) << "\n";
std::cout << "B Zero Points (all columns): ";
for (size_t i = 0; i < static_cast<size_t>(helper.N()); ++i) {
  std::cout << static_cast<int>(b_offset_ptr[i]) << " ";
}
std::cout << "\n";

std::cout << "Matrix Dimensions:\n";
std::cout << "M (rows A): " << gemm_shape.M << ", K (width): " << gemm_shape.K 
          << ", N (cols B): " << gemm_shape.N << "\n";

std::cout << "Signedness:\n";
std::cout << "AIsSigned: " << (gemm_shape.AIsSigned ? "true" : "false") << "\n";
std::cout << "BIsSigned: " << (gemm_shape.BIsSigned ? "true" : "false") << "\n";


std::cout << "Matrix A (sample):\n";
for (size_t i = 0; i < 5; ++i) {
  for (size_t j = 0; j < 5; ++j) {
    std::cout << static_cast<unsigned int>(a_data[i * helper.K() + j]) << " ";
  }
  std::cout << "\n";
}

std::cout << "Matrix B (sample):\n";
for (size_t i = 0; i < 5; ++i) {
  for (size_t j = 0; j < 5; ++j) {
    std::cout << static_cast<unsigned int>(b_data[i * helper.N() + j]) << " ";
  }
  std::cout << "\n";
}
std::cout << "Offsets Debug:\n";
std::cout << "Left Offsets (A): ";
for (size_t i = 0; i < batch_size; ++i) {
  std::cout << helper.LeftOffsets()[i] << " ";
}
std::cout << "\n";

std::cout << "Right Offsets (B): ";
for (size_t i = 0; i < batch_size; ++i) {
  std::cout << helper.RightOffsets()[i] << " ";
}
std::cout << "\n";

std::cout << "B is packed: " << (packed_b_ ? "true" : "false") << "\n";


// Manually compute the first value of the first row of the output
uint32_t manual_result = 0;

std::cout << "Dimensions: M = " << helper.M() << ", K = " << helper.K() << ", N = " << helper.N() << "\n";

    std::cout << "Manually computing first value of the output matrix (Row 0, Col 0):\n";

    int64_t temp_result = 0; // Use a signed type for accumulation to handle potential negatives
    for (size_t k = 0; k < static_cast<size_t>(helper.K()); ++k) {
        uint8_t a_value = static_cast<uint8_t>(a_data[k]);  // First row of A (unsigned)
        int8_t b_value = static_cast<int8_t>(b_data[k * helper.N()]); // First column of B (signed)

        // Adjust for zero points
        int32_t adjusted_a = static_cast<int32_t>(a_value) - static_cast<int32_t>(a_offset); // A is unsigned
        int32_t adjusted_b = static_cast<int32_t>(b_value) - static_cast<int32_t>(b_offset_ptr[0]); // B is signed

        // Accumulate the signed result
        temp_result += static_cast<int64_t>(adjusted_a) * static_cast<int64_t>(adjusted_b);

        // Debugging individual terms
        std::cout << "k = " << k
                  << ", A[k] = " << static_cast<int>(a_value)
                  << ", B[k, 0] = " << static_cast<int>(b_value)
                  << ", Adjusted A[k] = " << adjusted_a
                  << ", Adjusted B[k, 0] = " << adjusted_b
                  << ", Partial Sum (signed) = " << temp_result << "\n";
    }

    // Ensure the result fits in uint32_t, saturating if necessary
    manual_result = static_cast<uint32_t>(std::max<int64_t>(0, temp_result)); // Clamp to 0 for unsigned range

    std::cout << "Manual computation result (Row 0, Col 0): " << manual_result << "\n";


    // Gemmology call
    std::cout << "Calling gemmology from onnx:\n";
    auto start_gemmology = Clock::now();

    int8Multiply(
                reinterpret_cast<const uint8_t*>(a_data),
               0, // a_offset,
               reinterpret_cast<const int8_t*>(b_data),
               b_offset_ptr[0],
               static_cast<size_t>(helper.M()),  // rows A
               static_cast<size_t>(helper.K()),  // width
               static_cast<size_t>(helper.N()),  // col B
               reinterpret_cast<float*>(gemmology_output.data()));

    auto end_gemmology = Clock::now();
    auto gemmology_time = std::chrono::duration_cast<Microseconds>(end_gemmology - start_gemmology).count();
    std::cout << "gemmology call complete.\n";

    std::cout << "Call done\n";

    std::cout << "Manually Clamping\n";

    for (size_t i = 0; i < static_cast<size_t>(helper.M()); ++i) {
    for (size_t j = 0; j < static_cast<size_t>(helper.N()); ++j) {
        size_t index = i * static_cast<size_t>(helper.N()) + j;

        // Interpret unsigned value as signed
        uint32_t raw_value = gemmology_output[index];
        //std::cout << "Index (" << i << ", " << j << "), Original Value (unsigned): " << raw_value << "\n";

        int32_t signed_value = static_cast<int32_t>(raw_value);
        //std::cout << "Index (" << i << ", " << j << "), Interpreted as Signed: " << signed_value << "\n";


        // Clamp to non-negative
        uint32_t clamped_value = static_cast<uint32_t>(std::max<int32_t>(0, signed_value));

        // Write clamped value back to output
        gemmology_output[index] = clamped_value;

        // Log for debugging
        if (i == 0 && j == 0) { // Only log the first value
            std::cout << "Post-process Clamping for Index (0, 0):\n";
            std::cout << "Raw Value (unsigned): " << raw_value << "\n";
            std::cout << "Interpreted as Signed: " << signed_value << "\n";
            std::cout << "Clamped Value: " << clamped_value << "\n";
        }
    }


}


  #endif
std::cout << "Calling MlasGemmBatch\n";

auto start_mblas = Clock::now();

  // Original MatmulInteger call
MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());
auto end_mblas = Clock::now();
auto mblas_time = std::chrono::duration_cast<Microseconds>(end_mblas - start_mblas).count();


std::cout << "Calling MlasGemmBatch done\n";
// Compute percentage difference
double percentage_diff = (static_cast<double>(gemmology_time - mblas_time) / mblas_time) * 100.0;

// Display the results
std::cout << "Execution Times (Microseconds): MBlas = " << mblas_time
          << ", Gemmology = " << gemmology_time 
          << ", Difference = " << percentage_diff << "%\n";




  // Compare the outputs
  std::cout << "Comparing Outputs:\n";
  //for (size_t i = 0; i < static_cast<size_t>(helper.M()); ++i) {
  for (size_t i = 0; i < 2; ++i) {
    //for (size_t j = 0; j < static_cast<size_t>(helper.N()); ++j) {
    for (size_t j = 0; j < 2; ++j) {
      std::cout << "Gemmology:";
      std::cout << static_cast<uint32_t>(gemmology_output[i * helper.N() + j]) << "\n";
      std::cout << "MBLas:";
      std::cout << static_cast<uint32_t>(y_data[i * helper.N() + j]) << "\n";
    }
    std::cout << "\n";
  }
std::cout << "Comparing\n";


for (size_t i = 0; i < static_cast<size_t>(helper.M()); ++i) {
  for (size_t j = 0; j < static_cast<size_t>(helper.N()); ++j) {
   std::cout << "Mismatch lookup\n";

    
    size_t index = i * helper.N() + j;
    std::cout << "Lookup at Row " << i << ", Col " << j << ": " << index << "\n";

    if (gemmology_output[index] != static_cast<float>(y_data[index])) {
      std::cout << "Mismatch";

      ORT_ENFORCE(false, "Mismatch at Row ", i, ", Col ", j, ": int8Multiply = ", gemmology_output[index],
                  ", MlasGemmBatch = ", static_cast<float>(y_data[index]));
    }
  }
}


  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
