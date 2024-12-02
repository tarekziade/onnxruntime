// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace test {

TEST(FirefoxMatMulInteger8OpTest, FirefoxMatMulInteger8_1) {
  OpTester test("FirefoxMatMulInteger8", 1, onnxruntime::kMSDomain);
  test.AddInput<int8_t>("T1", {1, 1}, {15});
  test.AddInput<int8_t>("T2", {1, 1}, {8});
  test.AddOutput<int32_t>("T3", {1, 1}, {120});  // Result is 15 * 8
  test.Run();
}

TEST(FirefoxMatMulInteger8OpTest, FirefoxMatMulInteger8_2) {
  OpTester test("FirefoxMatMulInteger8", 1, onnxruntime::kMSDomain);
  test.AddInput<int8_t>("T1", {1, 2}, {-7, 10});
  test.AddInput<int8_t>("T2", {2, 1}, {-8, -11});
  test.AddOutput<int32_t>("T3", {1, 1}, {8});  // Result is (-7 * -8) + (10 * -11)
  test.Run();
}

TEST(FirefoxMatMulInteger8OpTest, FirefoxMatMulInteger8_Empty_input) {
  OpTester test("FirefoxMatMulInteger8", 1, onnxruntime::kMSDomain);
  test.AddInput<int8_t>("T1", {0, 2}, {});
  test.AddInput<int8_t>("T2", {2, 1}, {-8, -11});
  test.AddOutput<int32_t>("T3", {0, 1}, {});  // Empty input produces an empty output
  test.Run();
}

TEST(FirefoxMatMulInteger8OpTest, FirefoxMatMulInteger8_3) {
  OpTester test("FirefoxMatMulInteger8", 1, onnxruntime::kMSDomain);
  test.AddInput<int8_t>("T1", {3, 2}, {-7, 10, 10, -113, 22, -36});
  test.AddInput<int8_t>("T2", {2, 4}, {-8, -11, 13, 14, -9, 12, 3, -6});
  test.AddOutput<int32_t>("T3", {3, 4},
                          {-158, 97, -61, -2,          // First row results
                           989, -1426, 1693, 1682,     // Second row results
                           282, -518, 280, -372});     // Third row results
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
