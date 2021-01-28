// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(elementwise_add_grad);

namespace paddle {
namespace operators {

static void Memcpy(void *dst, const void *src, size_t n, bool copy_to_gpu) {
  if (copy_to_gpu) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
#elif defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_CUDA_SUCCESS(
        hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
#else
    PADDLE_THROW(
        platform::errors::InvalidArgument("Check your paddle version, current "
                                          "version is not compiled with cuda"));
#endif
  } else {
    std::memcpy(dst, src, n);
  }
}

template <typename T>
void TestMain(const platform::Place &place, 
              const framework::DDim &x_dims,
              const framework::DDim &y_dims) {
  framework::OpDesc desc;
  framework::Scope scope;

  desc.SetType("elementwise_add_grad");
  desc.SetInput("X", {"X"});
  desc.SetInput("Y", {"Y"});
  desc.SetInput(framework::GradVarName("Out"), {framework::GradVarName("Out")});
  desc.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc.SetOutput(framework::GradVarName("Y"), {framework::GradVarName("Y")});
  desc.SetAttr("axis", -1);
  desc.SetAttr("use_mkldnn", false);
  desc.SetAttr("x_data_format", "");
  desc.SetAttr("y_data_format", "");

  size_t x_numel = static_cast<size_t>(framework::product(x_dims));
  size_t y_numel = static_cast<size_t>(framework::product(y_dims));

  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  x_tensor->Resize(x_dims);
  x_tensor->mutable_data<T>(place);

  auto y_tensor = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  y_tensor->Resize(y_dims);
  y_tensor->mutable_data<T>(place);

  auto out_grad_tensor = scope.Var(framework::GradVarName("Out"))->GetMutable<framework::LoDTensor>();
  out_grad_tensor->Resize(x_dims);
  out_grad_tensor->mutable_data<T>(place);

  // feed data
  auto x_ptr = x_tensor->mutable_data<T>(place);
  auto y_ptr = y_tensor->mutable_data<T>(place);
  auto out_ptr = out_grad_tensor->mutable_data<T>(place);
  std::vector<T> x_data(x_numel), y_data(y_numel), out_data(x_numel);
  for (size_t i = 0; i < x_numel; ++i) {
    x_data[i] = 1.0;
    out_data[i] = 1.0;
  }
  for (size_t i = 0; i < y_numel; ++i) {
    y_data[i] = 1.0;
  }
  bool is_gpu_place = platform::is_gpu_place(place);
  Memcpy(x_ptr, x_data.data(), sizeof(T) * x_numel, is_gpu_place);
  Memcpy(y_ptr, y_data.data(), sizeof(T) * y_numel, is_gpu_place);
  Memcpy(out_ptr, out_data.data(), sizeof(T) * x_numel, is_gpu_place);

  scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();
  scope.Var(framework::GradVarName("Y"))->GetMutable<framework::LoDTensor>();

  auto op = framework::OpRegistry::CreateOp(desc);

  auto before_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << before_run_str;

  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  auto after_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << after_run_str;

  // get output
  framework::LoDTensor x_out;
  framework::LoDTensor y_out;
  auto &x_out_tensor = scope.FindVar(framework::GradVarName("X"))->Get<framework::LoDTensor>();
  auto &y_out_tensor = scope.FindVar(framework::GradVarName("Y"))->Get<framework::LoDTensor>();

  if (is_gpu_place) {
    framework::TensorCopySync(x_out_tensor, platform::CPUPlace(), &x_out);
    framework::TensorCopySync(y_out_tensor, platform::CPUPlace(), &y_out);
  } else {
    x_out = x_out_tensor;
    y_out = y_out_tensor;
  }

  auto *x_out_ptr = x_out.data<T>();
  auto *y_out_ptr = y_out.data<T>();
  for (size_t i = 0; i < x_numel; ++i) {
    std::cout << "x_grad[" << i << "]=" << x_out_ptr[i] << std::endl;
  }
  for (size_t i = 0; i < y_numel; ++i) {
    std::cout << "y_grad[" << i << "]=" << y_out_ptr[i] << std::endl;
  }
}

TEST(test_elementwise_add_grad_normal, cpu_place) {
  framework::DDim x_dims({1, 2, 2});
  framework::DDim y_dims({1, 1, 2});
  platform::CPUPlace p;
  TestMain<float>(p, x_dims, y_dims);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_elementwise_add_grad_normal, gpu_place) {
  framework::DDim x_dims({1, 2, 2});
  framework::DDim y_dims({1, 1, 2});
  platform::CUDAPlace p(0);
  TestMain<float>(p, x_dims, y_dims);
}
#endif

}  // namespace operators
}  // namespace paddle
