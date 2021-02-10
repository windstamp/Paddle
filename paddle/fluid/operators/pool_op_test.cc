// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

USE_OP(pool2d);
USE_OP_DEVICE_KERNEL(pool2d, CUDNN);
USE_OP(pool2d_grad);
USE_OP_DEVICE_KERNEL(pool2d_grad, CUDNN);

namespace paddle {
namespace operators {

// input
const int batch_size = 1;
const int input_channel = 1;
const int input_height = 2;
const int input_width = 2;
// attr
const int pool_size = 1;
const int pool_stride = 1;
const int pool_padding = 0;
const std::string padding_algorithm = "EXPLICIT";
// attr
const bool ceil_mode = false;
const bool exclusive = true;
const bool adaptive = false;
const bool global_pooling = false;
const std::string pooling_t = "avg";
const std::string data_format = "NCHW";

template <typename T>
static void feed_tensor_data(const platform::DeviceContext& ctx, 
                             const framework::DDim dims,
                             framework::LoDTensor* tensor) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = 1;
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

template <typename T>
static void print_tensor_data(const platform::DeviceContext& ctx,
                             framework::LoDTensor* tensor) {
  size_t numel = static_cast<size_t>(framework::product(tensor->dims()));
  std::vector<T> data(numel);
  framework::TensorToVector(*tensor, ctx, &data);
  for (size_t i = 0; i < numel; ++i) {
    printf("output[%02d] = %5.1f\n", i, data[i]);
  }
}

template <typename T>
void TestPool2D(const platform::DeviceContext& ctx, const bool use_cudnn) {
  auto place = ctx.GetPlace();
  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  // input dims
  framework::DDim input_dims({batch_size, input_channel, input_height, input_width});
  LOG(INFO) << "input_dims:" << input_dims.to_str();

  // --------------- forward ----------------------
  desc_fwd.SetType("pool2d");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetOutput("Out", {"Out"});
  desc_fwd.SetAttr("pooling_type", pooling_t);
  desc_fwd.SetAttr("ksize", std::vector<int>({pool_size, pool_size}));
  desc_fwd.SetAttr("strides", std::vector<int>({pool_stride, pool_stride}));
  desc_fwd.SetAttr("paddings", std::vector<int>({pool_padding, pool_padding}));
  desc_fwd.SetAttr("global_pooling", true);
  desc_fwd.SetAttr("exclusive", exclusive);
  desc_fwd.SetAttr("adaptive", adaptive);
  desc_fwd.SetAttr("use_cudnn", use_cudnn);

  auto input_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto output_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();

  // feed input data
  feed_tensor_data<T>(ctx, input_dims, input_tensor);
  
  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << "Before run: " << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << "After run: " << op_fwd->DebugStringEx(&scope);

  // get output
  print_tensor_data<T>(ctx, output_tensor);

  // --------------- backward ----------------------
  desc_bwd.SetType("pool2d_grad");
  desc_bwd.SetInput("X", {"X"});
  desc_bwd.SetInput("Out", {"Out"});
  desc_bwd.SetInput(framework::GradVarName("Out"), {framework::GradVarName("Out")});
  desc_bwd.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc_bwd.SetAttr("pooling_type", pooling_t);
  desc_bwd.SetAttr("ksize", std::vector<int>({pool_size, pool_size}));
  desc_bwd.SetAttr("strides", std::vector<int>({pool_stride, pool_stride}));
  desc_bwd.SetAttr("paddings", std::vector<int>({pool_padding, pool_padding}));
  desc_bwd.SetAttr("exclusive", exclusive);
  desc_bwd.SetAttr("adaptive", adaptive);
  desc_bwd.SetAttr("data_format", data_format);
  desc_bwd.SetAttr("global_pooling", false);
  desc_bwd.SetAttr("padding_algorithm", padding_algorithm);
  desc_bwd.SetAttr("use_cudnn", use_cudnn);
  desc_bwd.SetAttr("use_mkldnn", false);

  auto output_grad_tensor = scope.Var(framework::GradVarName("Out"))->GetMutable<framework::LoDTensor>();
  auto input_grad_tensor = scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();

  // feed output_grad data
  feed_tensor_data<T>(ctx, output_tensor->dims(), output_grad_tensor);

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << "Before run: " << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << "After run: " << op_bwd->DebugStringEx(&scope);

  // get input grad data
  print_tensor_data<T>(ctx, input_tensor);
  print_tensor_data<T>(ctx, output_tensor);
  print_tensor_data<T>(ctx, output_grad_tensor);
  print_tensor_data<T>(ctx, input_grad_tensor);
}

TEST(test_pool2d_op, cpu_place) {
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  TestPool2D<float>(ctx, false);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_pool2d_op, gpu_place) {
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);
  TestPool2D<float>(ctx, false);
}

TEST(test_pool2d_cudnn_op, gpu_place) {
  int cudnn_version = platform::CudnnVersion();
  printf("CUDNN version is: %d\n", cudnn_version);
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);
  TestPool2D<float>(ctx, true);
}
#endif

}  // namespace operators
}  // namespace paddle
