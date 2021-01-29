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

USE_OP(conv2d);
USE_OP_DEVICE_KERNEL(conv2d, CUDNN);

namespace paddle {
namespace operators {

template <typename T>
void TestConv2D(const platform::DeviceContext& ctx, const bool use_cudnn) {
  auto place = ctx.GetPlace();
  framework::OpDesc desc;
  framework::Scope scope;

  // input
  const int batch_size = 1;
  const int input_channel = 4;
  const int input_height = 2;
  const int input_width = 2;
  framework::DDim input_dims({batch_size, input_channel, input_height, input_width});
  size_t input_numel = static_cast<size_t>(framework::product(input_dims));
  // filter
  const int output_channel = 4;
  const int groups = 1;
  const int kernel_h = 1;
  const int kernel_w = 1;
  framework::DDim filter_dims({output_channel, static_cast<int>(input_channel/groups), kernel_h, kernel_w});
  size_t filter_numel = static_cast<size_t>(framework::product(filter_dims));
  // attr
  const int conv_stride = 1;
  const int conv_padding = 0;
  const int conv_dilation = 1;
  // output
  const int output_height = static_cast<int>((input_height + 2 * conv_padding - (conv_dilation * (kernel_h - 1) + 1)) / conv_stride + 1);
  const int output_width = static_cast<int>((input_width + 2 * conv_padding - (conv_dilation * (kernel_w - 1) + 1)) / conv_stride + 1);
  framework::DDim output_dims({batch_size, output_channel, output_height, output_width});
  size_t output_numel = static_cast<size_t>(framework::product(output_dims));

  // op desc
  desc.SetType("conv2d");
  desc.SetInput("Input", {"Input"});
  desc.SetInput("Filter", {"Filter"});
  // desc.SetInput("Bias", {"Bias"});
  desc.SetOutput("Output", {"Output"});
  desc.SetAttr("groups", groups);
  desc.SetAttr("strides", std::vector<int>({conv_stride, conv_stride}));
  desc.SetAttr("paddings", std::vector<int>({conv_padding, conv_padding}));
  desc.SetAttr("dilations", std::vector<int>({conv_dilation, conv_dilation}));
  desc.SetAttr("use_cudnn", use_cudnn);

  auto input_tensor = scope.Var("Input")->GetMutable<framework::LoDTensor>();
  // input_tensor->Resize(input_dims);
  // input_tensor->mutable_data<T>(place);

  auto filter_tensor = scope.Var("Filter")->GetMutable<framework::LoDTensor>();
  // filter_tensor->Resize(filter_dims);
  // filter_tensor->mutable_data<T>(place);

  auto output_tensor = scope.Var("Output")->GetMutable<framework::LoDTensor>();
  output_tensor->Resize(output_dims);
  output_tensor->mutable_data<T>(place);

  // feed input data
  std::vector<T> input_data(input_numel);
  for (size_t i = 0; i < input_numel; ++i) {
    input_data[i] = i;
  }
  framework::TensorFromVector(input_data, ctx, input_tensor);
  input_tensor->Resize(input_dims);


  // feed filter data
  std::vector<T> filter_data(filter_numel);
  for (size_t i = 0; i < filter_numel; ++i) {
    filter_data[i] = i;
  }
  framework::TensorFromVector(filter_data, ctx, filter_tensor);
  filter_tensor->Resize(filter_dims);
  
  auto op = framework::OpRegistry::CreateOp(desc);

  auto before_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << before_run_str;

  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  auto after_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << after_run_str;

  // get output
  std::vector<T> output_data;
  framework::TensorToVector(*output_tensor, ctx, &output_data);

  for (size_t i = 0; i < output_numel; ++i) {
    printf("output[%02d] = %5.1f\n", i, output_data[i]);
  }
}

TEST(test_conv2d_op, cpu_place) {
  // framework::DDim input_dims({1, 4, 2, 2});
  // framework::DDim filter_dims({4, 4, 2, 2});
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  TestConv2D<float>(ctx, false);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_conv2d_op, gpu_place) {
  // framework::DDim input_dims({1, 4, 2, 2});
  // framework::DDim filter_dims({4, 4, 2, 2});
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);
  TestConv2D<float>(ctx, false);
}

TEST(test_conv2d_cudnn_op, gpu_place) {
  // framework::DDim input_dims({1, 4, 2, 2});
  // framework::DDim filter_dims({4, 4, 2, 2});
  int cudnn_version = platform::CudnnVersion();
  printf("CUDNN version is: %d\n", cudnn_version);
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);
  TestConv2D<float>(ctx, true);
}
#endif

}  // namespace operators
}  // namespace paddle
