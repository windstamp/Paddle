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
USE_OP(conv2d_grad);
USE_OP_DEVICE_KERNEL(conv2d_grad, CUDNN);

namespace paddle {
namespace operators {

// input
const int batch_size = 1;
const int input_channel = 3;
const int input_height = 1;
const int input_width = 1;
// filter
const int output_channel = 3;
const int groups = 3;
const int kernel_h = 1;
const int kernel_w = 1;
// attr
const std::vector<int> conv_stride = {1, 1};
const std::vector<int> conv_padding = {1, 1, 2, 0};
const std::vector<int> conv_dilation = {1, 1};
const std::string padding_algorithm = "EXPLICIT";
const std::string data_format = "NCHW";
const bool exhaustive_search = false;

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
                             framework::LoDTensor* tensor,
                             const char * name) {
  size_t numel = static_cast<size_t>(framework::product(tensor->dims()));
  std::vector<T> data(numel);
  framework::TensorToVector(*tensor, ctx, &data);

  printf("=============%s============\n", name);
  size_t stride_h = tensor->dims()[3];
  size_t stride_w = tensor->dims()[2] * stride_h;
  size_t index = 0;
  while(index < numel) {
    printf("%5.1f ", data[index]);
    if((index+1) % stride_h == 0) printf("\n");
    if((index+1) % stride_w == 0) printf("\n");
    index ++;
  }
}

template <typename T>
void TestConv2D(const platform::DeviceContext& ctx, const bool use_cudnn = false) {
  auto place = ctx.GetPlace();
  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim input_dims({batch_size, input_channel, input_height, input_width});
  framework::DDim filter_dims({output_channel, static_cast<int>(input_channel/groups), kernel_h, kernel_w});

  // --------------- forward ----------------------
  desc_fwd.SetType("conv2d");
  desc_fwd.SetInput("Input", {"Input"});
  desc_fwd.SetInput("Filter", {"Filter"});
  desc_fwd.SetOutput("Output", {"Output"});
  desc_fwd.SetAttr("groups", groups);
  desc_fwd.SetAttr("strides", conv_stride);
  desc_fwd.SetAttr("paddings", conv_padding);
  desc_fwd.SetAttr("dilations", conv_dilation);
  desc_fwd.SetAttr("fuse_relu_before_depthwise_conv", false);
  desc_fwd.SetAttr("padding_algorithm", padding_algorithm);
  desc_fwd.SetAttr("use_cudnn", use_cudnn);
  desc_fwd.SetAttr("use_mkldnn", false);
  desc_fwd.SetAttr("data_format", data_format);
  desc_fwd.SetAttr("exhaustive_search", false);

  auto input_tensor = scope.Var("Input")->GetMutable<framework::LoDTensor>();
  auto filter_tensor = scope.Var("Filter")->GetMutable<framework::LoDTensor>();
  auto output_tensor = scope.Var("Output")->GetMutable<framework::LoDTensor>();

  // feed input data
  feed_tensor_data<T>(ctx, input_dims, input_tensor);
  feed_tensor_data<T>(ctx, filter_dims, filter_tensor);
  
  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  // get output
  print_tensor_data<T>(ctx, output_tensor, "output");

  // --------------- backward ----------------------
  desc_bwd.SetType("conv2d_grad");
  desc_bwd.SetInput("Input", {"Input"});
  desc_bwd.SetInput("Filter", {"Filter"});
  desc_bwd.SetInput(framework::GradVarName("Output"), {framework::GradVarName("Output")});
  desc_bwd.SetOutput(framework::GradVarName("Input"), {framework::GradVarName("Input")});
  desc_bwd.SetOutput(framework::GradVarName("Filter"), {framework::GradVarName("Filter")});
  desc_bwd.SetAttr("groups", groups);
  desc_bwd.SetAttr("strides", conv_stride);
  desc_bwd.SetAttr("paddings", conv_padding);
  desc_bwd.SetAttr("dilations", conv_dilation);
  desc_bwd.SetAttr("fuse_relu_before_depthwise_conv", false);
  desc_bwd.SetAttr("use_cudnn", use_cudnn);
  desc_bwd.SetAttr("use_mkldnn", false);
  desc_bwd.SetAttr("padding_algorithm", padding_algorithm);
  desc_bwd.SetAttr("data_format", data_format);
  desc_bwd.SetAttr("exhaustive_search", exhaustive_search);
  desc_bwd.SetAttr("use_addto", false);

  auto output_grad_tensor = scope.Var(framework::GradVarName("Output"))->GetMutable<framework::LoDTensor>();
  auto input_grad_tensor = scope.Var(framework::GradVarName("Input"))->GetMutable<framework::LoDTensor>();
  auto filter_grad_tensor = scope.Var(framework::GradVarName("Filter"))->GetMutable<framework::LoDTensor>();

  // feed output_grad data
  feed_tensor_data<T>(ctx, output_tensor->dims(), output_grad_tensor);

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  // get input grad data
  print_tensor_data<T>(ctx, input_grad_tensor, "input_grad");
  print_tensor_data<T>(ctx, filter_grad_tensor, "filter_grad");
}

TEST(test_conv2d_op, cpu_place) {
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  TestConv2D<float>(ctx, false);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_conv2d_op, gpu_place) {
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);
  TestConv2D<float>(ctx, false);
}

TEST(test_conv2d_cudnn_op, gpu_place) {
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);
  TestConv2D<float>(ctx, true);
}
#endif

}  // namespace operators
}  // namespace paddle
