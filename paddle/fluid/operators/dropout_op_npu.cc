/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class DropoutNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    float dropout_prob = ctx.Attr<float>("dropout_prob");
    // LOG(WARNING) << "dropout_prob: " << dropout_prob;

    // std::ostringstream oss;
    // SerializeToStream(oss, *x);
    // LOG(WARNING) << "x: " << oss.str();

    LOG(WARNING) << "x: " << x;
    LOG(WARNING) << "out: " << out;

    int numel = x->numel();
    LOG(WARNING) << "x numel: " << numel;
    LOG(WARNING) << "out numel: " << out->numel();
    LOG(WARNING) << "x dims: " << x->dims();
    LOG(WARNING) << "out dims: " << out->dims();

    // std::ostringstream oss2;
    // for (int i = 0; i < numel; ++i) {
    //   // oss2 << x->data<T>()[i] << ",";
    //   // printf("%f, ", x->data<T>()[i]);
    //   // printf("%f, ", *(x->data<T>()));
    // }

    out->mutable_data<T>(ctx.GetPlace());

    LOG(WARNING) << "x: " << x;
    LOG(WARNING) << "x->data: " << x->data<T>();
    // LOG(WARNING) << "x->data: " << oss2.str();
    LOG(WARNING) << "out: " << out;

    framework::NPUAttributeMap attr_input = {{"dropout_ratio", dropout_prob}};

    const auto& runner = NpuOpRunner("Dropout",
                                     {
                                         *x,
                                     },
                                     {*out}, attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);

    LOG(WARNING) << "out: " << out;
  }
};

template <typename DeviceContext, typename T>
class DropoutV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* seed = ctx.Input<LoDTensor>("Seed");
    auto* out = ctx.Output<LoDTensor>("Out");
    auto* mask = ctx.Output<LoDTensor>("Mask");
    float dropout_prob = ctx.Attr<float>("dropout_prob");
    LOG(WARNING) << "dropout_prob: " << dropout_prob;

    LOG(WARNING) << "x: " << x;
    LOG(WARNING) << "seed: " << seed;
    LOG(WARNING) << "out: " << out;
    LOG(WARNING) << "mask: " << mask;

    LOG(WARNING) << "x numel: " << x->numel();
    LOG(WARNING) << "seed numel: " << seed->numel();
    LOG(WARNING) << "out numel: " << out->numel();
    LOG(WARNING) << "mask numel: " << mask->numel();

    LOG(WARNING) << "x dims: " << x->dims();
    LOG(WARNING) << "seed dims: " << seed->dims();
    LOG(WARNING) << "out dims: " << out->dims();
    LOG(WARNING) << "mask dims: " << mask->dims();

    out->mutable_data<T>(ctx.GetPlace());
    mask->mutable_data<T>(ctx.GetPlace());

    LOG(WARNING) << "out: " << out;
    LOG(WARNING) << "mask: " << mask;

    // const auto& runner = NpuOpRunner("DropoutV2",
    //                                  {
    //                                      *x, *seed,
    //                                  },
    //                                  {*out, *mask}, {{"p", dropout_prob}});

    framework::Tensor new_seed;
    new_seed.Resize(seed->dims());
    new_seed.mutable_data<T>(ctx.GetPlace());

    LOG(WARNING) << "seed: " << &seed;
    LOG(WARNING) << "seed numel: " << seed->numel();
    LOG(WARNING) << "seed dims: " << seed->dims();

    framework::NPUAttributeMap attr_input = {{"p", dropout_prob}};

    const auto& runner = NpuOpRunner("DropoutV2",
                                     {
                                         *x, *seed,
                                     },
                                     {*out, *mask, new_seed}, attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);

    LOG(WARNING) << "out: " << out;
  }
};

template <typename DeviceContext, typename T>
class DropoutGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    dx->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("DropoutGrad", {*x, *dout}, {*dx}, {});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    dropout, ops::DropoutV2NPUKernel<plat::NPUDeviceContext, float>,
    // ops::DropoutV2NPUKernel<plat::NPUDeviceContext, double>,
    ops::DropoutV2NPUKernel<plat::NPUDeviceContext, plat::float16>);
// REGISTER_OP_NPU_KERNEL(
//     dropout_grad,
//     ops::DropoutGradNPUKernel<plat::NPUDeviceContext, float>,
//     // ops::DropoutGradNPUKernel<plat::NPUDeviceContext, double>,
//     ops::DropoutGradNPUKernel<plat::NPUDeviceContext, plat::float16>);
