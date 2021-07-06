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

#include "paddle/fluid/operators/interpolate_v2_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class BilinearInterpV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* outSize = ctx.Input<Tensor>("OutSize");
    auto* out = ctx.Output<Tensor>("Out");

    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    bool half_pixel_centers = (align_mode == 1) ? false : true;

    LOG(WARNING) << "align_corners: " << align_corners;
    LOG(WARNING) << "align_mode: " << align_mode;
    LOG(WARNING) << "half_pixel_centers: " << half_pixel_centers;

    LOG(WARNING) << "x: " << x;
    LOG(WARNING) << "outSize: " << outSize;
    LOG(WARNING) << "out: " << out;

    int numel = x->numel();
    LOG(WARNING) << "x numel: " << numel;
    LOG(WARNING) << "outSize numel: " << outSize->numel();
    LOG(WARNING) << "out numel: " << out->numel();
    LOG(WARNING) << "x dims: " << x->dims();
    LOG(WARNING) << "out dims: " << out->dims();
    LOG(WARNING) << "outSize dims: " << outSize->dims();

    // std::ostringstream oss2;
    // for (int i = 0; i < numel; ++i) {
    //   // oss2 << x->data<T>()[i] << ",";
    //   // printf("%f, ", x->data<T>()[i]);
    // }

    out->Resize(x->dims());
    // out->Resize(framework::make_ddim({2, 3, 3, 3}));

    out->mutable_data<T>(ctx.GetPlace());

    LOG(WARNING) << "out: " << out;
    LOG(WARNING) << "out numel: " << out->numel();
    LOG(WARNING) << "out dims: " << out->dims();

    // const auto& runner = NpuOpRunner("ResizeBilinearV2",
    //                                  {
    //                                      *x,
    //                                  },
    //                                  {*out}, {});

    // const auto& runner = NpuOpRunner("ResizeBilinearV2", {*x,}, {*out},
    // {{"align_corners", align_corners}});

    // const auto& runner =
    //     NpuOpRunner("ResizeBilinearV2",
    //                 {
    //                     *x,
    //                 },
    //                 {*out}, {{"align_corners", align_corners},
    //                          {"half_pixel_centers", half_pixel_centers}});

    const auto& runner =
        NpuOpRunner("ResizeBilinearV2", {*x, *outSize}, {*out},
                    {{"align_corners", align_corners},
                     {"half_pixel_centers", half_pixel_centers}});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);

    LOG(WARNING) << "out: " << out;
  }
};

template <typename DeviceContext, typename T>
class BilinearInterpV2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* outSize = ctx.Input<Tensor>("OutSize");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    dx->mutable_data<T>(ctx.GetPlace());

    bool half_pixel_centers = (align_mode == 1) ? false : true;

    // const auto& runner = NpuOpRunner("ResizeBilinearV2Grad", {*x, *outSize,
    // *dout}, {*dx}, {});
    const auto& runner =
        NpuOpRunner("ResizeBilinearV2Grad", {*x, *outSize, *dout}, {*dx},
                    {{"align_corners", align_corners},
                     {"half_pixel_centers", half_pixel_centers}});

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
    bilinear_interp_v2,
    ops::BilinearInterpV2NPUKernel<plat::NPUDeviceContext, float>,
    // ops::BilinearInterpV2NPUKernel<plat::NPUDeviceContext, double>,
    ops::BilinearInterpV2NPUKernel<plat::NPUDeviceContext, plat::float16>);
REGISTER_OP_NPU_KERNEL(
    bilinear_interp_v2_grad,
    ops::BilinearInterpV2GradNPUKernel<plat::NPUDeviceContext, float>,
    // ops::BilinearInterpV2GradNPUKernel<plat::NPUDeviceContext, double>,
    ops::BilinearInterpV2GradNPUKernel<plat::NPUDeviceContext, plat::float16>);
