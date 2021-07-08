#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.nn.functional import interpolate
import paddle

paddle.enable_static()


def bilinear_interp_np(input,
                       out_h,
                       out_w,
                       scale_w=0,
                       scale_h=0,
                       out_size=None,
                       actual_shape=None,
                       align_corners=True,
                       align_mode=0,
                       data_layout='NCHW'):
    """bilinear interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for i in range(out_h):
        if (align_mode == 0 and not align_corners):
            h = int(ratio_h * (i + 0.5) - 0.5)
        else:
            h = int(ratio_h * i)

        h = max(0, h)
        hid = 1 if h < in_h - 1 else 0
        if (align_mode == 0 and not align_corners):
            idx_src_h = max(ratio_h * (i + 0.5) - 0.5, 0)
            h1lambda = idx_src_h - h
        else:
            h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
        for j in range(out_w):
            if (align_mode == 0 and not align_corners):
                w = int(ratio_w * (j + 0.5) - 0.5)
            else:
                w = int(ratio_w * j)
            w = max(0, w)
            wid = 1 if w < in_w - 1 else 0
            if (align_mode == 0 and not align_corners):
                idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
                w1lambda = idx_src_w - w
            else:
                w1lambda = ratio_w * j - w
            w2lambda = 1.0 - w1lambda

            out[:, :, i, j] = h2lambda*(w2lambda*input[:, :, h, w] +
                                        w1lambda*input[:, :, h, w+wid]) + \
                h1lambda*(w2lambda*input[:, :, h+hid, w] +
                          w1lambda*input[:, :, h+hid, w+wid])

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUBilinearInterpV2Op(OpTest):
    def setUp(self):
        self.op_type = "bilinear_interp_v2"
        self.set_npu()
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        input_np = np.random.random(self.input_shape).astype("float32")

        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]
        scale_h = 0
        scale_w = 0
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.:
                    scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[1]
                scale_h = self.scale[0]
            out_h = int(in_h * scale_h)
            out_w = int(in_w * scale_w)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = bilinear_interp_np(
            input_np, out_h, out_w, 0, 0, self.out_size, self.actual_shape,
            self.align_corners, self.align_mode, self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout
        }
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    #    def test_check_grad(self):
    #        self.check_grad_with_place(self.place, ['X'], 'Out', in_place=True, check_dygraph=False)

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(5)

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


if __name__ == "__main__":
    unittest.main()
