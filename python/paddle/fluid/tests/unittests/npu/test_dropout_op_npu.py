#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu() or True,
                 "core is not compiled with NPU")
class TestNPUDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.35, 'fix_seed': True, 'is_test': False}
        self.outputs = {'Out': self.inputs['X']}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

#    def test_check_grad(self):
#        self.check_grad_with_place(self.place, ['X'], 'Out', check_dygraph=False)

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(5)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUDropoutOpWithSeed(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.inputs = {
            "X": np.random.random((32, 64)).astype("float32"),
            "Seed": np.asarray(
                [12.5], dtype="float32")
        }
        self.attrs = {'dropout_prob': 0.35, }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('int32')
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

#    def test_check_grad_normal(self):
#        self.check_grad_with_place(self.place, ['X'], 'Out', max_relative_error=0.05, check_dygraph=False)

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(5)

if __name__ == "__main__":
    unittest.main()
