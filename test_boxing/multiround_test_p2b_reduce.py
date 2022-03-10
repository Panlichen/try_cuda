"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import itertools
import os
import unittest
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft


def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            if "gpu" in set:
                set.remove("gpu")
            if "cuda" in set:
                set.remove("cuda")
    return itertools.product(*sets)

def GenArgList(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    assert all([isinstance(x, list) for x in arg_dict.values()])
    sets = [arg_set for (_, arg_set) in arg_dict.items()]
    return GenCartesianProduct(sets)

def _test_partial_sum_to_broadcast(test_case, src_device_type, dst_device_type):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_p2b(input_blob, src_device_num, dst_device_num):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(input_blob.with_distribute(flow.distribute.split(0)))
            src = flow.math.reduce_sum(src, axis=0)
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
        return dst

    @flow.global_function(function_config=func_config)
    def partial_sum_to_broadcast_job(input_blob: oft.Numpy.Placeholder(shape)):
        result_list = []
        # for i in (2, 3):
        #     for j in (1, 2, 3):
        for i in (4, ):
            for j in (1, ):
                result_list.append(build_p2b(input_blob, i, j))
        return tuple(result_list)

    x = np.random.uniform(-1e-05, 1e-05, shape).astype(np.float32)
    # x = np.ones(shape).astype(np.float32)
    # x = np.full(shape, 1).astype(np.float32)
    # x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)
    # x = np.array([[4, 4], [3, 3], [2, 2], [1, 1]]).astype(np.float32)
    ori_x = x.copy()
    print("the shape is", x.shape)
    for round in range(warm_up_rounds):
        warmup_result_tuple = partial_sum_to_broadcast_job(x).get()
    print("warmup done")
    for round in range(test_rounds):
        result_tuple = partial_sum_to_broadcast_job(x).get()
        # print("result_tuple[0].numpy()", result_tuple[0].numpy())
        if round >= 10 and round % int(test_rounds / 10) == 0:
            print("round", round, "done")
    # print("np.allclose(x, ori_x):", np.allclose(x, ori_x))
    # print("np.allclose(warmup_result_tuple[0], result_tuple[0]):", np.allclose(warmup_result_tuple[0].numpy(), result_tuple[0].numpy()))
    # print("warmup_result_tuple[0].numpy():", warmup_result_tuple[0].numpy())
    for out in result_tuple:
        test_case.assertTrue(np.allclose(np.sum(x, axis=0), out.numpy()))

shape = (1024, 1024, 4)
# shape = (4, 2)
warm_up_rounds = 10
test_rounds = 100

@flow.unittest.skip_unless_1n4d()
class TestBoxingV2(flow.unittest.TestCase):

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_partial_sum_to_broadcast(test_case):
        arg_dict = OrderedDict()
        # arg_dict["src_device_type"] = ["cpu", "gpu"]
        # arg_dict["dst_device_type"] = ["cpu", "gpu"]
        arg_dict["src_device_type"] = ["gpu"]
        arg_dict["dst_device_type"] = ["gpu"]
        for arg in GenArgList(arg_dict):
            _test_partial_sum_to_broadcast(test_case, *arg)


if __name__ == "__main__":
    import os
    if "ONEFLOW_ENABLE_OFCCL" in os.environ.keys():
        print("ONEFLOW_ENABLE_OFCCL is", os.environ.get("ONEFLOW_ENABLE_OFCCL"))
    else:
        print("ONEFLOW_ENABLE_OFCCL is None")
    unittest.main()
