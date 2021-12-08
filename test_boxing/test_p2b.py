import itertools
import os
import unittest
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import oneflow as flow


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

    def build_p2b(input_blob, src_device_num, dst_device_num):
        print(src_device_type, src_device_num, dst_device_type, dst_device_num)
        src_placement = flow.placement(src_device_type, {0:range(src_device_num)})
        src_sbp = flow.sbp.split(0)
        src = flow.Tensor(input_blob)
        print(src_placement, src_sbp)
        src = src.to_consistent(src_placement, src_sbp)
        src = flow.sum(src, dim=0)
        
        dst_placement = flow.placement(dst_device_type, {0:range(dst_device_num)})
        dst_sbp = flow.sbp.broadcast
        dst = src.to_consistent(dst_placement, dst_sbp)
        return dst.to_local()

    def partial_sum_to_broadcast_job(input_blob):
        result_list = []
        for i in (2, 3):
            for j in (1, 2, 3):
                result_list.append(build_p2b(input_blob, i, j))
        return tuple(result_list)

    x = np.random.uniform(-1e-05, 1e-05, (96, 96, 96)).astype(np.float32)
    result_tuple = partial_sum_to_broadcast_job(x)
    for out in result_tuple:
        test_case.assertTrue(np.allclose(np.sum(x, axis=0), out.numpy()))


class TestBoxingV2(unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_partial_sum_to_broadcast(test_case):
        arg_dict = OrderedDict()
        # arg_dict["src_device_type"] = ["cpu", "cuda"]
        # arg_dict["dst_device_type"] = ["cpu", "cuda"]
        arg_dict["src_device_type"] = ["cuda"]
        arg_dict["dst_device_type"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            _test_partial_sum_to_broadcast(test_case, *arg)

if __name__ == "__main__":
    unittest.main()