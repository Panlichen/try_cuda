# Bug分析
```
Check failed: (parallel_desc->sorted_machine_ids().size()) == (parallel_desc->parallel_num()) (1 vs 2)
```
==符号两边的不相等，在你的代码里。我认为这个bug的大意是，你指定的parallel程度（等号右边），和你实际启动的进程数（等号左边）不一样。

可以参照其与Consistent View有关的[官方文档](https://docs.oneflow.org/master/parallelism/03_consistent_tensor.html)对代码进行改动。
# 使用方法
在15、16上使用可能会受到代理的负面影响，建议先
```shell
unset HTTP_PROXY
unset HTTPS_PROXY
unset https_proxy
unset http_proxy

echo $HTTP_PROXY
echo $HTTPS_PROXY
echo $https_proxy
echo $http_proxy
```


随后启动三个terminal分别输入，然后同时执行
```shell
MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=3 RANK=0 LOCAL_RANK=0 python test_p2b.py
```
```shell
MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=3 RANK=1 LOCAL_RANK=1 python test_p2b.py
```
```shell
MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=3 RANK=2 LOCAL_RANK=2 python test_p2b.py
```

运行single_client版本的p2b测试程序
```shell
ONEFLOW_TEST_DEVICE_NUM=4 python ../test_boxing/test_p2b_single_client.py
```

即可看到结果
# 代码解释
## 变量准备
```
import oneflow as flow

rank = int(flow.env.get_rank())
placement = flow.placement("cuda",{0:[0,1,2]})

sbps = flow.sbp.split(0)
sbpp = flow.sbp.partial_sum
sbpb = flow.sbp.broadcast

x = flow.Tensor([[1, 2, 3, 4], 
                 [1, 2, 3, 4], 
                 [1, 2, 3, 4], 
                 [1, 2, 3, 4]])

x = flow.sum(x, dim=0)
print("Before consistent, the x is :", x)
```
此时输出x在经过sum后的内容
```
Before consistent, the x is : tensor([ 4.,  8., 12., 16.], dtype=oneflow.float32)
```
其它值得注意的是，我前面说要启动三个terminal来同时运行函数是与placement对应的。如果placement只有0,1，那就启动两个terminal就行了，这也是引发那个bug的来源。
可以尝试不export任何内容，然后直接运行，能看到类似的错误。
## x consistent化
```
x_consistent = x.to_consistent(placement=placement, sbp=sbpp)
print("----------------------------------------")

print("After consistent, the x_consistent is :", x_consistent)
print("After consistent, the sbp of x_consistent is :", x_consistent.sbp)
```
得到的输出应该是三个卡上的x变量的partial_sum，即[12 24 36 48]。
```
----------------------------------------
After consistent, the x_consistent is : tensor([12., 24., 36., 48.],
       placement=oneflow.placement(device_type="cuda", machine_device_ids={0 : [0, 1, 2]}, hierarchy=(3,)),
       sbp=(oneflow.sbp.partial_sum,), dtype=oneflow.float32)
After consistent, the sbp of x_consistent is : (oneflow.sbp.partial_sum,)
```
## 导入到一个broadcast中
```
y = flow.zeros(4, 4, placement=placement, sbp=sbpb)
y = flow.add(x_consistent, y)

print("----------------------------------------")
print("the y is :",y)
print("the sbp of y is :", y.sbp)
```
y本来是一个4*4的全零矩阵，然后，加了一个1\*4的矩阵，OneFlow的add有广播的作用，所以最后的结果如下：
```
----------------------------------------
the y is : tensor([[12., 24., 36., 48.],
        [12., 24., 36., 48.],
        [12., 24., 36., 48.],
        [12., 24., 36., 48.]],
       placement=oneflow.placement(device_type="cuda", machine_device_ids={0 : [0, 1, 2]}, hierarchy=(3,)),
       sbp=(oneflow.sbp.broadcast,), dtype=oneflow.float32)
the sbp of y is : (oneflow.sbp.broadcast,)
```
请师兄评估一下这段程序的正确性？
