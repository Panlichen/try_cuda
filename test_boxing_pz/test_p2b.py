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
x_consistent = x.to_consistent(placement=placement, sbp=sbpp)
print("----------------------------------------")

print("After consistent, the x_consistent is :", x_consistent)
print("After consistent, the sbp of x_consistent is :", x_consistent.sbp)

y = flow.zeros(4, 4, placement=placement, sbp=sbpb)
y = flow.add(x_consistent, y)

print("----------------------------------------")
print("the y is :",y)
print("the sbp of y is :", y.sbp)
