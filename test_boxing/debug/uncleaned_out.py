import itertools

from cv2 import reduce

num_round = 10

def calc_out(input):
    reduce_result = []
    out = [0 for i in range(4)]
    for round in range(num_round):
        out[0] = int(input[0])
        for i in range(3):
            out[i + 1] = out[i] + out[i + 1] + int(input[i + 1])
        reduce_result.append(out[3])
    print("for {}, reduce_result for {} rounds is {}".format(
        input, num_round, reduce_result
    ))

for input in itertools.permutations('1234', 4):
    calc_out(input)

print("----------")

calc_out([1, 1, 1, 1])
calc_out([2, 2, 2, 2])