from tinygrad import Tensor, nn

test = Tensor([1,2,3,4]).grad().numpy()

print(test)
