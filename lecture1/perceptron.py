from tinygrad import Tensor

def perceptron():

  inputs= Tensor.randint(1,10, low=-10,high=10)
  weights = Tensor.randn(1,10)
  bias = Tensor.randn(1,1)
  

  output = inputs.dot(weights.T).add(bias).sigmoid().numpy()

  return output

print(perceptron())
