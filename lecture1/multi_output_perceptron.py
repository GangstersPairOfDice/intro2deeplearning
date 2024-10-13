from tinygrad import Tensor

def multi_layer_nn():

  inputs= Tensor.randint(1,10, low=-10,high=10)
  weights_1 = Tensor.randn(1,10)
  weights_2 = Tensor.randn(1,10)
  bias_1 = Tensor.randn(1,1)
  bias_2 = Tensor.randn(1,1)

  output = inputs.dot(weights_1.T).add(bias_1).sigmoid().cat(inputs.dot(weights_2.T).add(bias_2).sigmoid(), dim=0)

  final_out 

  # only realize at end, to save ops maybe
  return final_output

print(multi_layer_nn())
