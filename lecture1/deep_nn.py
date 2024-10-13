from tinygrad import Tensor, nn

# making a deep nn using
# Linear layers the easy way

class deep_nn:
  def __init__(self):
    self.l1 = Linear(3, 4, bias=False)
    self.l2 = Linear(4, 2, bias=False)

  def __call__(self, x):
    x = self.l1(x)
    x = x.sigmoid()
    x = self.l2(x)
    x = x.sigmoid()
    return x

deep_nn()
