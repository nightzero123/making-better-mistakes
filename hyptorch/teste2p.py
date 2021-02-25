import sys
print(sys.path)

from hyptorch.nn import ToPoincare
import torch as th
e2p = ToPoincare(1, False, False)
p = e2p(th.tensor([1, 2, 3, 4, 5], dtype=float))
print(p)

