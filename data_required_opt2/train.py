import numpy as np
from tqdm import trange
import random
from typing import Callable

from nn import Tensor, Parameter, Network, fn_of, param_of, prim_of, adjoint_of, LossFn

### Training ###

def run(network: Network, input: Tensor) -> list[Tensor]:
  outputs = []
  for (fn, param) in zip(fn_of(network), param_of(network)):
    input = prim_of(fn)(param, input)
    outputs.append(input)
  return outputs

def gradient(input: Tensor, outputs: list[Tensor], network: Network, target: Tensor, loss: LossFn) -> list[Parameter]:
  guess = outputs[-1]
  (_, adj) = loss(guess, target)
  depth = len(network[0])
  gradient = [[]] * depth
  fns = fn_of(network)
  params = param_of(network)
  for i in range(depth - 1, 0, -1):
    (adj, grad) = adjoint_of(fns[i])(params[i], adj, outputs[i - 1])
    gradient[i] = grad
  (_, gradient[0]) = adjoint_of(fns[0])(params[0], adj, input)
  return gradient

def map2_params(f: Callable[[Tensor, Tensor], Tensor], old: list[Parameter], diff: list[Parameter]) -> list[Parameter]:
  pss = []
  for (os, ds) in zip(old, diff):
    ps = []
    for (o, d) in zip(os, ds):
      ps.append(f(o, d))
    pss.append(ps)
  return pss

def map_params(f: Callable[[Tensor], Tensor], base: list[Parameter]) -> list[Parameter]:
  pss = []
  for ts in base:
    ps = []
    for t in ts:
      ps.append(f(t))
    pss.append(ps)
  return pss

def batch_grad(n: int, samples: Tensor, targets: Tensor, network: Network, loss: LossFn) -> list[Parameter]:
  all = len(samples)
  diff = map_params(np.zeros_like, param_of(network))
  for _ in range(n):
    ind = random.randint(0, all - 1)
    input = samples[ind]
    outputs = run(network, input)
    diff = map2_params(lambda t, s: t + s,
                       diff,
                       gradient(input, outputs, network, targets[ind], loss))
  return map_params(lambda t: t / n, diff)

# hyper
step = 0
nround = 0
batch_size = 0
momentum = 0
def set_step(x: float):
  global step
  step = x
def set_nround(x: float):
  global nround
  nround = x
def set_batch_size(x: float):
  global batch_size
  batch_size = x
def set_momentum(x: float):
  global momentum
  momentum = x

# hyper (global) parameters needed: nround, batch_size, momentum, step
def train(net: Network, loss: LossFn, samples: Tensor, targets: Tensor) -> Network:
  acc = batch_grad(batch_size, samples, targets, net, loss)
  for _ in trange(nround):
    diff = batch_grad(batch_size, samples, targets, net, loss)
    acc = map2_params(lambda t, s: momentum * t + (1 - momentum) * s, acc, diff)
    net = (fn_of(net),
           map2_params(lambda t, s: s - step * t, acc, param_of(net)))
  return net
