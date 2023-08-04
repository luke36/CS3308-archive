import numpy as np
from dataset import get_data
from matplotlib import pyplot as plt
from tqdm import trange
import random
from typing import Tuple

from nn import Parameter, Real, Tensor, Layer, Network, convolve_of, l2_loss, max_pool_of, mean_pool_of, rectify, flatten, fully_connect, repeate_of, softmax, cross_entropy, sigmoid, reshape_of, dilate_of, reparam, kl_std, fn_of, param_of, stack
from train import gradient, run, map_params, map2_params

E2: Layer = convolve_of(1, 1)
E3: Layer = max_pool_of(2, 2)
E4: Layer = rectify
E6: Layer = convolve_of(1, 1)
E7: Layer = max_pool_of(2, 2)
E8: Layer = rectify
E9: Layer = flatten
E11: Layer = rectify
E10: Layer = fully_connect

S0: Layer = reparam

D1: Layer = fully_connect
D2: Layer = reshape_of([16, 6, 6])
D3: Layer = repeate_of(2, 2)
D4: Layer = convolve_of(1, 1)
D5: Layer = rectify
D6: Layer = repeate_of(2, 2)
D7: Layer = convolve_of(1, 1)
D8: Layer = rectify
D9: Layer = repeate_of(2, 2)
D10: Layer = convolve_of(1, 1)
D11: Layer = sigmoid

E2_w_s = [6, 1, 5, 5]
E2_b_s = [6]
E6_w_s = [16, 6, 5, 5]
E6_b_s = [16]
E9_w_s = [120, 16, 5, 5]
E9_b_s = [120]
E10_w_s = [32, 120]
E10_b_s = [32]

D1_w_s = [16 * 6 * 6, 16]
D1_b_s = [16 * 6 * 6]
D4_w_s = [16, 16, 3, 3]
D4_b_s = [16]
D7_w_s = [6, 16, 4, 4]
D7_b_s = [6]
D10_w_s = [1, 6, 3, 3]
D10_b_s = [1]

def He_init_filter(s: list[int]) -> Tensor:
  return np.random.normal(0,
                          2 / np.sqrt(2 * s[1] * s[2] * s[3]),
                          s)

def Xavier_init_filter(s: list[int]) -> Tensor:
  return np.random.normal(0,
                          1 / np.sqrt(s[1] * s[2] * s[3]),
                          s)

def He_init_mat(s: list[int]) -> Tensor:
  return np.random.normal(0,
                          2 / np.sqrt(2 * s[1]),
                          s)

def Encoder_init() -> Network:
  E2_w = He_init_filter(E2_w_s)
  E6_w = He_init_filter(E6_w_s)
  E9_w = He_init_filter(E9_w_s)
  E10_w = He_init_mat(E10_w_s)
  E2_b = np.zeros(E2_b_s)
  E6_b = np.zeros(E6_b_s)
  E9_b = np.zeros(E9_b_s)
  E10_b = np.zeros(E10_b_s)
  return ([ E2,          E3, E4,  E6,          E7, E8,  E9,          E11, E10],
          [[E2_w, E2_b], [], [], [E6_w, E6_b], [], [], [E9_w, E9_b], [], [E10_w, E10_b]])

sampler: Network = ([S0], [[]])

def Decoder_init() -> Network:
  D1_w = He_init_mat(D1_w_s)
  D4_w = He_init_filter(D4_w_s)
  D7_w = He_init_filter(D7_w_s)
  D10_w = Xavier_init_filter(D10_w_s)
  # for f in [D4_w, D7_w, D10_w]:
  #   for b in range(np.shape(f)[0]):
  #     for d in range(np.shape(f)[1]):
  #       f[b][d][0][0] = np.random.normal(0, 1/np.sqrt(64))
  #       f[b][d][0][2] = np.random.normal(0, 1/np.sqrt(64))
  #       f[b][d][2][0] = np.random.normal(0, 1/np.sqrt(64))
  #       f[b][d][2][2] = np.random.normal(0, 1/np.sqrt(64))
  #       f[b][d][0][1] = np.random.normal(0, 1/np.sqrt(32))
  #       f[b][d][1][0] = np.random.normal(0, 1/np.sqrt(32))
  #       f[b][d][1][2] = np.random.normal(0, 1/np.sqrt(32))
  #       f[b][d][2][1] = np.random.normal(0, 1/np.sqrt(32))
  #       f[b][d][1][1] = np.random.normal(0, 1/np.sqrt(16))
  #   print(f)
  D1_b = np.zeros(D1_b_s)
  D4_b = np.zeros(D4_b_s)
  D7_b = np.zeros(D7_b_s)
  D10_b = np.zeros(D10_b_s)
  return ([ D1,          D2, D3,  D4,          D5, D6,  D7,          D8, D9,  D10,           D11],
          [[D1_w, D1_b], [], [], [D4_w, D4_b], [], [], [D7_w, D7_b], [], [], [D10_w, D10_b], []])

X_train, _, _, _ = get_data('dataset')
# better if returns X/255 rather than (X-x_min)/(x_max-x_min). but likely to be the same

step = 0
nround = 0
batch_size = 0
momentum = 0

def VAE_batch_grad(n: int, samples: Tensor, targets: Tensor, enc: Network, dec: Network) -> Tuple[list[Parameter], list[Parameter]]:
  all = len(samples)
  ed = stack(enc, stack(sampler, dec))
  diff_e = map_params(np.zeros_like, param_of(enc))
  diff_ed = map_params(np.zeros_like, param_of(ed))
  for _ in range(n):
    ind = random.randint(0, all - 1)
    input = samples[ind]
    outputs_e = run(enc, input)
    outputs_ed = run(ed, input)
    diff_e = map2_params(lambda t, s: t + s,
                         diff_e,
                         gradient(input, outputs_e, enc, targets[ind], kl_std))
    diff_ed = map2_params(lambda t, s: t + s,
                          diff_ed,
                          gradient(input, outputs_ed, ed, targets[ind], l2_loss))
  return (map_params(lambda t: t / n, diff_e), map_params(lambda t: t / n, diff_ed))

def VAE_train(enc: Network, dec: Network, samples: Tensor) -> Tuple[Network, Network]:
  le = len(fn_of(enc))
  acc = VAE_batch_grad(batch_size, samples, samples, enc, dec)
  for _ in trange(nround):
    diff = VAE_batch_grad(batch_size, samples, samples, enc, dec)
    acc = (map2_params(lambda t, s: momentum * t + (1 - momentum) * s, acc[0], diff[0]),
           map2_params(lambda t, s: momentum * t + (1 - momentum) * s, acc[1], diff[1]))
    enc = (fn_of(enc),
           map2_params(lambda t, s: s - step * t, acc[1][:le],
           map2_params(lambda t, s: s - 0.0001 * step * t, acc[0], param_of(enc))))
    dec = (fn_of(dec),
           map2_params(lambda t, s: s - step * t, acc[1][le + 1:], param_of(dec)))
  return (enc, dec)

def unit(n: int) -> Tensor:
  return np.random.multivariate_normal(np.zeros([n]), np.diag(np.ones([n])))

def show(rgb: Tensor):
  img = np.empty([32, 32, 3])
  for i in range(32):
    for j in range(32):
      for k in range(3):
        img[i, j, k] = rgb[k, i, j]
  plt.imshow(img)
  plt.show()
  plt.cla()

def test(enc: Network, dec: Network) -> Real:
  L: Real = np.float64(0)
  ed = stack(enc, stack(sampler, dec))
  for i in trange(len(X_train)):
    L += l2_loss(run(ed, X_train[i])[-1], X_train[i])[0]
  return L

def VAE_wrapper():
  global step, nround, batch_size, momentum
  enc = Encoder_init()
  dec = Decoder_init()
  step = 0.00001
  nround = 100
  batch_size = 16
  momentum = 0.9
  for _ in range(5):
    print(test(enc, dec))
    enc, dec = VAE_train(enc, dec, X_train)
  print(test(enc, dec))
  # step = 0.000005
  # nround = 10
  # batch_size = 64
  # for _ in range(5):
    # print(test(enc, dec))
    # enc, dec = VAE_train(enc, dec, X_train)
  return enc, dec

