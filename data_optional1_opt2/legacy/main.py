# This is very, very slow. The main reason is: one batch is not processed in one
# run like that in torch. e.g., a batch of size 64 is processed by running the
# network 64 times. That means the same fft of the convolution kernel is
# computed 64 times. I didn't expect this when designing the architecture and
# there's no turning back. But I doubt whether I can properly implement it even
# if I knew it because it's much more complex.

import numpy as np
from dataset import get_data
from matplotlib import pyplot as plt
from tqdm import trange
import random
from typing import Tuple
import time

from nn import Parameter, Real, Tensor, Layer, Network, convolve_of, l2_loss, max_pool_of, rectify, fully_connect, repeate_of, sigmoid, reshape_of, dilate_of, reparam, kl_std, fn_of, param_of, stack, He_init_filter, He_init_mat, Xavier_init_filter, Xavier_init_mat
from train import gradient, run, map_params, map2_params

E1: Layer = dilate_of(34, 34, 1, 1, 1, 1)
E2: Layer = convolve_of(1, 1)
E3: Layer = max_pool_of(2, 2)
E4: Layer = rectify
E5: Layer = convolve_of(1, 1)
E6: Layer = max_pool_of(2, 2)
E7: Layer = rectify
E8: Layer = reshape_of([32 * 7 * 7])
E9: Layer = fully_connect
E10: Layer = rectify
E11: Layer = fully_connect

S0: Layer = reparam

D1: Layer = fully_connect
D2: Layer = rectify
D3: Layer = fully_connect
D4: Layer = rectify
D5: Layer = reshape_of([32, 6, 6])
D6: Layer = repeate_of(2, 2)
D7: Layer = convolve_of(1, 1)
D8: Layer = rectify
D9: Layer = repeate_of(2, 2)
D10: Layer = convolve_of(1, 1)
D11: Layer = rectify
D12: Layer = repeate_of(2, 2)
D13: Layer = convolve_of(1, 1)
D14: Layer = sigmoid

E2_w_s = [8, 3, 3, 3]
E2_b_s = [8]
E5_w_s = [32, 8, 3, 3]
E5_b_s = [32]
E9_w_s = [32 * 16, 32 * 7 * 7]
E9_b_s = [32 * 16]
E11_w_s = [64, 32 * 16]
E11_b_s = [64]

D1_w_s = [32 * 8, 32]
D1_b_s = [32 * 8]
D3_w_s = [32 * 6 * 6, 32 * 8]
D3_b_s = [32 * 6 * 6]
D7_w_s = [32, 32, 3, 3]
D7_b_s = [32]
D10_w_s = [8, 32, 4, 4]
D10_b_s = [8]
D13_w_s = [3, 8, 3, 3]
D13_b_s = [3]

enc_layers = [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11]
dec_layers = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13]

def Encoder_init() -> Network:
  E2_w = He_init_filter(E2_w_s)
  E5_w = He_init_filter(E5_w_s)
  E9_w = He_init_mat(E9_w_s)
  E11_w = Xavier_init_mat(E11_w_s)
  E2_b = np.zeros(E2_b_s, dtype=Real)
  E5_b = np.zeros(E5_b_s, dtype=Real)
  E9_b = np.zeros(E9_b_s, dtype=Real)
  E11_b = np.zeros(E11_b_s, dtype=Real)
  return ([E1,  E2,          E3, E4,  E5,          E6, E7, E8,  E9,          E10, E11],
          [[], [E2_w, E2_b], [], [], [E5_w, E5_b], [], [], [], [E9_w, E9_b], [], [E11_w, E11_b]])

sampler: Network = ([S0], [[]])

def Decoder_init() -> Network:
  D1_w = He_init_mat(D1_w_s)
  D3_w = He_init_mat(D3_w_s)
  D7_w = He_init_filter(D7_w_s)
  D10_w = He_init_filter(D10_w_s)
  D13_w = Xavier_init_filter(D13_w_s)
  D1_b = np.zeros(D1_b_s, dtype=Real)
  D3_b = np.zeros(D3_b_s, dtype=Real)
  D7_b = np.zeros(D7_b_s, dtype=Real)
  D10_b = np.zeros(D10_b_s, dtype=Real)
  D13_b = np.zeros(D13_b_s, dtype=Real)
  return ([ D1,          D2,  D3,          D4, D5, D6,  D7,          D8, D9,  D10,           D11, D12, D13],
          [[D1_w, D1_b], [], [D3_w, D3_b], [], [], [], [D7_w, D7_b], [], [], [D10_w, D10_b], [],  [], [D13_w, D13_b]])

X_train = get_data('dataset')
# better if returns X/255 rather than (X-x_min)/(x_max-x_min). but likely to be the same

step = 0
nround = 0
batch_size = 0
momentum = 0
duration = 0 # time to train

def VAE_batch_grad(n: int, samples: Tensor, targets: Tensor, enc: Network, dec: Network) -> list[Parameter]:
  all = len(samples)
  gen = stack(sampler, dec)
  ed = stack(enc, gen)
  diff_ed = map_params(np.zeros_like, param_of(ed))
  for _ in range(n):
    ind = random.randint(0, all - 1)
    input = samples[ind]
    outputs_e = run(enc, input)
    outputs_ed = outputs_e + run(gen, outputs_e[-1])
    diff_ed = map2_params(lambda t, s: t + s,
                          diff_ed,
                          gradient(input, outputs_ed, ed, targets[ind], l2_loss))
  return map_params(lambda t: t / n, diff_ed)

def VAE_train(enc: Network, dec: Network, samples: Tensor) -> Tuple[Network, Network]:
  le = len(fn_of(enc))
  acc = VAE_batch_grad(batch_size, samples, samples, enc, dec)
  begin = time.time()
  while time.time() - begin < duration:
  # for _ in trange(nround):
    diff = VAE_batch_grad(batch_size, samples, samples, enc, dec)
    acc = (
           map2_params(lambda t, s: momentum * t + (1 - momentum) * s, acc, diff))
    enc = (fn_of(enc),
           map2_params(lambda t, s: s - step * t, acc[:le], param_of(enc)))
    dec = (fn_of(dec),
           map2_params(lambda t, s: s - step * t, acc[le + 1:], param_of(dec)))
  return (enc, dec)

def unit(n: int) -> Tensor:
  return np.random.multivariate_normal(np.zeros([n]), np.diag(np.ones([n]))).astype(Real)

def show(rgb: Tensor):
  img = np.empty([32, 32, 3])
  for i in range(32):
    for j in range(32):
      for k in range(3):
        img[i, j, k] = rgb[k, i, j]
  plt.axis('off')
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img)

def test(enc: Network, dec: Network) -> Real:
  L: Real = np.float32(0)
  ed = stack(enc, stack(sampler, dec))
  for i in trange(len(X_train)):
    L += l2_loss(run(ed, X_train[i])[-1], X_train[i])[0]
  return L

enc_files: list[list[str]] = [[], ["saved/E2_w.npy", "saved/E2_b.npy"], [], [], ["saved/E5_w.npy", "saved/E5_b.npy"], [], [], [], ["saved/E9_w.npy", "saved/E9_b.npy"], [], ["saved/E11_w.npy", "saved/E11_b.npy"]]
dec_files: list[list[str]] = [["saved/D1_w.npy", "saved/D1_b.npy"], [], ["saved/D3_w.npy", "saved/D3_b.npy"], [], [], [], ["saved/D7_w.npy", "saved/D7_b.npy"], [], [], ["saved/D10_w.npy", "saved/D10_b.npy"], [],  [], ["saved/D13_w.npy", "saved/D13_b.npy"]]

def save_models(enc: Network, dec: Network):
  for (ps, fs) in zip(param_of(enc), enc_files):
    for (p, f) in zip(ps, fs):
      np.save(f, p)
  for (ps, fs) in zip(param_of(dec), dec_files):
    for (p, f) in zip(ps, fs):
      np.save(f, p)

def load_models() -> Tuple[Network, Network]:
  enc_params = []
  dec_params = []
  for fs in enc_files:
    t = []
    for f in fs:
      t.append(np.load(f))
    enc_params.append(t)
  for fs in dec_files:
    t = []
    for f in fs:
      t.append(np.load(f))
    dec_params.append(t)
  return ((enc_layers, enc_params),
          (dec_layers, dec_params))

def VAE_wrapper(enc: Network, dec: Network):
  global step, nround, batch_size, momentum
  step = 0.000003
  batch_size = 32
  momentum = 0.3
  enc, dec = VAE_train(enc, dec, X_train)
  # print(test(enc, dec))
  # step = 0.000005
  # nround = 10
  # batch_size = 64
  # for _ in range(5):
    # print(test(, dec))
    # enc, dec = VAE_train(enc, dec, X_train)
  return enc, dec

enc, dec = load_models()
for i in range(32):
  t = random.randint(a=0, b=len(X_train) - 1)
  plt.subplot(8, 8, 2 * i + 1)
  show(X_train[t])
  plt.subplot(8, 8, 2 * i + 2)
  show(run(stack(sampler, dec), run(enc, X_train[t])[-1])[-1])
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
