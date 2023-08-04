# all dependencies: numpy, scipy for decent convolution and entropy, tqdm for a
# cute progress bar.

# usage: call LeNet_wrapper()/MyNet_wrapper and wait.

# I first dreamed to do everything from scratch. Luckily I didn't carry it
# out in practice. If I do it in C/C++, memory management would be tedious;
# if I do it in a language with GC, floating point arithmetic would almost
# certainly involve expensive boxing/unboxing -- the implementation here (though
# can still be optimized) is already painfully slow! not to say a naieve
# AD engine.

import numpy as np
from dataset import get_data
from matplotlib import pyplot as plt
from tqdm import trange
from typing import Tuple

from nn import Real, Tensor, Layer, Network, LossFn, convolve_of, mean_pool_of, max_pool_of, linear1, rectify, flatten, fully_connect, softmax, cross_entropy, sigmoid, He_init_filter, He_init_mat, Xavier_init_filter, Xavier_init_mat
from train import train, run, set_step, set_nround, set_batch_size, set_momentum
from vis import squash, t_sne2


def one_hot(x: int, n: int) -> Tensor:
  h = np.zeros([n], dtype=Real)
  h[x] = 1
  return h

### LeNet ###

# structure
# I have to use rectify here because of gradient vanishing
input_shape = [1, 32, 32]
C1: Layer = convolve_of(1, 1)
C1_w_s = [6, 1, 5, 5]
C1_b_s = [6]
S2_1: Layer = mean_pool_of(2, 2)
S2_2: Layer = linear1
S2_2_w_s = [6]
S2_2_b_s = [6]
S2_3: Layer = rectify
C3: Layer = convolve_of(1, 1)
C3_w_s = [16, 6, 5, 5]
C3_b_s = [16]
S4_1: Layer = mean_pool_of(2, 2)
S4_2: Layer = linear1
S4_2_w_s = [16]
S4_2_b_s = [16]
S4_3: Layer = rectify
C5: Layer = flatten
C5_w_s = [120, 16, 5, 5]
C5_b_s = [120]
F6_1: Layer = fully_connect
F6_1_w_s = [84, 120]
F6_1_b_s = [84]
F6_2: Layer = rectify
F7: Layer = fully_connect
# in original LeNet, the output here is compared with standard bitmap. but
# we are not classifying digits so that's not very appropriate. however, I
# tried, and it's not so bad.
F7_w_s = [10, 84]
F7_b_s = [10]
Out: Layer = softmax
Loss: LossFn = cross_entropy

ntypes = 10

def LeNet_init() -> Network:
  C1_w = He_init_filter(C1_w_s)
  C1_b = np.zeros(C1_b_s, dtype=Real)
  S2_2_w = np.random.normal(0, np.sqrt(2), S2_2_w_s).astype(dtype=Real)
  S2_2_b = np.zeros(S2_2_b_s, dtype=Real)
  C3_w = He_init_filter(C3_w_s)
  C3_b = np.zeros(C3_b_s, dtype=Real)
  S4_2_w = np.random.normal(0, np.sqrt(2), S4_2_w_s).astype(dtype=Real)
  S4_w_b = np.zeros(S4_2_b_s, dtype=Real)
  C5_w = He_init_filter(C5_w_s)
  C5_b = np.zeros(C5_b_s, dtype=Real)
  F6_1_w = He_init_mat(F6_1_w_s)
  F6_1_b = np.zeros(F6_1_b_s, dtype=Real)
  F7_w = He_init_mat(F7_w_s)
  F7_b = np.zeros(F7_b_s, dtype=Real)
  return ([ C1,           S2_1, S2_2,             S2_3, C3,           S4_1, S4_2,            S4_3, C5,           F6_1,             F6_2, F7,           Out],
          [[C1_w, C1_b],  [],  [S2_2_w, S2_2_b],  [],  [C3_w, C3_b],  [],  [S4_2_w, S4_w_b], [],  [C5_w, C5_b], [F6_1_w, F6_1_b],  [],  [F7_w, F7_b],  []])

X_train, X_test, Y_train, Y_test = get_data('dataset')


def LeNet_test(net: Network, Xs: Tensor, Ys: Tensor) -> Tuple[float, Real]:
  correct = 0
  loss = np.float32(0)
  for i in trange(len(Xs)):
    g = run(net, Xs[i])[-1]
    if np.argmax(g) == Ys[i]:
      correct += 1
    loss += Loss(g, one_hot(Ys[i], ntypes))[0]
  return (correct / len(Xs), loss)


T_train = np.empty([len(Y_train), ntypes], dtype=Real)
for i in range(len(Y_train)):
  T_train[i] = one_hot(Y_train[i], ntypes)

### My network ###
M1: Layer = convolve_of(1, 1)
M2: Layer = max_pool_of(2, 2)
M3: Layer = rectify
M4: Layer = convolve_of(1, 1)
M5: Layer = max_pool_of(2, 2)
M6: Layer = rectify
M7: Layer = flatten
M8: Layer = rectify
M9: Layer = fully_connect
M10: Layer = rectify
M11: Layer = fully_connect
M12: Layer = softmax

M1_w_s = [4, 1, 5, 5]
M1_b_s = [4]
M4_w_s = [12, 4, 5, 5]
M4_b_s = [12]
M7_w_s = [96, 12, 5, 5]
M7_b_s = [96]
M9_w_s = [48, 96]
M9_b_s = [48]
M11_w_s = [10, 48]
M11_b_s = [10]

def MyNet_init() -> Network:
  M1_w = He_init_filter(M1_w_s)
  M1_b = np.zeros(M1_b_s, dtype=Real)
  M4_w = He_init_filter(M4_w_s)
  M4_b = np.zeros(M4_b_s, dtype=Real)
  M7_w = He_init_filter(M7_w_s)
  M7_b = np.zeros(M7_b_s, dtype=Real)
  M9_w = He_init_mat(M9_w_s)
  M9_b = np.zeros(M9_b_s, dtype=Real)
  M11_w = He_init_mat(M11_w_s)
  M11_b = np.zeros(M11_b_s, dtype=Real)
  return ([ M1,          M2, M3,  M4,          M5, M6,  M7,          M8,  M9,          M10,  M11,           M12],
          [[M1_w, M1_b], [], [], [M4_w, M4_b], [], [], [M7_w, M7_b], [], [M9_w, M9_b], [],  [M11_w, M11_b], []])



def linearize(t: Tensor) -> Tensor:
  return np.reshape(t, [np.size(t)])


colors = ['#232627',
          '#ed1515',
          '#11d116',
          '#f67400',
          '#1d99f3',
          '#9b59b6',
          '#1abc9c',
          '#fdbc4b',
          '#7f8c8d',
          '#1cdc9a']

def LeNet_wrapper():
  set_step(0.01)
  set_nround(50)
  set_batch_size(16)
  set_momentum(0.7)
  n = LeNet_init()
  (accu, loss) = LeNet_test(n, X_test, Y_test)
  accus = [accu]
  losses = [loss]
  for _ in range(4):
    n = train(n, Loss, X_train, T_train)
    (accu, loss) = LeNet_test(n, X_test, Y_test)
    losses.append(loss)
    accus.append(accu)

  set_step(0.005)
  set_batch_size(32)
  set_momentum(0.3)
  for _ in range(4):
    n = train(n, Loss, X_train, T_train)
    (accu, loss) = LeNet_test(n, X_test, Y_test)
    losses.append(loss)
    accus.append(accu)

  plt.plot(losses)
  plt.show()
  plt.cla()
  plt.plot(accus)
  plt.show()
  plt.cla()

  nsamples = 1000
  cs = [''] * nsamples
  conv_feats = np.empty([nsamples, C5_w_s[0]], dtype=Real)
  fc_feats = np.empty([nsamples, F6_1_w_s[0]], dtype=Real)
  out_feats = np.empty([nsamples, F7_w_s[0]], dtype=Real)
  for i in range(nsamples):
    cs[i] = colors[Y_test[i]]
    outputs = run(n, X_test[i])
    conv_feats[i] = outputs[8]
    fc_feats[i] = outputs[9]
    out_feats[i] = outputs[12] # the diagram is ... too weird

  feats = [conv_feats, fc_feats, out_feats]
  for feat in feats:
    xys = np.transpose(squash(feat, 2))
    xs = xys[0]
    ys = xys[1]
    for i in range(ntypes):
      plt.scatter([], [], color=colors[i], label=i)
    plt.legend()
    plt.scatter(xs, ys, color=cs)
    plt.show()
    plt.cla()

  for feat in feats:
    xs, ys = t_sne2(feat, 30)
    for i in range(ntypes):
      plt.scatter([], [], color=colors[i], label=i)
    plt.legend()
    plt.scatter(xs, ys, color=cs)
    plt.show()
    plt.cla()

def MyNet_wrapper():
  set_step(0.01)
  set_nround(50)
  set_batch_size(32)
  set_momentum(0.7)
  n = MyNet_init()
  (accu, loss) = LeNet_test(n, X_test, Y_test)
  accus = [accu]
  losses = [loss]
  for _ in range(4):
    n = train(n, Loss, X_train, T_train)
    (accu, loss) = LeNet_test(n, X_test, Y_test)
    losses.append(loss)
    accus.append(accu)

  set_step(0.005)
  set_batch_size(64)
  set_momentum(0.3)
  for _ in range(4):
    n = train(n, Loss, X_train, T_train)
    (accu, loss) = LeNet_test(n, X_test, Y_test)
    losses.append(loss)
    accus.append(accu)

  plt.plot(losses)
  plt.show()
  plt.cla()
  plt.plot(accus)
  plt.show()
  plt.cla()

  nsamples = 1000
  cs = [''] * nsamples
  conv_feats = np.empty([nsamples, M7_w_s[0]], dtype=Real)
  fc_feats = np.empty([nsamples, M9_w_s[0]], dtype=Real)
  out_feats = np.empty([nsamples, M11_w_s[0]], dtype=Real)
  for i in range(nsamples):
    cs[i] = colors[Y_test[i]]
    outputs = run(n, X_test[i])
    conv_feats[i] = outputs[6]
    fc_feats[i] = outputs[8]
    out_feats[i] = outputs[11]

  feats = [conv_feats, fc_feats, out_feats]
  for feat in feats:
    xys = np.transpose(squash(feat, 2))
    xs = xys[0]
    ys = xys[1]
    for i in range(ntypes):
      plt.scatter([], [], color=colors[i], label=i)
    plt.legend()
    plt.scatter(xs, ys, color=cs)
    plt.show()
    plt.cla()

  for feat in feats:
    xs, ys = t_sne2(feat, 30)
    for i in range(ntypes):
      plt.scatter([], [], color=colors[i], label=i)
    plt.legend()
    plt.scatter(xs, ys, color=cs)
    plt.show()
    plt.cla()

