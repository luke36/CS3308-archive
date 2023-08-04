import numpy as np
import numpy.typing as npt
from scipy.signal import correlate, convolve
from typing import Callable, Tuple

### Types ###

Real = np.float32
Tensor = npt.NDArray[Real]
Parameter = list[Tensor]
# given parameter and primal from last layer, output its primal
PrimFn = Callable[[Parameter, Tensor], Tensor]
# given parameter,  adjoint of next layer and primal from last layer, output
# its adjoint and gradient of parameter
#                            adjoint primal
AdjFn = Callable[[Parameter, Tensor, Tensor], Tuple[Tensor, Parameter]]
# a neuro network layer
Layer = Tuple[PrimFn, AdjFn]
#                  guess   target
LossFn = Callable[[Tensor, Tensor], Tuple[Real, Tensor]]
Network = Tuple[list[Layer], list[Parameter]]


### Some layers (or layer generators) ###

# fully connected layer
def fc_p(w_b: Parameter, v: Tensor) -> Tensor:
  return np.dot(w_b[0], v) + w_b[1]
def fc_a(w_b: Parameter, adj: Tensor, v: Tensor) -> Tuple[Tensor, Parameter]:
  return (np.dot(np.transpose(w_b[0]), adj),
          [np.dot(np.expand_dims(adj, axis=1),
                  np.expand_dims(v, axis=0)), adj])
fully_connect: Layer = (fc_p, fc_a)


# radial basis
# def rb_p(genuien: Parameter, v: Tensor) -> Tensor:
#   diff = v - genuien[0]
#   return np.sum(diff * diff, axis=1)
# def rb_a(genuine: Parameter, adj: Tensor, v: Tensor) -> Tuple[Tensor, Parameter]:
#   diff = 2 * adj * (v - genuine[0])
#   return (np.sum(diff, axis=0),
#           [-diff])
# radial_basis = (rb_p, rb_a)


# sigmoid
def sigmoid_p(_, t: Tensor) -> Tensor:
  return 1 / (1 + np.exp(-t))
def sigmoid_a(_, adj: Tensor, t: Tensor) -> Tuple[Tensor, Parameter]:
  ex = np.exp(-t)
  return (adj * ex / np.square(1 + ex), [])
sigmoid: Layer = (sigmoid_p, sigmoid_a)

# relu
def relu_p(_, t: Tensor) -> Tensor:
  return np.maximum(0, t)
def relu_a(_, adj: Tensor, t: Tensor) -> Tuple[Tensor, Parameter]:
  return (adj * (t >= 0), [])
rectify: Layer = (relu_p, relu_a)

# 0 dimension linear. weight and bias are scalars
def lin0_p(w_b: Parameter, t: Tensor) -> Tensor:
  return w_b[0] * t + w_b[1]
def lin0_a(w_b: Parameter, adj: Tensor, t: Tensor) -> Tuple[Tensor, Parameter]:
  return (w_b[0] * adj,
          [(adj * t).sum(), adj.sum()]) # np.sum() cause type error. why?
linear0: Layer = (lin0_p, lin0_a)

# only work if ndim(t)=3
def lin1_p(w_b: Parameter, t: Tensor) -> Tensor:
  return np.expand_dims(w_b[0], axis=(1,2)) * t + np.expand_dims(w_b[1], axis=(1,2))
def lin1_a(w_b: Parameter, adj: Tensor, t: Tensor) -> Tuple[Tensor, Parameter]:
  return (np.expand_dims(w_b[0], axis=(1,2)) * adj,
          [np.sum(adj * t, axis=(1,2)), np.sum(adj, axis=(1,2))])
linear1: Layer = (lin1_p, lin1_a)

def lin_p(w_b: Parameter, t: Tensor) -> Tensor:
  return w_b[0] * t + w_b[1]
def lin_a(w_b: Parameter, adj: Tensor, t: Tensor) -> Tuple[Tensor, Parameter]:
  return (w_b[0] * adj,
          [adj * t, adj])
linear: Layer = (lin_p, lin_a)

# mean pooling
# multiple channels (features)
# only works if kn | n and km | m
def mean_pool_of(kn: int, km: int) -> Layer:
  def mp_p(_, ms: Tensor) -> Tensor:
    d, n, m = np.shape(ms)
    return np.mean(np.reshape(ms, [d, n // kn, kn, m // km, km]), axis=(2, 4))
  def mp_a(_, adj: Tensor, ms: Tensor) -> Tuple[Tensor, Parameter]:
    return (np.repeat(np.repeat(adj, kn, axis=1), km, axis=2) / (kn * km), [])
  return (mp_p, mp_a)

def max_pool_of(kn: int, km: int) -> Layer:
  def mp_p(_, ms: Tensor) -> Tensor:
    d, n, m = np.shape(ms)
    return np.max(np.reshape(ms, [d, n // kn, kn, m // km, km]), axis=(2, 4))
  def mp_a(_, adj: Tensor, ms: Tensor) -> Tuple[Tensor, Parameter]:
    d, n, m = np.shape(ms)
    mx = np.max(np.reshape(ms, [d, n // kn, kn, m // km, km]), axis=(2, 4))
    return (np.repeat(np.repeat(adj, kn, axis=1), km, axis=2) *
            (np.repeat(np.repeat(mx, kn, axis=1), km, axis=2) == ms), [])
  return (mp_p, mp_a)

def repeate_of(kn: int, km: int) -> Layer:
  def rp_p(_, ms: Tensor) -> Tensor:
    return np.repeat(np.repeat(ms, kn, axis=1), km, axis=2)
  def rp_a(_, adj: Tensor, ms: Tensor) -> Tuple[Tensor, Parameter]:
    d, n, m = np.shape(adj)
    return (np.sum(np.reshape(adj, [d, n // kn, kn, m // km, km]), axis=(2, 4)), [])
  return (rp_p, rp_a)


# 2 dimensional convolution without padding
# a bank of filters, multiple channels
# loop here make it slow. but in scipy there's only: 1. non-extended
#   (not vectorized) 2d correlation 2. n-d correlation. both are not suited.
#   it's possible to do it in pure numpy functions but it's too tricky (for me).
def convolve_of(sn: int, sm: int) -> Layer:
  if sn == 1 and sm == 1:
    def cv_p(w_bs: Parameter, ms: Tensor) -> Tensor:
      ws = w_bs[0]
      bs = w_bs[1]
      b, _, kn, km = np.shape(ws)
      d, n, m = np.shape(ms)
      out = np.empty([b, n - kn + 1, m - km + 1], dtype=Real)
      for i in range(b):
        out[i] = (correlate(ms, ws[i], mode='valid', method='fft') + bs[i])
      return out
    def cv_a(w_bs: Parameter, adj: Tensor, ms: Tensor) -> Tuple[Tensor, Parameter]:
      ws = w_bs[0]
      b = np.shape(ws)[0]
      a = np.zeros_like(ms)
      for i in range(b):
        a += convolve(np.expand_dims(adj[i], axis=0), ws[i], method='fft')
      wg = np.empty_like(ws)
      for i in range(b):
        wg[i] = correlate(ms, np.expand_dims(adj[i], axis=0), mode='valid', method='fft')
      return (a,
              [wg, np.sum(adj, axis=(1,2))])
    return (cv_p, cv_a)

  def cv_p(w_bs: Parameter, ms: Tensor) -> Tensor:
    ws = w_bs[0]
    bs = w_bs[1]
    b, _, kn, km = np.shape(ws)
    d, n, m = np.shape(ms)
    out = np.empty([b, (n - kn) // sn + 1, (m - km) // sm + 1], dtype=Real)
    for i in range(b):
      out[i] = (correlate(ms, ws[i], mode='valid', method='fft') + bs[i])[0, ::sn, ::sm]
    return out
  def cv_a(w_bs: Parameter, adj: Tensor, ms: Tensor) -> Tuple[Tensor, Parameter]:
    ws = w_bs[0]
    b, _, kn, km = np.shape(ws)
    d, n, m = np.shape(ms)
    sadj = np.zeros([b, n - kn + 1, m - km + 1])
    sadj[:, ::sn, ::sm] = adj
    a = np.zeros_like(ms)
    for i in range(b):
      a += convolve(np.expand_dims(sadj[i], axis=0), ws[i], method='fft')
    wg = np.empty_like(ws)
    for i in range(b):
      wg[i] = correlate(ms, np.expand_dims(sadj[i], axis=0), mode='valid', method='fft')
    return (a,
            [wg, np.sum(adj, axis=(1,2))])
  return (cv_p, cv_a)


# a very, very ad-hoc layer. mostly because correlation is slow
# well, to be honest, all the layers are highly ad-hoc.
def flt_p(w_bs: Parameter, ms: Tensor) -> Tensor:
  return np.sum(w_bs[0] * ms, axis=(1, 2, 3)) + w_bs[1]
def flt_a(w_bs: Parameter, adj: Tensor, ms: Tensor) -> Tuple[Tensor, Parameter]:
  adjm = np.expand_dims(adj, axis=(1,2,3))
  return (np.sum(adjm * w_bs[0], axis=0),
          [adjm * ms, adj])
flatten: Layer = (flt_p, flt_a)

def sm_p(_, v: Tensor) -> Tensor:
  ev = np.exp(v)
  return ev / np.sum(ev)
def sm_a(_, adj: Tensor, v: Tensor) -> Tuple[Tensor, Parameter]:
  ev = np.exp(v)
  s = np.sum(ev)
  return (np.dot((np.diagflat(s * ev) -
                  np.dot(np.expand_dims(ev, axis=1), np.expand_dims(ev, axis=0))) / (s * s),
                 adj), [])
softmax: Layer = (sm_p, sm_a)

# add zeros
# [1, 1      [1, 0, 1
#  1, 1]  ->  0, 0, 0
#             1, 0, 1 ]
def dilate_of(nn: int, mm: int, bn: int, bm: int, sn: int, sm: int) -> Layer:
  def dl_p(_, ms: Tensor) -> Tensor:
    d, n, m = np.shape(ms)
    sp = np.zeros([d, nn, mm], dtype=Real)
    sp[:, bn:bn+n*sn:sn, bm:bm+m*sm:sm] = ms
    return sp
  def dl_a(_, adj: Tensor, ms: Tensor) -> Tuple[Tensor, Parameter]:
    d, n, m = np.shape(ms)
    a = np.empty_like(ms)
    a = adj[:, bn:bn+n*sn:sn, bm:bm+m*sm:sm]
    return (a,
            [])
  return (dl_p, dl_a)

def reshape_of(s: list[int]) -> Layer:
  def rs_p(_, t: Tensor) -> Tensor:
    return np.reshape(t, s)
  def rs_a(_, adj: Tensor, t: Tensor) -> Tuple[Tensor, Parameter]:
    return (np.reshape(adj, np.shape(t)), [])
  return (rs_p, rs_a)

# reparameterization (only for my need)
# input is like [μ_1, ... ,μ_n, 2 * log σ_1, ... , 2 * log σ_n]
# gradient is respect to last sample sampled (reasonable?).
__sample = 0
def rp_p(_, m_v: Tensor) -> Tensor:
  global __sample
  n = len(m_v) // 2
  mean = m_v[:n]
  std = np.exp(m_v[n:] / 2)
  __sample = np.random.multivariate_normal(np.zeros([n]), np.diag(np.ones([n])))
  return mean + std * __sample
def rp_a(_, adj: Tensor, m_v: Tensor) -> Tuple[Tensor, Parameter]:
  n = len(adj)
  a = np.empty([n * 2], dtype=Real)
  logvar = m_v[n:]
  a[:n] = adj
  a[n:] = adj * __sample * np.exp(logvar / 2)
  return (a,
          []) # not needed
reparam: Layer = (rp_p, rp_a)

# batch normalization.


# l2 loss.
def l2_loss(g: Tensor, t: Tensor) -> Tuple[Real, Tensor]:
  d = g - t
  return ((d * d).sum(),
          2 * d)

def cross_entropy(g: Tensor, t: Tensor) -> Tuple[Real, Tensor]:
  return (-np.sum(np.log(g) * t),
          -t / g)

# KL distance with standarn Gaussian distribution
# input is like [μ_1, ... ,μ_n, 2 * log σ_1, ... , 2 * log σ_n]
def kl_std(m_v: Tensor, _) -> Tuple[Real, Tensor]:
  n = len(m_v) // 2
  m = m_v[:n]
  logv = m_v[n:]
  v = np.exp(logv)
  a = np.empty(2 * n, dtype=Real)
  a[:n] = m
  a[n:] = (v - 1) / 2
  return ((v.sum() + np.dot(m, m) - n - logv.sum()),
          a)


# Initialization
def He_init_filter(s: list[int]) -> Tensor:
  return np.random.normal(0,
                          2 / np.sqrt(2 * s[1] * s[2] * s[3]),
                          s).astype(dtype=Real)

def Xavier_init_filter(s: list[int]) -> Tensor:
  return np.random.normal(0,
                          1 / np.sqrt(s[1] * s[2] * s[3]),
                          s).astype(dtype=Real)

def He_init_mat(s: list[int]) -> Tensor:
  return np.random.normal(0,
                          2 / np.sqrt(2 * s[1]),
                          s).astype(dtype=Real)

def Xavier_init_mat(s: list[int]) -> Tensor:
  return np.random.normal(0,
                          1 / np.sqrt(s[1]),
                          s).astype(dtype=Real)


### Some helper functions ###

def prim_of(l: Layer) -> PrimFn:
  return l[0]
def adjoint_of(l: Layer) -> AdjFn:
  return l[1]
def fn_of(n: Network) -> list[Layer]:
  return n[0]
def param_of(n: Network) -> list[Parameter]:
  return n[1]
def stack(f: Network, s: Network) -> Network:
  return (fn_of(f) + fn_of(s),
          param_of(f) + param_of(s))
