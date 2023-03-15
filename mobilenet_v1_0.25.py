# mobilenet v1 implemented in RISC-V assembly code

import numpy as np
import keras as kr
from keras.layers import Conv2D, DepthwiseConv2D
from keras.initializers import constant
from machine import machine
# alternatively, uncomment the line below to use the tinyfive package
#   from tinyfive.machine import machine
from layers import *

np.random.seed(5)  # fix seed for reproducible results
m = machine(mem_size=8000000)  # instantiate RISC-V machine with 8MB of memory
# TODO: reduce to 500KB once we use branches to reduce image size

#-------------------------------------------------------------------------------
# run inference (golden reference)
#-------------------------------------------------------------------------------

# mobilenet v1 with 0.25 depth multiplier:
#
# Layer Type         Stride  In-chan  Out-chan  In-res  Out-res
# ----------------------------------------------------------------
# 1     Conv 3x3      2      3        8         96       48
# 2     Conv DW 3x3   1      8        8         48       48
# 3     Conv 1x1      1      8        16        48       48
# 4     Conv DW 3x3   2      16       16        48       24
# 5     Conv 1x1      1      16       32        24       24
# 6     Conv DW 3x3   1      32       32        24       24
# 7     Conv 1x1      1      32       32        24       24
# 8     Conv DW 3x3   2      32       32        24       12
# 9     Conv 1x1      1      32       64        12       12
# 10    Conv DW 3x3   1      64       64        12       12
# 11    Conv 1x1      1      64       64        12       12
# 12    Conv DW 3x3   2      64       64        12       6
# 13    Conv 1x1      1      64       128       6        6
# 14    Conv DW 3x3   1      128      128       6        6
# 15    Conv 1x1      1      128      128       6        6
# 16    Conv DW 3x3   1      128      128       6        6
# 17    Conv 1x1      1      128      128       6        6
# 18    Conv DW 3x3   1      128      128       6        6
# 19    Conv 1x1      1      128      128       6        6
# 20    Conv DW 3x3   1      128      128       6        6
# 21    Conv 1x1      1      128      128       6        6
# 22    Conv DW 3x3   1      128      128       6        6
# 23    Conv 1x1      1      128      128       6        6
# 24    Conv DW 3x3   2      128      128       6        3
# 25    Conv 1x1      1      128      256       3        3
# 26    Conv DW 3x3   1      256      256       3        3
# 27    Conv 1x1      1      256      256       3        3
# 28    avg pool      1      256      256       3        1
# 29    FC layer      1      256      2         1        1

# generate random input activations and weights
inp = np.random.normal(size=(1, 96, 96,   3)).astype(np.float32)
w1  = np.random.normal(size=(3, 3,   3,   8)).astype(np.float32)
w2  = np.random.normal(size=(3, 3,        8)).astype(np.float32)
w3  = np.random.normal(size=(1, 1,   8,  16)).astype(np.float32)
w4  = np.random.normal(size=(3, 3,       16)).astype(np.float32)
w5  = np.random.normal(size=(1, 1,  16,  32)).astype(np.float32)
w6  = np.random.normal(size=(3, 3,       32)).astype(np.float32)
w7  = np.random.normal(size=(1, 1,  32,  32)).astype(np.float32)
w8  = np.random.normal(size=(3, 3,       32)).astype(np.float32)
w9  = np.random.normal(size=(1, 1,  32,  64)).astype(np.float32)
w10 = np.random.normal(size=(3, 3,       64)).astype(np.float32)
w11 = np.random.normal(size=(1, 1,  64,  64)).astype(np.float32)
w12 = np.random.normal(size=(3, 3,       64)).astype(np.float32)
w13 = np.random.normal(size=(1, 1,  64, 128)).astype(np.float32)
w14 = np.random.normal(size=(3, 3,      128)).astype(np.float32)
w15 = np.random.normal(size=(1, 1, 128, 128)).astype(np.float32)
# skip layers 16-23 because they are the same as layers 14-15; TODO: add layers16-23
w24 = np.random.normal(size=(3, 3,      128)).astype(np.float32)
w25 = np.random.normal(size=(1, 1, 128, 256)).astype(np.float32)
w26 = np.random.normal(size=(3, 3,      256)).astype(np.float32)
w27 = np.random.normal(size=(1, 1, 256, 256)).astype(np.float32)

l1  = Conv2D(      8, 3, strides=2, padding='same',    kernel_initializer=constant( w1))(inp)
l2  = DepthwiseConv2D(3,            padding='same', depthwise_initializer=constant( w2))( l1)
l3  = Conv2D(         16, 1,                           kernel_initializer=constant( w3))( l2)
l4  = DepthwiseConv2D(3, strides=2, padding='same', depthwise_initializer=constant( w4))( l3)
l5  = Conv2D(         32, 1,                           kernel_initializer=constant( w5))( l4)
l6  = DepthwiseConv2D(3,            padding='same', depthwise_initializer=constant( w6))( l5)
l7  = Conv2D(         32, 1,                           kernel_initializer=constant( w7))( l6)
l8  = DepthwiseConv2D(3, strides=2, padding='same', depthwise_initializer=constant( w8))( l7)
l9  = Conv2D(         64, 1,                           kernel_initializer=constant( w9))( l8)
l10 = DepthwiseConv2D(3,            padding='same', depthwise_initializer=constant(w10))( l9)
l11 = Conv2D(         64, 1,                           kernel_initializer=constant(w11))(l10)
l12 = DepthwiseConv2D(3, strides=2, padding='same', depthwise_initializer=constant(w12))(l11)
l13 = Conv2D(         128, 1,                          kernel_initializer=constant(w13))(l12)
l14 = DepthwiseConv2D(3,            padding='same', depthwise_initializer=constant(w14))(l13)
l15 = Conv2D(         128, 1,                          kernel_initializer=constant(w15))(l14)
# skip layers 16-23 because they are the same as layers 14-15; TODO: add layers16-23
l24 = DepthwiseConv2D(3, strides=2, padding='same', depthwise_initializer=constant(w24))(l15)
l25 = Conv2D(         256, 1,                          kernel_initializer=constant(w25))(l24)
l26 = DepthwiseConv2D(3,            padding='same', depthwise_initializer=constant(w26))(l25)
l27 = Conv2D(         256, 1,                          kernel_initializer=constant(w27))(l26)

# TODOs:
#  - replace above by keras sequence (or import a reference model from HuggingFace or Keras)
#  - use for-loops (for i range(30)) for these 29 layers to clean up the code
#  - add the remaining layers 15 to 29
#  - add ReLU
#  - add biases

#-------------------------------------------------------------------------------
# store activations and weights in memory
#-------------------------------------------------------------------------------
w1_base  = 0
w2_base  = w1.size * 4
w3_base  = w2.size * 4  + w2_base
w4_base  = w3.size * 4  + w3_base
w5_base  = w4.size * 4  + w4_base
w6_base  = w5.size * 4  + w5_base
w7_base  = w6.size * 4  + w6_base
w8_base  = w7.size * 4  + w7_base
w9_base  = w8.size * 4  + w8_base
w10_base = w9.size * 4  + w9_base
w11_base = w10.size * 4 + w10_base
w12_base = w11.size * 4 + w11_base
w13_base = w12.size * 4 + w12_base
w14_base = w13.size * 4 + w13_base
w15_base = w14.size * 4 + w14_base
# skip layers 16-23 because they are the same as layers 14-15; TODO: add layers16-23
w24_base = w15.size * 4 + w15_base
w25_base = w24.size * 4 + w24_base
w26_base = w25.size * 4 + w25_base
w27_base = w26.size * 4 + w26_base

# TODO: optimize the memory footprint of the activations
a1_base  = w27.size * 4         + w27_base
a2_base  = inp.size * 4         + a1_base
a3_base  = l1.numpy().size * 4  + a2_base
a4_base  = l2.numpy().size * 4  + a3_base
a5_base  = l3.numpy().size * 4  + a4_base
a6_base  = l4.numpy().size * 4  + a5_base
a7_base  = l5.numpy().size * 4  + a6_base
a8_base  = l6.numpy().size * 4  + a7_base
a9_base  = l7.numpy().size * 4  + a8_base
a10_base = l8.numpy().size * 4  + a9_base
a11_base = l9.numpy().size * 4  + a10_base
a12_base = l10.numpy().size * 4 + a11_base
a13_base = l11.numpy().size * 4 + a12_base
a14_base = l12.numpy().size * 4 + a13_base
a15_base = l13.numpy().size * 4 + a14_base
# skip layers 16-23 because they are the same as layers 14-15; TODO: add layers16-23
a24_base = l14.numpy().size * 4 + a15_base
a16_base = a24_base  # TODO: remove eventually
a25_base = l15.numpy().size * 4 + a24_base
a26_base = l24.numpy().size * 4 + a25_base
a27_base = l25.numpy().size * 4 + a26_base
out_base = l26.numpy().size * 4 + a27_base
a28_base = out_base  # TODO: remove eventually
code_start = l27.numpy().size * 4 + out_base

# TODO: clean up, minimize memory size
a16_base = a1_base
a24_base = a1_base
a25_base = a2_base
a26_base = a3_base
a27_base = a4_base
a28_base = a5_base

m.write_f32_vec(np.transpose( w1, axes=[3, 0, 1, 2]).flatten(), w1_base)
m.write_f32_vec(np.transpose( w2, axes=[2, 0, 1]).flatten(), w2_base)
m.write_f32_vec(              w3.flatten(),                  w3_base)
m.write_f32_vec(np.transpose( w4, axes=[2, 0, 1]).flatten(), w4_base)
m.write_f32_vec(              w5.flatten(),                  w5_base)
m.write_f32_vec(np.transpose( w6, axes=[2, 0, 1]).flatten(), w6_base)
m.write_f32_vec(              w7.flatten(),                  w7_base)
m.write_f32_vec(np.transpose( w8, axes=[2, 0, 1]).flatten(), w8_base)
m.write_f32_vec(              w9.flatten(),                  w9_base)
m.write_f32_vec(np.transpose(w10, axes=[2, 0, 1]).flatten(), w10_base)
m.write_f32_vec(             w11.flatten(),                  w11_base)
m.write_f32_vec(np.transpose(w12, axes=[2, 0, 1]).flatten(), w12_base)
m.write_f32_vec(             w13.flatten(),                  w13_base)
m.write_f32_vec(np.transpose(w14, axes=[2, 0, 1]).flatten(), w14_base)
m.write_f32_vec(             w15.flatten(),                  w15_base)
# skip layers 16-23 because they are the same as layers 14-15; TODO: add layers16-23
m.write_f32_vec(np.transpose(w24, axes=[2, 0, 1]).flatten(), w24_base)
m.write_f32_vec(             w25.flatten(),                  w25_base)
m.write_f32_vec(np.transpose(w26, axes=[2, 0, 1]).flatten(), w26_base)
m.write_f32_vec(             w27.flatten(),                  w27_base)

m.write_f32_vec(inp.flatten(), a1_base)

#-------------------------------------------------------------------------------
# run assembly code and compare with keras
#-------------------------------------------------------------------------------
def compare_cpu_vs_ref(m, C, R, y_base, ref, trans=False):
  """compare CPU machine versus reference (Keras, PyTorch)"""
  if trans==False:
    cpu = m.read_f32_vec(y_base, size=R*R*C).reshape(R, R, C)
  else:
    cpu = np.transpose(m.read_f32_vec(y_base, size=R*R*C).reshape(C, R, R), axes=[1, 2, 0])
  m.print_rel_err(cpu, ref.numpy().reshape(R, R, C))

#-------------------------------------------------------------------------------
# layers 1, 2, 3
conv_3x3x3_stride2( m, 8, 96, a1_base, w1_base, a2_base)
dw_conv_3x3_stride1(m, 8, 48, a2_base, w2_base, a3_base, out_chan_first=False)
conv_1x1(       m, 8, 16, 48, a3_base, w3_base, a4_base, code_start)

compare_cpu_vs_ref(m,  8, 48, a2_base, l1, trans=True)
compare_cpu_vs_ref(m,  8, 48, a3_base, l2)
compare_cpu_vs_ref(m, 16, 48, a4_base, l3)

# TODO: remove below hack, temp hack to transpose the input activations
l3_hack = np.transpose(m.read_f32_vec(a4_base, size=48*48*16).reshape(48, 48, 16), axes=[2, 0, 1])
m.write_f32_vec(l3_hack.flatten(), a4_base)

#-------------------------------------------------------------------------------
# layers 4, 5
dw_conv_3x3_stride2(m, 16, 48, a4_base, w4_base, a5_base, out_chan_first=False)
conv_1x1(       m, 16, 32, 24, a5_base, w5_base, a6_base, code_start)

compare_cpu_vs_ref(m, 16, 24, a5_base, l4)
compare_cpu_vs_ref(m, 32, 24, a6_base, l5)

# TODO: remove below hack, temp hack to transpose the input activations
l5_hack = np.transpose(m.read_f32_vec(a6_base, size=24*24*32).reshape(24, 24, 32), axes=[2, 0, 1])
m.write_f32_vec(l5_hack.flatten(), a6_base)

#-------------------------------------------------------------------------------
# layers 6, 7
dw_conv_3x3_stride1(m, 32, 24, a6_base, w6_base, a7_base, out_chan_first=False)
conv_1x1(       m, 32, 32, 24, a7_base, w7_base, a8_base, code_start)

compare_cpu_vs_ref(m, 32, 24, a7_base, l6)
compare_cpu_vs_ref(m, 32, 24, a8_base, l7)

# TODO: remove below hack, temp hack to transpose the input activations
l7_hack = np.transpose(m.read_f32_vec(a8_base, size=24*24*32).reshape(24, 24, 32), axes=[2, 0, 1])
m.write_f32_vec(l7_hack.flatten(), a8_base)

#-------------------------------------------------------------------------------
# layers 8-15
dw_conv_3x3_stride2(m,  32, 24,  a8_base,  w8_base,  a9_base, out_chan_first=False)
conv_1x1(m,        32,  64, 12,  a9_base,  w9_base, a10_base, code_start, trans=True)
dw_conv_3x3_stride1(m,  64, 12, a10_base, w10_base, a11_base, out_chan_first=False)
conv_1x1(m,        64,  64, 12, a11_base, w11_base, a12_base, code_start, trans=True)
dw_conv_3x3_stride2(m,  64, 12, a12_base, w12_base, a13_base, out_chan_first=False)
conv_1x1(m,        64, 128,  6, a13_base, w13_base, a14_base, code_start, trans=True)
dw_conv_3x3_stride1(m, 128,  6, a14_base, w14_base, a15_base, out_chan_first=False)
conv_1x1(m,       128, 128,  6, a15_base, w15_base, a16_base, code_start, trans=True)

compare_cpu_vs_ref(m,  32, 12,  a9_base,  l8)
compare_cpu_vs_ref(m,  64, 12, a10_base,  l9, trans=True)
compare_cpu_vs_ref(m,  64, 12, a11_base, l10)
compare_cpu_vs_ref(m,  64, 12, a12_base, l11, trans=True)
compare_cpu_vs_ref(m,  64,  6, a13_base, l12)
compare_cpu_vs_ref(m, 128,  6, a14_base, l13, trans=True)
compare_cpu_vs_ref(m, 128,  6, a15_base, l14)
compare_cpu_vs_ref(m, 128,  6, a16_base, l15, trans=True)

#-------------------------------------------------------------------------------
# skip layers 16-23 because they are identical to layers 14-15;
# TODO: add layers 16-23 here

#-------------------------------------------------------------------------------
# layers 24-27
dw_conv_3x3_stride2(m, 128, 6, a24_base, w24_base, a25_base, out_chan_first=False)
conv_1x1_big(m,   128, 256, 3, a25_base, w25_base, a26_base, code_start, S=3, trans=True)
dw_conv_3x3_stride1(m, 256, 3, a26_base, w26_base, a27_base, out_chan_first=False)
conv_1x1_big(m,   256, 256, 3, a27_base, w27_base, a28_base, code_start, S=3)

compare_cpu_vs_ref(m, 128, 3, a25_base, l24)
compare_cpu_vs_ref(m, 256, 3, a26_base, l25, trans=True)
compare_cpu_vs_ref(m, 256, 3, a27_base, l26)
compare_cpu_vs_ref(m, 256, 3, a28_base, l27)
