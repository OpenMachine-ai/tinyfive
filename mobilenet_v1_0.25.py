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
m = machine(mem_size=4000000)  # instantiate RISC-V machine with 4MB of memory
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

l1 = Conv2D(8, 3, strides=2, padding='same', kernel_initializer=constant(w1))(inp)

l2 = DepthwiseConv2D(3, padding='same', depthwise_initializer=constant(w2))(l1)

l3 = Conv2D(16, 1, kernel_initializer=constant(w3))(l2)

l4 = DepthwiseConv2D(3, strides=2, padding='same', depthwise_initializer=constant(w4))(l3)

l5 = Conv2D(32, 1, kernel_initializer=constant(w5))(l4)

l6 = DepthwiseConv2D(3, padding='same', depthwise_initializer=constant(w6))(l5)

l7 = Conv2D(32, 1, kernel_initializer=constant(w7))(l6)

l8 = DepthwiseConv2D(3, strides=2, padding='same', depthwise_initializer=constant(w8))(l7)

l9 = Conv2D(64, 1, kernel_initializer=constant(w9))(l8)

l10 = DepthwiseConv2D(3, padding='same', depthwise_initializer=constant(w10))(l9)

l11 = Conv2D(64, 1, kernel_initializer=constant(w11))(l10)

l12 = DepthwiseConv2D(3, strides=2, padding='same', depthwise_initializer=constant(w12))(l11)

l13 = Conv2D(128, 1, kernel_initializer=constant(w13))(l12)

l14 = DepthwiseConv2D(3, padding='same', depthwise_initializer=constant(w14))(l13)

print(l14.numpy().shape)

# TODOs:
#  - replace above by keras sequence (or import a reference model from HuggingFace or Keras)
#  - add the remaining layers 15 to 29
#  - add ReLU
#  - add biases

#-------------------------------------------------------------------------------
# store activations and weights in mem
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

# TODO: optimize the memory footprint of the activations
a1_base = w14.size * 4        + w14_base
a2_base = inp.size * 4        + a1_base
a3_base = l1.numpy().size * 4 + a2_base
a4_base = l2.numpy().size * 4 + a3_base
a5_base = l3.numpy().size * 4 + a4_base
# TODO: add more layers and update below
out_base   = l4.numpy().size * 4 + a5_base
code_start = l5.numpy().size * 4 + out_base

m.write_f32_vec(np.transpose(w1, axes=[3, 0, 1, 2]).flatten(), w1_base)

m.write_f32_vec(np.transpose(w2, axes=[2, 0, 1]).flatten(), w2_base)

m.write_f32_vec(w3.flatten(), w3_base)

m.write_f32_vec(inp.flatten(), a1_base)

# abbreviations for shape dimensions:
#   C : input channels (and output channels if the same as input channels)
#   F : output channels (or filters), only used if F is not the same as C
#   R : input resolution (and output resolution if the same as input).
#   Q : output resolution, only used if Q is not the same as R

#-------------------------------------------------------------------------------
# layer 1: Conv2D 3x3, 3 in-channels, 8 out-channels, 96x96 resolution, stride=2
#-------------------------------------------------------------------------------
F = 8
R = 96
Q = R//2
y_base = a2_base

# run assembly
conv_3x3x3_stride2(m, F, R, a_base=a1_base, w_base=w1_base, y_base=a2_base)

# compare results against expected
l1_asm = np.transpose(m.read_f32_vec(y_base, size=Q*Q*8).reshape(F, Q, Q), axes=[1, 2, 0])
l1_ref = l1.numpy().reshape(Q, Q, F)
m.print_rel_err(l1_asm, l1_ref)

#-------------------------------------------------------------------------------
# layer 2: Depthwise Conv2D 3x3 with 8 channels, 48x48 resolution, stride=1
#-------------------------------------------------------------------------------
R = 48
C = 8
y_base = a3_base

# run assembly
dw_conv_3x3_stride1(m, C, R, a_base=a2_base, w_base=w2_base, y_base=y_base,
                    out_chan_first=False)
# Note: set out_chan_first=False so that output is non-transposed shape (R, R, C)

# compare results against expected
l2_asm = m.read_f32_vec(y_base, size=R*R*C).reshape(R, R, C)
l2_ref = l2.numpy().reshape(R, R, C)
m.print_rel_err(l2_asm, l2_ref)

#-------------------------------------------------------------------------------
# layer 3: Conv2D 1x1, 8 in-channels, 16 out-channels, 48x48 resolution
#-------------------------------------------------------------------------------
R = 48
C = 8
F = 2*C
y_base = a4_base

# run assembly
conv_1x1(m, C, F, R, a_base=a3_base, w_base=w3_base, y_base=y_base,
         code_start=code_start)

# compare results against expected
l3_asm = m.read_f32_vec(y_base, size=R*R*F).reshape(R, R, F)
l3_ref = l3.numpy().reshape(R, R, F)
m.print_rel_err(l3_asm, l3_ref)

# TODO: add more layers here
