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
m = machine(mem_size=6000000)  # instantiate RISC-V machine with 6MB of memory
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
#  - use for-loops (for i range(30)) for these 29 layers to clean up the code
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
a1_base  = w14.size * 4         + w14_base
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
# TODO: add more layers and update below
out_base   = l13.numpy().size * 4 + a14_base
code_start = l14.numpy().size * 4 + out_base

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

conv_3x3x3_stride2(m, F, R, a_base=a1_base, w_base=w1_base, y_base=a2_base)

# compare results against keras
l1_asm = np.transpose(m.read_f32_vec(y_base, size=Q*Q*8).reshape(F, Q, Q), axes=[1, 2, 0])
l1_ref = l1.numpy().reshape(Q, Q, F)
m.print_rel_err(l1_asm, l1_ref)

#-------------------------------------------------------------------------------
# layer 2: Depthwise Conv2D 3x3, 8 channels, 48x48 resolution, stride=1
#-------------------------------------------------------------------------------
R = 48
C = 8
y_base = a3_base

dw_conv_3x3_stride1(m, C, R, a_base=a2_base, w_base=w2_base, y_base=y_base,
                    out_chan_first=False)
# Note: set out_chan_first=False so that output is shape (R, R, C)

# compare results against keras
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

conv_1x1(m, C, F, R, a_base=a3_base, w_base=w3_base, y_base=y_base,
         code_start=code_start)

# compare results against keras
l3_asm = m.read_f32_vec(y_base, size=R*R*F).reshape(R, R, F)
l3_ref = l3.numpy().reshape(R, R, F)
m.print_rel_err(l3_asm, l3_ref)

#-------------------------------------------------------------------------------
# layer 4: Depthwise Conv2D 3x3, 16 channels, 48x48 resolution, stride=2
#-------------------------------------------------------------------------------
R = 48
Q = R//2
C = 16
y_base = a5_base

# TODO: remove below hack, temp hack to transpose the input activations
l3_hack = np.transpose(l3_asm, axes=[2, 0, 1])
m.write_f32_vec(l3_hack.flatten(), a4_base)

dw_conv_3x3_stride2(m, C, R, a_base=a4_base, w_base=w4_base, y_base=y_base,
                    out_chan_first=False)

# compare results against keras
l4_asm = m.read_f32_vec(y_base, size=Q*Q*C).reshape(Q, Q, C)
l4_ref = l4.numpy().reshape(Q, Q, C)
m.print_rel_err(l4_asm, l4_ref)

#-------------------------------------------------------------------------------
# layer 5: Conv2D 1x1, 16 in-channels, 32 out-channels, 24x24 resolution
#-------------------------------------------------------------------------------
R = 24
C = 16
F = 2*C
y_base = a6_base

conv_1x1(m, C, F, R, a_base=a5_base, w_base=w5_base, y_base=y_base,
         code_start=code_start)

# compare results against keras
l5_asm = m.read_f32_vec(y_base, size=R*R*F).reshape(R, R, F)
l5_ref = l5.numpy().reshape(R, R, F)
m.print_rel_err(l5_asm, l5_ref)

#-------------------------------------------------------------------------------
# layer 6: Depthwise Conv2D 3x3, 32 channels, 24x24 resolution, stride=1
#-------------------------------------------------------------------------------
R = 24
C = 32
y_base = a7_base

# TODO: remove below hack, temp hack to transpose the input activations
l5_hack = np.transpose(l5_asm, axes=[2, 0, 1])
m.write_f32_vec(l5_hack.flatten(), a6_base)

dw_conv_3x3_stride1(m, C, R, a_base=a6_base, w_base=w6_base, y_base=y_base,
                    out_chan_first=False)

# compare results against keras
l6_asm = m.read_f32_vec(y_base, size=R*R*C).reshape(R, R, C)
l6_ref = l6.numpy().reshape(R, R, C)
m.print_rel_err(l6_asm, l6_ref)
