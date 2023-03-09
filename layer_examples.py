# These examples show how to implement neural networks with RISC-V assembly code

import numpy as np
import keras as kr
from keras.layers import Conv2D, DepthwiseConv2D
from keras.initializers import constant
from machine import machine
# alternatively, uncomment the line below to use the tinyfive package
#   from tinyfive.machine import machine
from layers import *

np.random.seed(5)  # fix seed for reproducible results
m = machine(mem_size=2000000)  # instantiate RISC-V machine with 2MB of memory
# TODO: reduce to 500KB once we use branches to reduce image size

# abbreviations for shape dimensions:
#   C : input channels (and output channels if the same as input channels)
#   F : output channels (or filters), only used if F is not the same as C
#   R : input resolution (and output resolution if the same as input).
#   Q : output resolution, only used if Q is not the same as R

#-------------------------------------------------------------------------------
# Example 1: 4x4 Dense layer
#-------------------------------------------------------------------------------
print('-------------- Example 1: Dense layer ---------------------------------')
# This is a very small dense layer example using floating point

# generate 4x4 matrices A and B (float32) and store them in memory
a = np.random.normal(size=(4, 4)).astype(np.float32)
b = np.random.normal(size=(4, 4)).astype(np.float32)
m.write_f32_vec(a.flatten(), 0)     # write matrix A to mem[0]
m.write_f32_vec(b.flatten(), 4*16)  # write matrix B to mem[4*16]

# TODO: merge this with conv_1x1 or parameterize it, move into def

# store assembly program starting at address 4*128
m.pc = 4*128
m.lbl('start')
# load the entire B matrix into registers f[16] ... f[31]
for i in range(4):
  for j in range(4):
    m.asm('flw.s', 16+4*i+j, 4*(16+4*i+j), 0)
# perform matmul in row-major order
for i in range(4):
  for k in range(4):                    # load f[10] ... f[13] with row i of A
    m.asm('flw.s', 10+k, 4*(4*i+k), 0)  # load f[10+k] with A[i, k]
  for j in range(4):
    m.asm('fmul.s', 15, 10, 16+j)       # f[15] = f[10] * f[16+j] = A[i, 0] * B[0, j]
    for k in range(1, 4):
      m.asm('fmadd.s', 15, 10+k, 16+4*k+j, 15)  # f[15] += A[i, k] * B[k, j]
    m.asm('fsw.s', 15, 4*(32+i*4+j), 0) # store res[i, j] from f[15]
m.lbl('end')

# execute program from 'start' to 'end'
m.exe(start='start', end='end')
m.print_perf()

# compare results against np.matmul(A, B)
res = m.read_f32_vec(4*32, size=4*4).reshape(4, 4)  # read result matrix
m.print_rel_err(res, np.matmul(a, b))
# Output: should be very small, e.g. smaller than 1e-06, but could be larger

#-------------------------------------------------------------------------------
# Example 2: Conv2D 1x1, 32 input and 32 output channels, 6x6 image
#-------------------------------------------------------------------------------
print('-------------- Example 2: Conv2D 1x1 layer ----------------------------')
C = 32  # input-channels (tested it up to 128)
F = C   # output-channels (tested it up to 128)
R = 6   # resolution of image (tested it up to 48 with C = 8)

#-------------------------------------------------------------------------------
# generate activations and weights for keras (suffix *k is for keras)
a_k = np.random.normal(size=(1, R, R, C)).astype(np.float32)
w_k = np.random.normal(size=(1, 1, C, F)).astype(np.float32)
  # input shape:  (1, R, R, C) : batch-size, 4x4 image, channels
  # output shape: (1, R, R, F) : batch-size, 4x4 image, channels
  # kernel shape: (1, 1, C, F) : 1x1 kernel, in-channels, out-channels

# run inference with keras (golden reference)
y_k = Conv2D(F, 1, kernel_initializer=constant(w_k))(a_k)

# TODO: use below if you want to use bias and ReLU activation
#  layer = Conv2D(128, 1, activation="relu", name="layer1", input_shape=(4, 4, 128),
#          kernel_initializer=constant(w_k),
#          bias_initializer=constant(b_k))
# Instead of using kr.initializer, you could use set_weights() as follows:
#   y_k = layer(a_k)  # dummy run with random weights, needed before set_weights()
#   layer.set_weights([w_k, b_k])
#   y_k = layer(a_k)
#   print(layer.get_weights()[0].shape)  # print weights
#   print(layer.get_weights()[1].shape)  # print biases

#-------------------------------------------------------------------------------
# flatten keras tensors, compare with matmul
a = a_k.reshape(R*R, C)  # a_k (1, R, R, C)  -> A (R*R, C)
w = w_k.reshape(C, F)    # w_k (1, 1, C, F) -> W (C, F)
y = y_k.numpy().reshape(R*R, F)
m.print_rel_err(np.matmul(a, w), y)  # compare matmul vs. keras conv2D

#-------------------------------------------------------------------------------
# proof of concept: split W and A into 4x4 submatrices, then compute matmul(A, W)
a_split = np.empty((R*R//4, C//4, 4, 4))
w_split = np.empty((C//4, F//4, 4, 4))
for i in range(C//4):
  for j in range(F//4):
    w_split[i, j] = w[i*4:i*4+4, j*4:j*4+4]
for i in range(R*R//4):
  for j in range(C//4):
    a_split[i, j] = a[i*4:i*4+4, j*4:j*4+4]
# compute the big matmul by smaller 4x4 matmuls
y_con = np.zeros((R*R, F))
for i in range(R*R//4):
  for j in range(F//4):
    for k in range(C//4):
      y_con[4*i:4*i+4, 4*j:4*j+4] += np.matmul(a_split[i, k], w_split[k, j])
m.print_rel_err(y_con, y)  # compare y_con against Y

#-------------------------------------------------------------------------------
# run assembly and compare
m.clear_mem()
m.clear_cpu()

# write A and W to memory
a_base = 0
w_base = a_base + R*R*C * 4
y_base = w_base + C*F * 4
code_start = y_base + R*R*F * 4
m.write_f32_vec(a.flatten(), a_base)  # write A to mem[a_base]
m.write_f32_vec(w.flatten(), w_base)  # write W to mem[w_base]

# run assembly
conv_1x1(m, C, F, R, a_base, w_base, y_base, code_start)
m.print_perf()

# compare results against expected Y
y_asm = m.read_f32_vec(y_base, size=R*R*F).reshape(R*R, F)  # read result matrix
m.print_rel_err(y_asm, y)

#-------------------------------------------------------------------------------
# Example 3: Depthwise Conv2D 3x3 with 4 channels, stride=1,2, 6x6 image
#-------------------------------------------------------------------------------
print('-------------- Example 3: Depthwise Conv2D 3x3 layer, stride=1,2 ------')
C = 4  # channels
R = 6  # resolution
Q = R//2  # output resolution for stride 2 only

#-------------------------------------------------------------------------------
# generate activations and weights, run inference
a = np.random.normal(size=(1, R, R, C)).astype(np.float32)
w = np.random.normal(size=(3, 3, C)).astype(np.float32)
# activation shape: (1, R, R, C) : batch-size, RxR image, channels
# output shape for stride=2: (1, Q, Q, C) : batch-size, QxQ image, channels
# kernel shape: (3, 3, C) : 3x3 kernel, channels

# run inference with keras (golden reference) for strides 1 and 2:
# y1 refers to stride=1; y2 refers to stride=2
y1_k = DepthwiseConv2D(3, padding='same', depthwise_initializer=constant(w))(a)
y2_k = DepthwiseConv2D(3, padding='same', strides=2, depthwise_initializer=constant(w))(a)
y1 = y1_k.numpy().reshape(R, R, C)  # flatten
y2 = y2_k.numpy().reshape(Q, Q, C)  # flatten

#-------------------------------------------------------------------------------
# run assembly and compare
m.clear_mem()
m.clear_cpu()

# write A and W to memory
a_base  = 0
w_base  = a_base  + R*R*C *4
y1_base = w_base  + 3*3*C *4  # y_base for stride=1
y2_base = y1_base + R*R*C *4  # y_base for stride=2

m.write_f32_vec(np.transpose(a, axes=[3, 0, 1, 2]).flatten(), a_base)
m.write_f32_vec(np.transpose(w, axes=[2, 0, 1]).flatten(), w_base)
# note on the transpose in the last two lines: We rearrange the matrices so
# that the last axis (channel) is now the first axis (aka 'channel-first order').
# That's important so that when we flatten it in row-major, all the pixels of the
# first channel are contigously in memory, because we process one channel at a time

# run assembly code for strides 1 and 2
dw_conv_3x3_stride1(m, C, R, a_base, w_base, y1_base)
dw_conv_3x3_stride2(m, C, R, a_base, w_base, y2_base)

# compare results against keras
y1_asm = np.transpose(m.read_f32_vec(y1_base, size=R*R*C).reshape(C, R, R), axes=[1, 2, 0])
y2_asm = np.transpose(m.read_f32_vec(y2_base, size=Q*Q*C).reshape(C, Q, Q), axes=[1, 2, 0])
m.print_rel_err(y1_asm, y1)
m.print_rel_err(y2_asm, y2)

# now rerun both cases with 'out_chan_first=False' and compare against previous runs
dw_conv_3x3_stride1(m, C, R, a_base, w_base, y1_base, out_chan_first=False)
dw_conv_3x3_stride2(m, C, R, a_base, w_base, y2_base, out_chan_first=False)
y1_asm_t = m.read_f32_vec(y1_base, size=R*R*C).reshape(R, R, C)
y2_asm_t = m.read_f32_vec(y2_base, size=Q*Q*C).reshape(Q, Q, C)
m.print_rel_err(y1_asm_t, y1_asm)
m.print_rel_err(y2_asm_t, y2_asm)

#-------------------------------------------------------------------------------
# Example 4: Conv2D 3x3, 3 in-channels, 8 out-channels, 12x12 image, stride=1,2
#-------------------------------------------------------------------------------
print('-------------- Example 4: Conv2D 3x3 layer, stride=1,2 ----------------')
F = 8   # output-channels
R = 12  # image resolution
Q = R//2  # output resolution for stride 2 only

#-------------------------------------------------------------------------------
# generate activations and weights, run inference
a = np.random.normal(size=(1, R, R, 3)).astype(np.float32)
w = np.random.normal(size=(3, 3, 3, F)).astype(np.float32)
# input shape:  (1, R, R, 3) : batch-size, RxR image, channels
# kernel shape: (3, 3, 3, F) : 3x3 kernel, in-channels, out-channels
# output shape for stride=1: (1, R, R, F) : batch-size, RxR image, channels
# output shape for stride=2: (1, Q, Q, F) : batch-size, QxQ image, channels

# run inference with keras (golden reference) for strides 1 and 2:
# y1 refers to stride=1; y2 refers to stride=2
y1_k = Conv2D(F, 3, padding='same', kernel_initializer=constant(w))(a)
y2_k = Conv2D(F, 3, padding='same', strides=2, kernel_initializer=constant(w))(a)
y1 = y1_k.numpy().reshape(R, R, F)
y2 = y2_k.numpy().reshape(Q, Q, F)

#-------------------------------------------------------------------------------
# run assembly and compare
m.clear_mem()
m.clear_cpu()

# write to memory
a_base = 0
w_base = a.size * 4
y1_base = (a.size + w.size) * 4  # y_base for stride=1
y2_base = y1_base + F*R*R * 4    # y_base for stride=2
m.write_f32_vec(a.flatten(), a_base)
m.write_f32_vec(np.transpose(w, axes=[3, 0, 1, 2]).flatten(), w_base)
# transpose W so that the output-channels is first axes

# run assembly code for strides 1 and 2
conv_3x3x3_stride1(m, F, R, a_base, w_base, y1_base)
conv_3x3x3_stride2(m, F, R, a_base, w_base, y2_base)

# compare results against expected
y1_asm = np.transpose(m.read_f32_vec(y1_base, size=R*R*F).reshape(F, R, R), axes=[1, 2, 0])
y2_asm = np.transpose(m.read_f32_vec(y2_base, size=Q*Q*F).reshape(F, Q, Q), axes=[1, 2, 0])
m.print_rel_err(y1_asm, y1)
m.print_rel_err(y2_asm, y2)
