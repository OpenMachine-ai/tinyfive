# These examples show how to implement neural networks with RISC-V assembly code

import numpy as np
import keras as kr
from keras.layers import Conv2D, DepthwiseConv2D
from keras.initializers import constant
from machine import machine
# alternatively, uncomment the line below to use the tinyfive package
#   from tinyfive.machine import machine

np.random.seed(5)  # fix seed for reproducible results
m = machine(mem_size=2000000)  # instantiate RISC-V machine with 2MB of memory
# TODO: reduce to 500KB once we use branches to reduce image size

# abbreviations for shape dimensions:
#   C : input channels (and output channels if the same as input channels)
#   F : output channels (or filters), only used if F is not the same as C
#   R : input resolution (and output resolution if the same as input).
#   Q : output resolution, only used if Q is not the same as R

#-------------------------------------------------------------------------------
# assembly code routines for some neural network layers
#-------------------------------------------------------------------------------
def conv_1x1(C, F, R, a_base, w_base, y_base):
  """assembly code with for conv2D 1x1 with F in-channels, F out-channels,
  and resolution R.
  Register map:
    x[9]  : 1st base address for A
    x[10] : 2nd base address for A
    x[11] : base address for W
    x[12] : base address for results Y
    f[11] : to store elements of A
    f[12] .. f[15]: 4 registers to store an entire row of W
    f[16] .. f[31]: the 16 outputs res[0, 0] ... res[4, 4]"""

  # store assembly program starting at address 'code_start'
  m.pc = code_start
  m.lbl('start')

  # matmul (R*R, C) x (C, F) -> (R*R, F)
  for i in range(R*R//4):
    m.asm('lui',  9,      m.hi20(a_base + 4*C*4*i))  # m.x[9] =  ...
    m.asm('addi', 9, 9,   m.lo12(a_base + 4*C*4*i))
    m.asm('lui',  12,     m.hi20(y_base + 4*F*4*i))  # m.x[12] = ...
    m.asm('addi', 12, 12, m.lo12(y_base + 4*F*4*i))

    # matmul (4, C) x (C, F) -> (4, F)
    for j in range(F//4):
      # set base address pointers
      m.asm('add', 10, 9, 0) # reset A pointer to x[9]
      m.asm('lui',  11,     m.hi20(w_base + 16*j))  # m.x[11] = w_base + 16*j
      m.asm('addi', 11, 11, m.lo12(w_base + 16*j))

      # matmul (4, C) x (C, 4) -> (4, 4)
      for k in range(C//4):
        # compute one 4x4 matmul (by computing 4 outer products)
        for ii in range(4):
          # load row ii of W into registers f[12] ... f[15]
          for col in range(4):
            m.asm('flw.s', 12+col, 4*(col+F*ii), 11)
          # compute outer-product in row-major order
          for row in range(4):
            m.asm('flw.s', 11, 4*(C*row+ii), 10)  # load f[11] with A[row, ii]
            for col in range(4):
              if ii==0 and k==0:  # no accumulation for the very first products
                m.asm('fmul.s', 16+4*row+col, 11, 12+col)  # f[] = f[11] * f[12]
              else:
                m.asm('fmadd.s', 16+4*row+col, 11, 12+col, 16+4*row+col) # f[] += f[11] * f[12]

        # increment base addresses for A and W
        m.asm('addi', 10, 10, 4*4)  # increment by 16
        m.asm('addi', 11, 11, 4*4*F//2)  # increment by 4*4*F by two times 4*4*F/2
        m.asm('addi', 11, 11, 4*4*F//2)
        # note on the last two lines: 12-bit immediates: -2048 .. +2047. So we increment
        # by 1024 two times to achieve 2048 increment. Alternatively, we could decrement
        # the index (because -2048 is possible in one instruction). Or store the W-matrix
        # in transposed form

      # store results in memory
      for row in range(4):
        for col in range(4):
          m.asm('fsw.s', 16+4*row+col, 4*(row*F+col), 12)
      m.asm('addi', 12, 12, 4*4)  # increment Y pointer by 16
  m.lbl('end')

  # execute program from 'start' to 'end'
  m.exe(start='start', end='end')
  m.print_perf()

  # TODOs:
  #  - replace the outer for-loop (i, j, k) by assembly code with branches to
  #    reduce the image size
  #  - clean up the indexing and base address pointers
  #  - use mnemonics x9 and f9 etc. instead of '9'
  #  - rewrite above using only upper-case instructions to speed up runtime
  #  - parameterize the assembly code and put it in a def so that it can be used
  #    for other shapes, too


def dw_conv_3x3_stride1(C, R, a_base, w_base, y_base):
  """assembly code with upper-case instruction for depthwise conv2D 3x3 with
  C channels, R resolution, stride = 1.
  Register map:
    x[10] : base address for A[chan]
    x[11] : base address for W[chan]
    x[12] : base address for Y[chan]
    f[0] .. f[8]: the 9 weights of a channel, stored in row-major order
    f[9]  : holds the latest activation loaded from memory
    f[10] : accumulation register 'out0' for current output
    f[11] : accumulation register 'out1' for next output
    f[12] : accumulation register 'out2' for next-next output"""

  # init base addresses
  m.LI(10, a_base)
  m.LI(11, w_base)
  m.LI(12, y_base)

  for chan in range(C):
    # load 3x3 weights for channel 'chan'
    for i in range(3):
      for j in range(3):
        m.FLW_S(3*i+j, (3*i + j)*4, 11)  # f[i, j] = W[chan, i, j]

    # compute all outputs (RxR) for channel 'chan'
    for row in range(R):
      for col in range(R):
        # load 3 activations, perform 9 muls, and store 1 output
        dot_start = 0 if row > 0 else 1    # first row is special
        dot_end   = 3 if row < R-1 else 2  # last row is special
        for dot in range(dot_start, dot_end):
          # load one activation from memory
          m.FLW_S(9, (R*(row-1+dot) + col)*4, 10)  # A[chan, row-1+dot, col]

          # compute 3 muls with weights W[dot, 0:3]
          if dot == dot_start:
            if col > 0:
              m.FMADD_S(10, 9, 3*dot+2, 11)  # f10 = f9 * W[dot, 2] + f11
              m.FMADD_S(11, 9, 3*dot+1, 12)  # f11 = f9 * W[dot, 1] + f12
            else:
              m.FMUL_S(11, 9, 3*dot+1)            # f11 = f9 * W[dot, 1]
            if col < R-1: m.FMUL_S(12, 9, 3*dot)  # f12 = f9 * W[dot, 0]
          else:
            m.FMADD_S(11, 9, 3*dot+1, 11)                # f11 += f9 * W[dot, 1]
            if col > 0:   m.FMADD_S(10, 9, 3*dot+2, 10)  # f10 += f9 * W[dot, 2]
            if col < R-1: m.FMADD_S(12, 9, 3*dot, 12)    # f12 += f9 * W[dot, 0]
        # store result
        if col > 0:    m.FSW_S(10, (R*row + col-1)*4, 12)  # y_asm[chan, row, col-1]
        if col == R-1: m.FSW_S(11, (R*row + col  )*4, 12)  # y_asm[chan, row, col]
    # increment base addresses
    m.ADDI(11, 11, 9*4)    # for W(chan)
    m.ADDI(10, 10, R*R*4)  # for A(chan)
    m.ADDI(12, 12, R*R*4)  # for Y(chan)
    # TODOs:
    #  - parameterize above and eventually move into a def
    #  - add example for stride=2
    #  - reduce number of loads by computing several outputs in parallel (each output
    #    requires three registers for stride=1, so here we could compute 6 outputs in
    #    parallel; and when image-size is 6x6, process an entire column in parallel)


def conv_3x3x3_stride1(F, R, a_base, w_base, y_base):
  """assembly code with upper-case instruction for conv2D 3x3 with 3 in-channels,
  F out-channels, stride = 1, R input and output resolution.
  Register map:
    x[10] : base address for A[chan]
    x[11] : base address for W[chan]
    x[12] : base address for Y[chan]
    f[0] .. f[26]: the 27 weights (3, 3, 3) for one output channel
    f[27] : holds the latest activation loaded from memory
    f[28] : accumulation register 'out0' for current output
    f[29] : accumulation register 'out1' for next output
    f[30] : accumulation register 'out2' for next-next output"""

  # init base addresses
  m.LI(10, a_base)
  m.LI(11, w_base)
  m.LI(12, y_base)

  for chan in range(F): # 'chan' refers to 'output-channel'
    # load 3x3x3 weights for output-channel 'chan'
    for i in range(3):
      for j in range(3):
        for k in range(3):  # 'k' is input-channel
          m.FLW_S(9*i+3*j+k, (9*i+3*j+k)*4, 11)  # f[i, j, k] = W[i, j, k, chan]

    # compute all outputs (6x6) for channel 'chan'
    for row in range(R):
      for col in range(R):
        # load 3*3 activations, perform 27 muls, and store 1 output
        dot_start = 0 if row > 0   else 1  # first row is special
        dot_end   = 3 if row < R-1 else 2  # last row is special
        for dot in range(dot_start, dot_end):
          for k in range(3):  # 'k' is input-channel
            # load one activation from memory
            m.FLW_S(27, (3*R*(row-1+dot) + 3*col + k)*4, 10)  # A[row-1+dot, col, k]

            # compute 3 muls with weights W[dot, 0:3]
            if dot == dot_start and k == 0:
              if col > 0:
                m.FMADD_S(28, 27, 9*dot+3*2, 29)  # f28 = f27 * W[dot, 2, 0] + f29
                m.FMADD_S(29, 27, 9*dot+3*1, 30)  # f29 = f27 * W[dot, 1, 0] + f30
              else:
                m.FMUL_S(29, 27, 9*dot+3*1)       # f29 = f27 * W[dot, 1, 0]
              if col < R-1:
                m.FMUL_S(30, 27, 9*dot)           # f30 = f27 * W[dot, 0, 0]
            else:
              m.FMADD_S(29, 27, 9*dot+3*1+k, 29)                # f29 += f27 * W[dot, 1, k]
              if col > 0:   m.FMADD_S(28, 27, 9*dot+3*2+k, 28)  # f28 += f27 * W[dot, 2, k]
              if col < R-1: m.FMADD_S(30, 27, 9*dot+k, 30)      # f30 += f27 * W[dot, 0, k]
        # store result
        if col > 0:
          m.FSW_S(28, (R*row + col-1)*4, 12)  # Y[chan, row, col-1]
        if col == R-1:
          m.FSW_S(29, (R*row + col)*4, 12)    # Y[chan, row, col]
    # increment base addresses
    m.ADDI(11, 11, 27*4)   # for W
    m.ADDI(12, 12, R*R*4)  # for Y


def conv_3x3x3_stride2(F, R, a_base, w_base, y_base):
  """assembly code with upper-case instruction for conv2D 3x3 with 3 in-channels,
  F out-channels, stride = 2, R input resolution, and R/2 output resolution.
  Register map:
    x[10] : base address for A[chan]
    x[11] : base address for W[chan]
    x[12] : base address for Y[chan]
    f[0] .. f[26]: the 27 weights (3, 3, 3) for one output channel
    f[27] : holds the latest activation loaded from memory
    f[28] : accumulation register 'out0' for current output
    f[29] : accumulation register 'out1' for next output"""

  # init base addresses
  m.LI(10, a_base)
  m.LI(11, w_base)
  m.LI(12, y_base)
  Q = R//2  # output resolution

  for chan in range(F): # 'chan' refers to 'output-channel'
    # load 3x3x3 weights for output-channel 'chan'
    for i in range(3):
      for j in range(3):
        for k in range(3):  # 'k' is input-channel
          m.FLW_S(9*i+3*j+k, (9*i+3*j+k)*4, 11)  # f[i, j, k] = W[i, j, k, chan]
    # compute all outputs (QxQ) for channel 'chan'
    for row in range(1, R, 2):
      for col in range(R):
        # load 3*3 activations, perform 27 muls, and store 1 output
        for dot in range(0, 3 if row < R-1 else 2):  # last row is special
          for k in range(3):  # 'k' is input-channel
            init = (dot == 0) and (k == 0)  # shortcut for below if/else

            # load one activation from memory
            m.FLW_S(27, (3*R*(row-1+dot) + 3*col + k)*4, 10)  # A[row-1+dot, col, k]

            # compute 3 muls with weights W[dot, 0:3]
            if (col % 2) == 0:  # even columns
              if col > 0:
                m.FMADD_S(28, 27, 9*dot+3*2+k, 28)  # f28 += f27 * W[dot, 2, k]
              if init:
                m.FMUL_S(29, 27, 9*dot+3*0)         # f29  = f27 * W[dot, 0, 0]
              else:
                m.FMADD_S(29, 27, 9*dot+3*0+k, 29)  # f29 += f27 * W[dot, 0, k]
            else:  # odd columns
              if init:
                m.FMADD_S(28, 27, 9*dot+3*1, 29)    # f28  = f27 * W[dot, 1, 0] + f29
              else:
                m.FMADD_S(28, 27, 9*dot+3*1+k, 28)  # f28 += f27 * W[dot, 1, k]
        # store result
        if col > 0 and (col % 2) == 0:
          m.FSW_S(28, (Q*(row-1)//2 + (col-2)//2)*4, 12)  # Y[chan, (row-1)/2, (col-2)/2]
        if (col == R-1):
          m.FSW_S(28, (Q*(row-1)//2 + (col-1)//2)*4, 12)  # Y[chan, (row-1)/2, (col-1)/2]
    # increment base addresses
    m.ADDI(11, 11, 27*4)   # for W
    m.ADDI(12, 12, Q*Q*4)  # for Y
    # TODOs:
    #  - reduce number of loads by computing several outputs in parallel (each output
    #    requires two registers, so we could compute two outputs in parallel)
    #  - above code uses values for immediates that might excee 12-bit, e.g. the immediate
    #    for loading the activations (and perhaps also for storing the results). Therefore,
    #    increment the base addresses more frequently

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
conv_1x1(C, F, R, a_base, w_base, y_base)

# compare results against expected Y
y_asm = m.read_f32_vec(y_base, size=R*R*F).reshape(R*R, F)  # read result matrix
m.print_rel_err(y_asm, y)

#-------------------------------------------------------------------------------
# Example 3: Depthwise Conv2D 3x3 with 4 channels, stride=1, 6x6 image
#-------------------------------------------------------------------------------
print('-------------- Example 3: Depthwise Conv2D 3x3 layer ------------------')
C = 4  # channels
R = 6  # resolution

#-------------------------------------------------------------------------------
# generate activations and weights, run inference
a = np.random.normal(size=(1, R, R, C)).astype(np.float32)
w = np.random.normal(size=(3, 3, C)).astype(np.float32)
# activation shape: (1, R, R, C) : batch-size, RxR image, channels
# kernel shape: (3, 3, C) : 3x3 kernel, channels

# run inference with keras (golden reference)
y_k = DepthwiseConv2D(3, padding='same', depthwise_initializer=constant(w))(a)
y = y_k.numpy().reshape(R, R, C)  # flatten

#-------------------------------------------------------------------------------
# run assembly and compare
m.clear_mem()
m.clear_cpu()

# write A and W to memory
a_base = 0
w_base = a_base + R*R*C *4
y_base = w_base + 3*3*C *4
m.write_f32_vec(np.transpose(a, axes=[3, 0, 1, 2]).flatten(), a_base)
m.write_f32_vec(np.transpose(w, axes=[2, 0, 1]).flatten(), w_base)
# note on the transpose in the last two lines: We rearrange the matrices so
# that the last axis (channel) is now the first axis (aka 'channel-first order').
# That's important so that when we flatten it in row-major, all the pixels of the
# first channel are contigously in memory, because we process one channel at a time

# run assembly code
dw_conv_3x3_stride1(C, R, a_base, w_base, y_base)

# compare results against expected Y
y_asm = np.transpose(m.read_f32_vec(y_base, size=R*R*C).reshape(C, R, R), axes=[1, 2, 0])
m.print_rel_err(y_asm, y)

#-------------------------------------------------------------------------------
# Example 4: Conv2D 3x3, 3 in-channels, 8 out-channels, 12x12 image, stride=1
#-------------------------------------------------------------------------------
print('-------------- Example 4: Conv2D 3x3 layer, stride=1 ------------------')
F = 8   # output-channels
R = 12  # image resolution

#-------------------------------------------------------------------------------
# generate activations and weights, run inference
a = np.random.normal(size=(1, R, R, 3)).astype(np.float32)
w = np.random.normal(size=(3, 3, 3, F)).astype(np.float32)
# input shape:  (1, R, R, 3) : batch-size, RxR image, channels
# output shape: (1, R, R, F) : batch-size, RxR image, channels
# kernel shape: (3, 3, 3, F) : 3x3 kernel, in-channels, out-channels

# run inference with keras (golden reference)
y_k = Conv2D(F, 3, padding='same', kernel_initializer=constant(w))(a)
y = y_k.numpy().reshape(R, R, F)

#-------------------------------------------------------------------------------
# run assembly and compare
m.clear_mem()
m.clear_cpu()

# write to memory
a_base = 0
w_base = a.size * 4
y_base = (a.size + w.size) * 4
m.write_f32_vec(a.flatten(), a_base)
m.write_f32_vec(np.transpose(w, axes=[3, 0, 1, 2]).flatten(), w_base)
# transpose W so that the output-channels is first axes

# run assembly code
conv_3x3x3_stride1(F, R, a_base, w_base, y_base)

# compare results against expected
y_asm = np.transpose(m.read_f32_vec(y_base, size=R*R*F).reshape(F, R, R), axes=[1, 2, 0])
m.print_rel_err(y_asm, y)

#-------------------------------------------------------------------------------
# Example 5: Conv2D 3x3, 3 in-channels, 8 out-channels, 12x12 image, stride=2
#-------------------------------------------------------------------------------
print('-------------- Example 5: Conv2D 3x3 layer, stride=2 ------------------')
F = 8     # output-channels
R = 12    # input resolution
Q = R//2  # output resolution

#-------------------------------------------------------------------------------
# generate activations and weights, run inference
a = np.random.normal(size=(1, R, R, 3)).astype(np.float32)
w = np.random.normal(size=(3, 3, 3, F)).astype(np.float32)
# input shape:  (1, R, R, 3) : batch-size, RxR image, channels
# output shape: (1, Q, Q, F) : batch-size, QxQ image, channels
# kernel shape: (3, 3, 3, F) : 3x3 kernel, in-channels, out-channels

# run inference with keras (golden reference)
y_k = Conv2D(F, 3, padding='same', strides=2, kernel_initializer=constant(w))(a)
y = y_k.numpy().reshape(Q, Q, F)

# keras does the striding as follows: the first valid output equals the
# [1, 1] output of the non-strided version, etc.

#-------------------------------------------------------------------------------
# run assembly and compare
m.clear_mem()
m.clear_cpu()

# write to memory
a_base = 0
w_base = a.size * 4
y_base = (a.size + w.size) * 4
m.write_f32_vec(a.flatten(), a_base)
m.write_f32_vec(np.transpose(w, axes=[3, 0, 1, 2]).flatten(), w_base)
# transpose W so that the output-channel is first axes

# run assembly code
conv_3x3x3_stride2(F, R, a_base, w_base, y_base)

y_asm = np.transpose(m.read_f32_vec(y_base, size=Q*Q*F).reshape(F, Q, Q), axes=[1, 2, 0])
m.print_rel_err(y_asm, y)
