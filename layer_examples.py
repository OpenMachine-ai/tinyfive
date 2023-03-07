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

#-------------------------------------------------------------------------------
# Example 1: 4x4 Dense layer
#-------------------------------------------------------------------------------
print('-------------- Example 1: Dense layer ---------------------------------')
# This is a very small dense layer example using floating point

# generate 4x4 matrices A and B (float32) and store them in memory
A = np.random.normal(size=(4, 4)).astype(np.float32)
B = np.random.normal(size=(4, 4)).astype(np.float32)
m.write_f32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_f32_vec(B.flatten(), 4*16)  # write matrix B to mem[4*16]

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
m.print_rel_err(res, np.matmul(A, B))
# Output: should be very small, e.g. smaller than 1e-06, but could be larger

#-------------------------------------------------------------------------------
# Example 2: Conv2D 1x1, 32 input and 32 output channels, 6x6 image
#-------------------------------------------------------------------------------
print('-------------- Example 2: Conv2D 1x1 layer ----------------------------')
m.clear_mem()
m.clear_cpu()

# shapes
I = 6   # resolution of image (tested it up to 48 with C = 8)
C = 32  # input-channels (tested it up to 128)
C2 = C  # output-channels (tested it up to 128)

# base addresses
Astart = 0
Wstart = Astart + I*I*C * 4
Ystart = Wstart + C*C2 * 4
code_start = Ystart + I*I*C2 * 4

#-------------------------------------------------------------------------------
# generate activations and weights for keras (suffix *k is for keras)
Ak = np.random.normal(size=(1, I, I, C)).astype(np.float32)
Wk = np.random.normal(size=(1, 1, C, C2)).astype(np.float32)
  # input shape:  (1, I, I, C)  : batch-size, 4x4 image, channels
  # output shape: (1, I, I, C2) : batch-size, 4x4 image, channels
  # kernel shape: (1, 1, C, C2) : 1x1 kernel, in-channels, out-channels

# run inference with keras (golden reference)
Yk = Conv2D(C2, 1, kernel_initializer=constant(Wk))(Ak)

# TODO: use below if you want to use bias and ReLU activation
#  layer = Conv2D(128, 1, activation="relu", name="layer1", input_shape=(4, 4, 128),
#          kernel_initializer=constant(Wk),
#          bias_initializer=constant(Bk))
# Instead of using kr.initializer, you could use set_weights() as follows:
#   Yk = layer(Ak)  # dummy run with random weights, needed before set_weights()
#   layer.set_weights([Wk, Bk])
#   Yk = layer(Ak)
#   print(layer.get_weights()[0].shape)  # print weights
#   print(layer.get_weights()[1].shape)  # print biases

#-------------------------------------------------------------------------------
# flatten keras tensors, compare with matmul
A = Ak.reshape(I*I, C)  # Ak (1, I, I, C)  -> A (I*I, C)
W = Wk.reshape(C, C2)   # Wk (1, 1, C, C2) -> W (C, C2)
Y = Yk.numpy().reshape(I*I, C2)
m.print_rel_err(np.matmul(A, W), Y)  # compare matmul vs. keras conv2D

#-------------------------------------------------------------------------------
# proof of concept: split W and A into 4x4 submatrices, then compute matmul(A, W)
Asplit = np.empty((I*I//4, C//4, 4, 4))
Wsplit = np.empty((C//4, C2//4, 4, 4))
for i in range(C//4):
  for j in range(C2//4):
    Wsplit[i, j] = W[i*4:i*4+4, j*4:j*4+4]
for i in range(I*I//4):
  for j in range(C//4):
    Asplit[i, j] = A[i*4:i*4+4, j*4:j*4+4]
# compute the big matmul by smaller 4x4 matmuls
Ycon = np.zeros((I*I, C2))
for i in range(I*I//4):
  for j in range(C2//4):
    for k in range(C//4):
      Ycon[4*i:4*i+4, 4*j:4*j+4] += np.matmul(Asplit[i, k], Wsplit[k, j])
m.print_rel_err(Ycon, Y)  # compare Ycon against Y

#-------------------------------------------------------------------------------
# assembly code

# register map:
#   x[9]  : 1st base address for Asplit
#   x[10] : 2nd base address for Asplit
#   x[11] : base address for Wsplit
#   x[12] : base address for results (or Ysplit)
#   f[11] : to store elements of Asplit
#   f[12] .. f[15]: 4 registers to store an entire row of Wsplit
#   f[16] .. f[31]: the 16 outputs res[0, 0] ... res[4, 4]

# write A and W to memory
m.write_f32_vec(A.flatten(), Astart)  # write A to mem[Astart]
m.write_f32_vec(W.flatten(), Wstart)  # write W to mem[Wstart]

# store assembly program starting at address 'code_start'
m.pc = code_start
m.lbl('start')

# matmul (I*I, C) x (C, C2) -> (I*I, C2)
for i in range(I*I//4):
  m.asm('lui',  9,      m.hi20(Astart + 4*C*4*i))   # m.x[9] =  ...
  m.asm('addi', 9, 9,   m.lo12(Astart + 4*C*4*i))
  m.asm('lui',  12,     m.hi20(Ystart + 4*C2*4*i))  # m.x[12] = ...
  m.asm('addi', 12, 12, m.lo12(Ystart + 4*C2*4*i))

  # matmul (4, C) x (C, C2) -> (4, C2)
  for j in range(C2//4):
    # set base address pointers
    m.asm('add', 10, 9, 0) # reset Asplit pointer to x[9]
    m.asm('lui',  11,     m.hi20(Wstart + 16*j))  # m.x[11] = Wstart + 16*j
    m.asm('addi', 11, 11, m.lo12(Wstart + 16*j))

    # matmul (4, C) x (C, 4) -> (4, 4)
    for k in range(C//4):
      # compute one 4x4 matmul (by computing 4 outer products)
      for ii in range(4):
        # load row ii of Wsplit into registers f[12] ... f[15]
        for col in range(4):
          m.asm('flw.s', 12+col, 4*(col+C2*ii), 11)
        # compute outer-product in row-major order
        for row in range(4):
          m.asm('flw.s', 11, 4*(C*row+ii), 10)  # load f[11] with A[row, ii]
          for col in range(4):
            if ii==0 and k==0:  # no accumulation for the very first products
              m.asm('fmul.s', 16+4*row+col, 11, 12+col)  # f[] = f[11] * f[12]
            else:
              m.asm('fmadd.s', 16+4*row+col, 11, 12+col, 16+4*row+col) # f[] += f[11] * f[12]

      # increment base addresses for Asplit and Wsplit
      m.asm('addi', 10, 10, 4*4)  # increment by 16
      m.asm('addi', 11, 11, 4*4*C2//2) # increment by 4*4*C2 by two times 4*4*C2/2
      m.asm('addi', 11, 11, 4*4*C2//2)
      # note on the last two lines: 12-bit immediates: -2048 .. +2047. So we increment
      # by 1024 two times to achieve 2048 increment. Alternatively, we could decrement
      # the index (because -2048 is possible in one instruction). Or store the W-matrix
      # in transposed form

    # store results in memory
    for row in range(4):
      for col in range(4):
        m.asm('fsw.s', 16+4*row+col, 4*(row*C2+col), 12)
    m.asm('addi', 12, 12, 4*4)  # increment Y pointer by 16
m.lbl('end')

# TODOs:
#  - replace the outer for-loop (i, j, k) by assembly code with branches to
#    reduce the image size
#  - clean up the indexing and base address pointers
#  - use mnemonics x9 and f9 etc. instead of '9'
#  - rewrite above using only upper-case instructions to speed up runtime
#  - parameterize the assembly code and put it in a def so that it can be used
#    for other shapes, too

# execute program from 'start' to 'end'
m.exe(start='start', end='end')
m.print_perf()

# compare results against expected Y
Yasm = m.read_f32_vec(Ystart, size=I*I*C2).reshape(I*I, C2)  # read result matrix
m.print_rel_err(Yasm, Y)
m.print_rel_err(Yasm, Ycon)

#-------------------------------------------------------------------------------
# Example 3: Depthwise Conv2D 3x3 with 4 channels, stride=1, 6x6 image
#-------------------------------------------------------------------------------
print('-------------- Example 3: Depthwise Conv2D 3x3 layer ------------------')
m.clear_mem()
m.clear_cpu()

# shapes and base addresses
I = 6
C = 4
Astart = 0
Wstart = Astart + I*I*C *4
Ystart = Wstart + 3*3*C *4

#-------------------------------------------------------------------------------
# generate activations and weights, run inference
A = np.random.normal(size=(1, I, I, C)).astype(np.float32)
W = np.random.normal(size=(3, 3, C)).astype(np.float32)
# activation shape: (1, I, I, C) : batch-size, IxI image, channels
# kernel shape: (3, 3, C) : 3x3 kernel, channels

# run inference with keras (golden reference)
Yk = DepthwiseConv2D(3, padding='same', depthwise_initializer=constant(W))(A)
Y = Yk.numpy().reshape(I, I, C)  # flatten

#-------------------------------------------------------------------------------
# assembly code (with uppercase instructions)

# register map:
#   x[10] : base address for A[chan]
#   x[11] : base address for W[chan]
#   x[12] : base address for Y[chan]
#   f[0] .. f[8]: the 9 weights of a channel, stored in row-major order
#   f[9]  : holds the latest activation loaded from memory
#   f[10] : accumulation register 'out0' for current output
#   f[11] : accumulation register 'out1' for next output
#   f[12] : accumulation register 'out2' for next-next output

# write A and W to memory
m.write_f32_vec(np.transpose(A, axes=[3, 0, 1, 2]).flatten(), Astart)
m.write_f32_vec(np.transpose(W, axes=[2, 0, 1]).flatten(), Wstart)
# note on the transpose in the last two lines: We rearrange the matrices so
# that the last axis (channel) is now the first axis (aka 'channel-first order').
# That's important so that when we flatten it in row-major, all the pixels of the
# first channel are contigously in memory, because we process one channel at a time

# init base addresses
m.LI(10, Astart)  # for A
m.LI(11, Wstart)  # for W
m.LI(12, Ystart)  # for Y

for chan in range(C):
  # load 3x3 weights for channel 'chan'
  for i in range(3):
    for j in range(3):
      m.FLW_S(3*i+j, (3*i + j)*4, 11)  # f[i, j] = W[chan, i, j]

  # compute all outputs (IxI) for channel 'chan'
  for row in range(I):
    for col in range(I):
      # load 3 activations, perform 9 muls, and store 1 output
      dot_start = 0 if row > 0 else 1    # first row is special
      dot_end   = 3 if row < I-1 else 2  # last row is special
      for dot in range(dot_start, dot_end):
        # load one activation from memory
        m.FLW_S(9, (I*(row-1+dot) + col)*4, 10)  # A[chan, row-1+dot, col]

        # compute 3 muls with weights W[dot, 0:3]
        if dot == dot_start:
          if col > 0:
            m.FMADD_S(10, 9, 3*dot+2, 11)  # f10 = f9 * W[dot, 2] + f11
            m.FMADD_S(11, 9, 3*dot+1, 12)  # f11 = f9 * W[dot, 1] + f12
          else:
            m.FMUL_S(11, 9, 3*dot+1)            # f11 = f9 * W[dot, 1]
          if col < I-1: m.FMUL_S(12, 9, 3*dot)  # f12 = f9 * W[dot, 0]
        else:
          m.FMADD_S(11, 9, 3*dot+1, 11)                # f11 += f9 * W[dot, 1]
          if col > 0:   m.FMADD_S(10, 9, 3*dot+2, 10)  # f10 += f9 * W[dot, 2]
          if col < I-1: m.FMADD_S(12, 9, 3*dot, 12)    # f12 += f9 * W[dot, 0]
      # store result
      if col > 0:    m.FSW_S(10, (I*row + col-1)*4, 12)  # Yasm[chan, row, col-1]
      if col == I-1: m.FSW_S(11, (I*row + col  )*4, 12)  # Yasm[chan, row, col]

  # increment base addresses
  m.ADDI(11, 11, 9*4)    # for W(chan)
  m.ADDI(10, 10, I*I*4)  # for A(chan)
  m.ADDI(12, 12, I*I*4)  # for Y(chan)

  # TODOs:
  #  - parameterize above and eventually move into a def
  #  - add example for stride=2
  #  - reduce number of loads by computing several outputs in parallel (each output
  #    requires three registers for stride=1, so here we could compute 6 outputs in
  #    parallel; and when image-size is 6x6, process an entire column in parallel)

# compare results against expected Y
Yasm = np.transpose(m.read_f32_vec(Ystart, size=6*6*4).reshape(C, I, I), axes=[1, 2, 0])
m.print_rel_err(Yasm, Y)  # compare Yasm against Y

#-------------------------------------------------------------------------------
# Example 4: Conv2D 3x3, 3 in-channels, 8 out-channels, 12x12 image, stride=1
#-------------------------------------------------------------------------------
print('-------------- Example 4: Conv2D 3x3 layer, stride=1 ------------------')
m.clear_mem()
m.clear_cpu()

#-------------------------------------------------------------------------------
# generate activations and weights, run inference
A = np.random.normal(size=(1, 12, 12, 3)).astype(np.float32)
W = np.random.normal(size=(3, 3, 3, 8)).astype(np.float32)
# input shape:  (1, 12, 12, 3) : batch-size, 12x12 image, channels
# output shape: (1, 12, 12, 8) : batch-size, 12x12 image, channels
# kernel shape: (3, 3, 3, 8) : 3x3 kernel, in-channels, out-channels

# run inference with keras (golden reference)
Yk = Conv2D(8, 3, padding='same', kernel_initializer=constant(W))(A)
Y = Yk.numpy().reshape(12, 12, 8)

#-------------------------------------------------------------------------------
# assembly code (with uppercase instructions)

# register map:
#   x[10] : base address for A[chan]
#   x[11] : base address for W[chan]
#   x[12] : base address for Y[chan]
#   f[0] .. f[26]: the 27 weights (3, 3, 3) for one output channel
#   f[27] : holds the latest activation loaded from memory
#   f[28] : accumulation register 'out0' for current output
#   f[29] : accumulation register 'out1' for next output
#   f[30] : accumulation register 'out2' for next-next output

Wstart = A.size * 4
Ystart = (A.size + W.size) * 4
m.write_f32_vec(A.flatten(), 0)
m.write_f32_vec(np.transpose(W, axes=[3, 0, 1, 2]).flatten(), Wstart)
# transpose W so that the output-channels is first axes

# init base addresses
m.ADD(10, 0, 0)   # for A
m.LI(11, Wstart)  # for W
m.LI(12, Ystart)  # for Y

I = 12  # image size
C = 8   # output-channels

for chan in range(C): # 'chan' refers to 'output-channel'
  # load 3x3x3 weights for output-channel 'chan'
  for i in range(3):
    for j in range(3):
      for k in range(3):  # 'k' is input-channel
        m.FLW_S(9*i+3*j+k, (9*i+3*j+k)*4, 11)  # f[i, j, k] = W[i, j, k, chan]

  # compute all outputs (6x6) for channel 'chan'
  for row in range(I):
    for col in range(I):
      # load 3*3 activations, perform 27 muls, and store 1 output
      dot_start = 0 if row > 0   else 1  # first row is special
      dot_end   = 3 if row < I-1 else 2  # last row is special
      for dot in range(dot_start, dot_end):
        for k in range(3):  # 'k' is input-channel
          # load one activation from memory
          m.FLW_S(27, (36*(row-1+dot) + 3*col + k)*4, 10)  # A[row-1+dot, col, k]

          # compute 3 muls with weights W[dot, 0:3]
          if dot == dot_start and k == 0:
            if col > 0:
              m.FMADD_S(28, 27, 9*dot+3*2, 29)  # f28 = f27 * W[dot, 2, 0] + f29
              m.FMADD_S(29, 27, 9*dot+3*1, 30)  # f29 = f27 * W[dot, 1, 0] + f30
            else:
              m.FMUL_S(29, 27, 9*dot+3*1)       # f29 = f27 * W[dot, 1, 0]
            if col < I-1:
              m.FMUL_S(30, 27, 9*dot)           # f30 = f27 * W[dot, 0, 0]
          else:
            m.FMADD_S(29, 27, 9*dot+3*1+k, 29)                # f29 += f27 * W[dot, 1, k]
            if col > 0:   m.FMADD_S(28, 27, 9*dot+3*2+k, 28)  # f28 += f27 * W[dot, 2, k]
            if col < I-1: m.FMADD_S(30, 27, 9*dot+k, 30)      # f30 += f27 * W[dot, 0, k]

      # store result
      if col > 0: m.FSW_S(28, (I*row + col-1)*4, 12)  # Yasm[chan, row, col-1]
      if col == I-1: m.FSW_S(29, (I*row + col)*4, 12) # Yasm[chan, row, col]

  # increment base addresses
  m.ADDI(11, 11, 27*4)   # for W
  m.ADDI(12, 12, I*I*4)  # for Y

  # TODO: parameterize above and eventually move into a def

# compare results against expected Y
Yasm = np.transpose(m.read_f32_vec(Ystart, size=I*I*8).reshape(8, I, I), axes=[1, 2, 0])
m.print_rel_err(Yasm, Y)

#-------------------------------------------------------------------------------
# Example 5: Conv2D 3x3, 3 in-channels, 8 out-channels, 12x12 image, stride=2
#-------------------------------------------------------------------------------
# same as example 4, but now with stride=2

print('-------------- Example 5: Conv2D 3x3 layer, stride=2 ------------------')
m.clear_mem()
m.clear_cpu()

#-------------------------------------------------------------------------------
# generate activations and weights, run inference
A = np.random.normal(size=(1, 12, 12, 3)).astype(np.float32)
W = np.random.normal(size=(3, 3, 3, 8)).astype(np.float32)
# input shape:  (1, 12, 12, 3) : batch-size, 12x12 image, channels
# output shape: (1, 6, 6, 8) : batch-size, 6x6 image, channels
# kernel shape: (3, 3, 3, 8) : 3x3 kernel, in-channels, out-channels

# run inference with keras (golden reference)
Yk = Conv2D(8, 3, padding='same', strides=2, kernel_initializer=constant(W))(A)
Y = Yk.numpy().reshape(6, 6, 8)

# Keras does the striding as follows: the first valid output equals the
# [1, 1] output of the non-strided version, etc.

#-------------------------------------------------------------------------------
# assembly code (with uppercase instructions)

# register map:
#   x[10] : base address for A[chan]
#   x[11] : base address for W[chan]
#   x[12] : base address for Y[chan]
#   f[0] .. f[26]: the 27 weights (3, 3, 3) for one output channel
#   f[27] : holds the latest activation loaded from memory
#   f[28] : accumulation register 'out0' for current output
#   f[29] : accumulation register 'out1' for next output

Wstart = A.size * 4
Ystart = (A.size + W.size) * 4
m.write_f32_vec(A.flatten(), 0)
m.write_f32_vec(np.transpose(W, axes=[3, 0, 1, 2]).flatten(), Wstart)
# transpose W so that the output-channel is first axes

# init base addresses
m.ADD(10, 0, 0)   # for A
m.LI(11, Wstart)  # for W
m.LI(12, Ystart)  # for Y

C = 8   # output-channels
I = 12  # input-image size
I2 = I//2  # output-image

for chan in range(C): # 'chan' refers to 'output-channel'
  # load 3x3x3 weights for output-channel 'chan'
  for i in range(3):
    for j in range(3):
      for k in range(3):  # 'k' is input-channel
        m.FLW_S(9*i+3*j+k, (9*i+3*j+k)*4, 11)  # f[i, j, k] = W[i, j, k, chan]

  # compute all outputs (6x6) for channel 'chan'
  for row in range(1, I, 2):
    for col in range(I):
      # load 3*3 activations, perform 27 muls, and store 1 output
      for dot in range(0, 3 if row < I-1 else 2):  # last row is special
        for k in range(3):  # 'k' is input-channel
          init = (dot == 0) and (k == 0)  # shortcut for below if/else

          # load one activation from memory
          m.FLW_S(27, (36*(row-1+dot) + 3*col + k)*4, 10)  # A[row-1+dot, col, k]

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
        m.FSW_S(28, (I2*(row-1)//2 + (col-2)//2)*4, 12)  # Yasm[chan, (row-1)/2, (col-2)/2]
      if (col == I-1):
        m.FSW_S(28, (I2*(row-1)//2 + (col-1)//2)*4, 12)  # Yasm[chan, (row-1)/2, (col-1)/2]

  # increment base addresses
  m.ADDI(11, 11, 27*4)     # for W
  m.ADDI(12, 12, I2*I2*4)  # for Y

  # TODOs:
  #  - parameterize above and eventually move into a def
  #  - reduce number of loads by computing several outputs in parallel (each output
  #    requires two registers, so we could compute two outputs in parallel)

# compare results against expected Y
Yasm = np.transpose(m.read_f32_vec(Ystart, size=6*6*8).reshape(8, 6, 6), axes=[1, 2, 0])
m.print_rel_err(Yasm, Y)
