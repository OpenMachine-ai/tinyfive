# These examples show how to implement neural networks with RISC-V assembly code

import numpy as np
import keras as kr
from keras.layers import Conv2D
from machine import machine
# alternatively, uncomment the line below to use the tinyfive package
#   from tinyfive.machine import machine

np.random.seed(5)  # fix seed for reproducible results
m = machine(mem_size=2000000)  # instantiate RISC-V machine with 2MB of memory
# TODO: reduce to 500KB once we use branches to reduce image size

#-------------------------------------------------------------------------------
# Example 1: 4x4 Dense layer
#-------------------------------------------------------------------------------
print('-------------- Example 1: ------------------------')
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
# Example 2: 1x1 Conv2D with 128 input and 128 output channels, 4x4 image
#-------------------------------------------------------------------------------
print('-------------- Example 2: ------------------------')
m.clear_mem()
m.clear_cpu()

#-------------------------------------------------------------------------------
# generate activations and weights for keras (suffix *k is for keras)
Ak = np.random.normal(size=(1, 4, 4, 128)).astype(np.float32)
Wk = np.random.normal(size=(1, 1, 128, 128)).astype(np.float32)
  # input shape:  (1, 4, 4, 128) : batch-size, 4x4 image, channels
  # output shape: (1, 4, 4, 128) : batch-size, 4x4 image, channels
  # kernel shape: (1, 1, 128, 128) : 1x1 kernel, in-channels, out-channels

# define layer and run inference
layer = Conv2D(128, 1, input_shape=(4, 4, 128),
               kernel_initializer=kr.initializers.constant(Wk))
Yk = layer(Ak)
# TODO: use below if you want to use bias and ReLU activation
#   Conv2D(128, 1, activation="relu", name="layer1", input_shape=(3, 4, 128),
#          kernel_initializer=kr.initializers.constant(Wk),
#          bias_initializer=kr.initializers.constant(Bk))
# Instead of using kr.initializer, you could use set_weights() as follows:
#   Yk = layer(Ak)  # dummy run with random weights, needed before set_weights()
#   layer.set_weights([Wk, Bk])
#   Yk = layer(Ak)
#   print(layer.get_weights()[0].shape)  # print weights
#   print(layer.get_weights()[1].shape)  # print biases

#-------------------------------------------------------------------------------
# flatten keras tensors, compare with matmul
A = Ak.reshape(16, 128)   # Ak (1, 4, 4, 128)   -> A (16, 128)
W = Wk.reshape(128, 128)  # Wk (1, 1, 128, 128) -> W (128, 128)
Y = Yk.numpy().reshape(16, 128)
m.print_rel_err(np.matmul(A, W), Y)  # compare matmul vs. keras conv2D

#-------------------------------------------------------------------------------
# proof of concept: split W and A into 4x4 submatrices, then compute matmul(A, W)
Asplit = np.empty((4, 32, 4, 4))
Wsplit = np.empty((32, 32, 4, 4))
for i in range(32):
  for j in range(32):
    Wsplit[i, j] = W[i*4:i*4+4, j*4:j*4+4]
for i in range(4):
  for j in range(32):
    Asplit[i, j] = A[i*4:i*4+4, j*4:j*4+4]
# compute the big matmul by smaller 4x4 matmuls
Ycon = np.zeros((16, 128))
for i in range(4):
  for j in range(32):
    for k in range(32):
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
Wstart = A.size * 4
Ystart = Wstart + W.size * 4
code_start = Ystart + Wstart + 1000
m.write_f32_vec(A.flatten(), 0)       # write A to mem[0]
m.write_f32_vec(W.flatten(), Wstart)  # write W to mem[Wstart]

# store assembly program starting at address 'code_start'
m.pc = code_start
m.lbl('start')

# matmul (16, 128) x (128, 128) -> (16, 128)
for i in range(4):
  m.asm('lui',  9,      m.hi20(4*128*4*i))  # m.x[9] = 4*128*4*i
  m.asm('addi', 9, 9,   m.lo12(4*128*4*i))
  m.asm('lui',  12,     m.hi20(Ystart + 4*128*4*i))  # m.x[12] = Ystart + ..
  m.asm('addi', 12, 12, m.lo12(Ystart + 4*128*4*i))

  # matmul (4, 128) x (128, 128) -> (4, 128)
  for j in range(32):
    # set base address pointers
    m.asm('add', 10, 9, 0) # reset Asplit pointer to x[9]
    m.asm('lui',  11,     m.hi20(Wstart + 16*j))  # m.x[11] = Wstart + 16*j
    m.asm('addi', 11, 11, m.lo12(Wstart + 16*j))

    # matmul (4, 128) x (128, 4) -> (4, 4)
    for k in range(32):
      # compute one 4x4 matmul (by computing 4 outer products)
      for ii in range(4):
        # load row ii of Wsplit into registers f[12] ... f[15]
        for col in range(4):
          m.asm('flw.s', 12+col, 4*(col+128*ii), 11)
        # compute outer-product in row-major order
        for row in range(4):
          m.asm('flw.s', 11, 4*(128*row+ii), 10)  # load f[11] with A[row, ii]
          for col in range(4):
            if ii==0 and k==0:  # no accumulation for the very first products
              m.asm('fmul.s', 16+4*row+col, 11, 12+col)  # f[] = f[11] * f[12]
            else:
              m.asm('fmadd.s', 16+4*row+col, 11, 12+col, 16+4*row+col) # f[] += f[11] * f[12]

      # increment base addresses for Asplit and Wsplit
      m.asm('addi', 10, 10, 4*4)  # increment by 16
      m.asm('addi', 11, 11, 1024) # increment by 2048 is done by two times 1024
      m.asm('addi', 11, 11, 1024)
      # note on the last two lines: 12-bit immediates: -2048 .. +2047. So we increment
      # by 1024 two times to achieve 2048 increment. Alternatively, we could decrement
      # the index (because -2048 is possible in one instruction). Or store the W-matrix
      # in transposed form

    # store results in memory
    for row in range(4):
      for col in range(4):
        m.asm('fsw.s', 16+4*row+col, 4*(row*128+col), 12)
    m.asm('addi', 12, 12, 4*4)  # increment Y pointer by 16
m.lbl('end')

# TODOs:
#  - replace the outer for loop (i, j, k) by assembly code with branches to
#    reduce the image size
#  - clean up the indexing and base address pointers
#  - use mnemonics x9 and f9 etc. instead of '9'

# execute program from 'start' to 'end'
m.exe(start='start', end='end')
m.print_perf()

# compare results against expected Y
Yasm = m.read_f32_vec(Ystart, size=16*128).reshape(16, 128)  # read result matrix
m.print_rel_err(Yasm, Y)
m.print_rel_err(Yasm, Ycon)
