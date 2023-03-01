# These examples show how to implement neural networks with RISC-V assembly code

import numpy as np
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
# Example 2: 1x1 Conv2D with 128 input and 128 output channels, 12x12 image
#-------------------------------------------------------------------------------
print('-------------- Example 2: ------------------------')
m.clear_mem()
m.clear_cpu()

# generate matrices A (12, 128) and W (128, 128) and store them in memory
A = np.random.normal(size=(12, 128)).astype(np.float32)
W = np.random.normal(size=(128, 128)).astype(np.float32)
W_start    = A.size * 4
Y_start    = W_start + W.size * 4
code_start = Y_start + W_start + 1000
m.write_f32_vec(A.flatten(), 0)        # write A to mem[0]
m.write_f32_vec(W.flatten(), W_start)  # write W to mem[W_start]

# proof of concept: split W and A into 4x4 submatrices, then compute matmul(A, W)
A_split = np.empty((3, 32, 4, 4))
W_split = np.empty((32, 32, 4, 4))
for i in range(32):
  for j in range(32):
    W_split[i, j] = W[i*4:i*4+4, j*4:j*4+4]
for i in range(3):
  for j in range(32):
    A_split[i, j] = A[i*4:i*4+4, j*4:j*4+4]
# compute the big matmul by smaller 4x4 matmuls
Y = np.zeros((12, 128))
for i in range(3):
  for j in range(32):
    for k in range(32):
      Y[4*i:4*i+4, 4*j:4*j+4] += np.matmul(A_split[i, k], W_split[k, j])
m.print_rel_err(Y, np.matmul(A, W))  # compare Y against matmul(A, W)

# register map:
#   x[9]  : 1st base address for A_split
#   x[10] : 2nd base address for A_split
#   x[11] : base address for W_split
#   x[12] : base address for results (or Y_split)
#   f[11] : to store elements of A_split
#   f[12] .. f[15]: 4 registers to store an entire row of W_split
#   f[16] .. f[31]: the 16 outputs res[0, 0] ... res[4, 4]

# TODOs:
#  - replace the outer for loop (i, j, k) by assembly code with branches to
#    reduce the image size
#  - clean up the indexing and base address pointers

# store assembly program starting at address 'code_start'
m.pc = code_start
m.lbl('start')

# matmul (12,128) x (128,128) -> (12,128)
for i in range(3):
  m.asm('lui',  9,      m.hi20(4*128*4*i))  # m.x[9] = 4*128*4*i
  m.asm('addi', 9, 9,   m.lo12(4*128*4*i))
  m.asm('lui',  12,     m.hi20(Y_start + 4*128*4*i))  # m.x[12] = Y_start + ..
  m.asm('addi', 12, 12, m.lo12(Y_start + 4*128*4*i))

  # matmul (4,128) x (128,128) -> (4,128)
  for j in range(32):
    # set base address pointers
    m.asm('add', 10, 9, 0) # reset A_split pointer to x[9]
    m.asm('lui',  11,     m.hi20(W_start + 16*j))  # m.x[11] = W_start + 16*j
    m.asm('addi', 11, 11, m.lo12(W_start + 16*j))

    # matmul (4,128) x (128,4) -> (4,4)
    for k in range(32):
      # compute one 4x4 matmul (by computing 4 outer products)
      for ii in range(4):
        # load row ii of W_split into registers f[12] ... f[15]
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

      # increment base addresses for A_split and W_split
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

# execute program from 'start' to 'end'
m.exe(start='start', end='end')
m.print_perf()

# compare results against expected
res = m.read_f32_vec(Y_start, size=12*128).reshape(12, 128)  # read result matrix
m.print_rel_err(res, np.matmul(A, W))
# m.print_rel_err(res, Y)
