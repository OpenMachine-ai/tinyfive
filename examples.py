from tinyfive import tinyfive
import numpy as np

# Four examples:
#   - Example 1: multiply two numbers
#   - Example 2: add two 8-element vectors
#   - Example 3: multiply two 4x4 matrices
#   - Example 4: multiply two 8x8 matrices
#
# TinyFive can be used in the following three ways:
#   - Option A: Use upper-case instructions such as ADD() and MUL(), see
#     examples 1.1, 1.2, and 2.1 below.
#   - Option B: Use asm() and exe() functions without branch instructions, see
#     examples 1.3 and 2.2 below.
#   - Option C: Use asm() and exe() functions with branch instructions, see
#     example 2.3 below.

#-------------------------------------------------------------------------------
# Example 1: multiply two numbers
#-------------------------------------------------------------------------------
m = tinyfive(mem_size=4000)  # instantiate RISC-V machine with 4KB of memory

#-------------------------------------------------------------------------------
# Example 1.1: use option A with back-door loading of registers
m.x[11] = 6        # manually load '6' into register x[11]
m.x[12] = 7        # manually load '7' into register x[12]
m.MUL(10, 11, 12)  # x[10] := x[11] * x[12]
print(m.x[10])

#-------------------------------------------------------------------------------
# Example 1.2: same as example 1.1, but now load the data from memory
m.clear_cpu()
m.write_i32(6, 0)  # manually write '6' into mem[0] (memory @ address 0)
m.write_i32(7, 4)  # manually write '7' into mem[4] (memory @ address 4)
m.LW (11, 0,  0)   # load register x[11] from mem[0 + 0]
m.LW (12, 4,  0)   # load register x[12] from mem[4 + 0]
m.MUL(10, 11, 12)  # x[10] := x[11] * x[12]
print(m.x[10])

#-------------------------------------------------------------------------------
# Example 1.3: same as example 1.2, but now use option B
m.clear_cpu()
m.clear_mem()
m.write_i32(6, 0)  # manually write '6' into mem[0] (memory @ address 0)
m.write_i32(7, 4)  # manually write '7' into mem[4] (memory @ address 4)

# store assembly program in mem[] starting at address 4*20
m.pc = 4*20
m.asm('lw',  11, 0,  0)   # load register x[11] from mem[0 + 0]
m.asm('lw',  12, 4,  0)   # load register x[12] from mem[4 + 0]
m.asm('mul', 10, 11, 12)  # x[10] := x[11] * x[12]

# execute program from address 4*20: execute 3 instructions and then stop
m.exe(start=4*20, instructions=3)
#m.print_perf(start=4*20, end=4*20 + 4*3)
print(m.x[10])

#-------------------------------------------------------------------------------
# Example 2: add two 8-element vectors
#-------------------------------------------------------------------------------
np.random.seed(5)  # fix seed for reproducible results

# memory map:
#
#  byte address | contents
# --------------------------------------------------------
#  0   .. 4*7   | a-vector (elements a[0] to a[7])
# 4*8  .. 4*15  | b-vector (elements b[0] to b[7])
# 4*16 .. 4*23  | output c-vector (elements c[0] to c[7])
# Note: each element is 32 bits wide, thus occupies 4 byte-addresses in mem[]

#-------------------------------------------------------------------------------
# Example 2.1: use upper-case instructions without branch instructions

# generate 8-element vectors a[] and b[] and store them in memory
a = np.random.randint(100, size=8)
b = np.random.randint(100, size=8)
m.write_i32_vec(a, 0)    # write vector a[] to mem[0]
m.write_i32_vec(b, 4*8)  # write vector b[] to mem[4*8]

# pseudo-assembly for adding vectors a[] and b[] using Python for-loop
for i in range(0, 8):
  m.LW (11, 4*i,      0)   # load x[11] with a[i] from mem[4*i + 0]
  m.LW (12, 4*(i+8),  0)   # load x[12] with b[i] from mem[4*(i+8) + 0]
  m.ADD(10, 11,       12)  # x[10] := x[11] + x[12]
  m.SW (10, 4*(i+16), 0)   # store results in mem[], starting at address 4*16

# compare results against golden reference
res = m.read_i32_vec(8, 4*16)  # read result vector from address 4*16
ref = a + b                    # golden reference: simply add a[] + b[]
print(res - ref)               # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]

#-------------------------------------------------------------------------------
# Example 2.2: same as example 2.1, but now use asm() and exe() functions without
# branch instructions (option B)
m.clear_mem()
m.clear_cpu()

# generate 8-element vectors a[] and b[] and store them in memory
a = np.random.randint(100, size=8)
b = np.random.randint(100, size=8)
m.write_i32_vec(a, 0)    # write vector a[] to mem[0]
m.write_i32_vec(b, 4*8)  # write vector b[] to mem[4*8]

# store assembly program in mem[] starting at address 4*48
m.pc = 4*48
for i in range(0, 8):
  m.asm('lw',  11, 4*i,      0)   # load x[11] with a[i] from mem[4*i + 0]
  m.asm('lw',  12, 4*(i+8),  0)   # load x[12] with b[i] from mem[4*(i+8) + 0]
  m.asm('add', 10, 11,       12)  # x[10] := x[11] + x[12]
  m.asm('sw',  10, 4*(i+16), 0)   # store results in mem[], starting at address 4*16

# execute program from address 4*48: execute 8*4 instructions and then stop
m.exe(start=4*48, instructions=8*4)
#m.print_perf(start=4*48, end=4*48+ 4*8*4)

# compare results against golden reference
res = m.read_i32_vec(8, 4*16)  # read result vector from address 4*16
ref = a + b                    # golden reference: simply add a[] + b[]
print(res - ref)               # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]

#-------------------------------------------------------------------------------
# Example 2.3: same as example 2.2, but now use asm() and exe() with branch
# instructions (option C)
m.clear_mem()
m.clear_cpu()

# generate 8-element vectors a[] and b[] and store them in memory
a = np.random.randint(100, size=8)
b = np.random.randint(100, size=8)
m.write_i32_vec(a, 0)    # write vector a[] to mem[0]
m.write_i32_vec(b, 4*8)  # write vector b[] to mem[4*8]

# store assembly program starting at address 4*48
m.pc = 4*48
# x[13] is the loop-variable that is incremented by 4: 0, 4, .., 28
# x[14] is the constant for detecting the end of the for-loop
m.lbl('start')                 # define label 'start'
m.asm('add',  13, 0, 0)        # x[13] := x[0] + x[0] = 0 (because x[0] is always 0)
m.asm('addi', 14, 0, 32)       # x[14] := x[0] + 32 = 32 (because x[0] is always 0)
m.lbl('loop')                  # label 'loop'
m.asm('lw',   11, 0,    13)    # load x[11] with a[] from mem[0 + x[13]]
m.asm('lw',   12, 4*8,  13)    # load x[12] with b[] from mem[4*8 + x[13]]
m.asm('add',  10, 11,   12)    # x[10] := x[11] + x[12]
m.asm('sw',   10, 4*16, 13)    # store x[10] in mem[4*16 + x[13]]
m.asm('addi', 13, 13,   4)     # x[13] := x[13] + 4 (increment x[13] by 4)
m.asm('bne',  13, 14, 'loop')  # branch to 'loop' if x[13] != x[14]
m.lbl('end')                   # label 'end'

# execute program: start at label 'start', stop when label 'end' is reached
m.exe(start='start', end='end')

# compare results against golden reference
res = m.read_i32_vec(8, 4*16)  # read result vector from address 4*16
ref = a + b                    # golden reference
print(res - ref)               # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]

# print performance and dump out state
m.print_perf()
m.dump_state()

# A slightly more efficient implementation decrements the loop variable x[13]
# (instead of incrementing) so that the branch instruction compares against
# x[0]=0 (instead of the constant stored in x[14]), which frees up register
# x[14] and reduces the total number of instructions by 1.

#-------------------------------------------------------------------------------
# Example 3: multiply two 4x4 matrices
#-------------------------------------------------------------------------------

# memory map:
#
#  byte address | contents
# ------------------------------------------------------------------------------
#  0    .. 4*31 | A-matrix in row-major order (A[0][0], A[0][1], ... A[3][3])
#  4*32 .. 4*63 | B-matrix in row-major order (B[i][j] is at address 4*(32+i*4+j)
#  4*64 .. 4*95 | result matrix res[0][0] ... res[3][3]

#-------------------------------------------------------------------------------
# Example 3.1: use upper-case instructions (option A) with Python for-loop
m.clear_mem()
m.clear_cpu()

# generate 4x4 matrices A and B and store them in memory
A = np.random.randint(100, size=(4,4))
B = np.random.randint(100, size=(4,4))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*32)  # write matrix B to mem[4*32]

# pseudo-assembly for matmul(A,B) using Python for-loops
for i in range(0, 4):
  # load x[10] ... x[13] with row i of A
  for k in range(0, 4):
    m.LW (10+k, 4*(4*i+k), 0)  # load x[10+k] with A[i][k]

  for j in range(0, 4):
    # calculate dot product
    m.LW (18, 4*(32+j), 0)        # load x[18] with B[0][j]
    m.MUL(19, 10, 18)             # x[19] := x[10] * x[18] = A[i][0] * B[0][j]
    for k in range(1, 4):
      m.LW (18, 4*(32+4*k+j), 0)  # load x[18] with B[k][j]
      m.MUL(18, 10+k, 18)         # x[18] := x[10+k] * x[18] = A[i][k] * B[k][j]
      m.ADD(19, 19, 18)           # x[19] := x[19] + x[18]
    m.SW (19, 4*(64+i*4+j), 0)    # store res[i][j] from x[19]

# compare results against golden reference
res = m.read_i32_vec(4*4, 4*64).reshape(4,4)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True

#-------------------------------------------------------------------------------
# Example 3.2: same as example 3.1, but now asm() with branch ops (option C)
m.clear_mem()
m.clear_cpu()

# generate 4x4 matrices A and B and store them in memory
A = np.random.randint(100, size=(4,4))
B = np.random.randint(100, size=(4,4))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*32)  # write matrix B to mem[4*32]

# store assembly program starting at address 4*128
m.pc = 4*128
# here, we decrement the loop variables down to 0 so that we don't need an
# additional register to hold the constant for detecting the end of the loop
# x[20] is 4*4*i (i.e. the outer-loop variable) and is decremented by 16 from 64
# x[21] is 4*j (i.e. the inner-loop variable) and is decremented by 4 from 16
m.lbl('start')
m.asm('addi', 20, 0, 64)          # x[20] := 0 + 64

m.lbl('outer-loop')
m.asm('addi', 20, 20, -16)        # decrement loop-variable: x[20] := x[20] - 16
m.asm('lw',   10, 0,   20)        # load x[10] with A[i][0] from mem[0 + x[20]]
m.asm('lw',   11, 4,   20)        # load x[11] with A[i][1] from mem[4 + x[20]]
m.asm('lw',   12, 2*4, 20)        # load x[12] with A[i][2] from mem[2*4 + x[20]]
m.asm('lw',   13, 3*4, 20)        # load x[13] with A[i][3] from mem[3*4 + x[20]]
m.asm('addi', 21, 0, 16)          # reset loop-variable j: x[21] := 0 + 16

m.lbl('inner-loop')
m.asm('addi', 21, 21, -4)         # decrement j: x[21] := x[21] - 4

m.asm('lw',  18, 4*32, 21)        # load x[18] with B[0][j] from mem[4*32 + x[21]]
m.asm('mul', 19, 10, 18)          # x[19] := x[10] * x[18] = A[i][0] * B[0][j]

m.asm('lw',  18, 4*(32+4), 21)    # load x[18] with B[1][j]
m.asm('mul', 18, 11, 18)          # x[18] := x[11] * x[18] = A[i][1] * B[1][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(32+2*4), 21)  # load x[18] with B[2][j]
m.asm('mul', 18, 12, 18)          # x[18] := x[11] * x[18] = A[i][2] * B[2][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(32+3*4), 21)  # load x[18] with B[3][j]
m.asm('mul', 18, 13, 18)          # x[18] := x[11] * x[18] = A[i][3] * B[3][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('add', 24, 20, 21)          # calculate base address for result-matrix
m.asm('sw',  19, 4*64, 24)        # store res[i][j] from x[19]

m.asm('bne', 21, 0, 'inner-loop') # branch to 'inner-loop' if x[21] != 0
m.asm('bne', 20, 0, 'outer-loop') # branch to 'outer-loop' if x[20] != 0
m.lbl('end')

# execute program from 'start' to 'end'
m.exe(start='start', end='end')
m.print_perf()

# compare results against golden reference
res = m.read_i32_vec(4*4, 4*64).reshape(4,4)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True

#-------------------------------------------------------------------------------
# Example 3.3: Same as example 3.2, but now use Python for-loops in the assembly
# code to improve readability
m.clear_mem()
m.clear_cpu()

# generate 4x4 matrices A and B and store them in memory
A = np.random.randint(100, size=(4,4))
B = np.random.randint(100, size=(4,4))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*32)  # write matrix B to mem[4*32]

# store assembly program starting at address 4*128
m.pc = 4*128
# here, we decrement the loop variables down to 0 so that we don't need an
# additional register to hold the constant for detecting the end of the loop
# x[20] is 4*4*i (i.e. the outer-loop variable) and is decremented by 16 from 64
# x[21] is 4*j (i.e. the inner-loop variable) and is decremented by 4 from 16
m.lbl('start')
m.asm('addi', 20, 0, 64)            # x[20] := 0 + 64
m.lbl('outer-loop')
m.asm('addi', 20, 20, -16)          # decrement loop-variable: x[20] := x[20] - 16
for k in range(0, 4):
  m.asm('lw', 10+k, k*4, 20)        # load x[10+k] with A[i][k] from mem[k*4 + x[20]]
m.asm('addi', 21, 0, 16)            # reset loop-variable j: x[21] := 0 + 16
m.lbl('inner-loop')
m.asm('addi', 21, 21, -4)           # decrement j: x[21] := x[21] - 4
m.asm('lw',   18, 4*32, 21)         # load x[18] with B[0][j] from mem[4*32 + x[21]]
m.asm('mul',  19, 10, 18)           # x[19] := x[10] * x[18] = A[i][0] * B[0][j]
for k in range(1, 4):
  m.asm('lw',  18, 4*(32+k*4), 21)  # load x[18] with B[k][j]
  m.asm('mul', 18, 10+k, 18)        # x[18] := x[10+k] * x[18] = A[i][k] * B[k][j]
  m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]
m.asm('add', 24, 20, 21)            # calculate base address for result-matrix
m.asm('sw',  19, 4*64, 24)          # store res[i][j] from x[19]
m.asm('bne', 21, 0, 'inner-loop')   # branch to 'inner-loop' if x[21] != 0
m.asm('bne', 20, 0, 'outer-loop')   # branch to 'outer-loop' if x[20] != 0
m.lbl('end')

# execute program from 'start' to 'end'
m.exe(start='start', end='end')
m.print_perf()

# compare results against golden reference
res = m.read_i32_vec(4*4, 4*64).reshape(4,4)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True

#-------------------------------------------------------------------------------
# Example 4: multiply two 8x8 matrices
#-------------------------------------------------------------------------------

# memory map:
#
#  byte address | contents
# ------------------------------------------------------------------------------
#  0    .. 4*63   | A-matrix in row-major order (A[0][0], A[0][1], ... A[7][7])
#  4*64 .. 4*127  | B-matrix in row-major order (B[i][j] is at address 4*(64+i*8+j)
# 4*128 .. 4*191  | result matrix C[0][0] ... C[7][7]

#-------------------------------------------------------------------------------
# Example 4.1:
m.clear_mem()
m.clear_cpu()

# generate 8x8 matrices A and B and store them in memory
A = np.random.randint(100, size=(8,8))
B = np.random.randint(100, size=(8,8))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*64)  # write matrix B to mem[4*64]

# pseudo-assembly for matmul(A,B) using Python for-loops
for i in range(0, 8):
  # load x[10] ... x[17] with row i of A
  for k in range(0, 8):
    m.LW (10+k, 4*(8*i+k), 0)  # load x[10+k] with A[i][k]

  for j in range(0, 8):
    # calculate dot product
    m.LW (18, 4*(64+j), 0)        # load x[18] with B[0][j]
    m.MUL(19, 10, 18)             # x[19] := x[10] * x[18] = A[i][0] * B[0][j]
    for k in range(1, 8):
      m.LW (18, 4*(64+8*k+j), 0)  # load x[18] with B[k][j]
      m.MUL(18, 10+k, 18)         # x[18] := x[10+k] * x[18] = A[i][k] * B[k][j]
      m.ADD(19, 19, 18)           # x[19] := x[19] + x[18]
    m.SW (19, 4*(128+i*8+j), 0)   # store res[i][j] from x[19]

# compare results against golden reference
res = m.read_i32_vec(8*8, 4*128).reshape(8,8)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True

#-------------------------------------------------------------------------------
# Example 4.2:
m.clear_mem()
m.clear_cpu()

# generate 8x8 matrices A and B and store them in memory
A = np.random.randint(100, size=(8,8))
B = np.random.randint(100, size=(8,8))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*64)  # write matrix B to mem[4*64]

# store assembly program starting at address 4*256
m.pc = 4*256
# here, we decrement the loop variables down to 0 so that we don't need an
# additional register to hold the constant for detecting the end of the loop
# x[20] is 4*8*i (i.e. the outer-loop variable) and is decremented by 32 from 256
# x[21] is 4*j (i.e. the inner-loop variable) and is decremented by 4 from 32
m.lbl('start')
m.asm('addi', 20, 0, 256)         # x[20] := 0 + 256

m.lbl('outer-loop')
m.asm('addi', 20, 20, -32)        # decrement loop-variable: x[20] := x[20] - 32
m.asm('lw',   10, 0,   20)        # load x[10] with A[i][0] from mem[0 + x[20]]
m.asm('lw',   11, 4,   20)        # load x[11] with A[i][1] from mem[4 + x[20]]
m.asm('lw',   12, 2*4, 20)        # load x[12] with A[i][2] from mem[2*4 + x[20]]
m.asm('lw',   13, 3*4, 20)        # load x[13] with A[i][3] from mem[3*4 + x[20]]
m.asm('lw',   14, 4*4, 20)        # load x[14] with A[i][4] from mem[4*4 + x[20]]
m.asm('lw',   15, 5*4, 20)        # load x[15] with A[i][5] from mem[5*4 + x[20]]
m.asm('lw',   16, 6*4, 20)        # load x[16] with A[i][6] from mem[6*4 + x[20]]
m.asm('lw',   17, 7*4, 20)        # load x[17] with A[i][7] from mem[7*4 + x[20]]
m.asm('addi', 21, 0, 32)          # reset loop-variable j: x[21] := 0 + 32

m.lbl('inner-loop')
m.asm('addi', 21, 21, -4)         # decrement j: x[21] := x[21] - 4

m.asm('lw',  18, 4*64, 21)        # load x[18] with B[0][j] from mem[4*64 + x[21]]
m.asm('mul', 19, 10, 18)          # x[19] := x[10] * x[18] = A[i][0] * B[0][j]

m.asm('lw',  18, 4*(64+8), 21)    # load x[18] with B[1][j]
m.asm('mul', 18, 11, 18)          # x[18] := x[11] * x[18] = A[i][1] * B[1][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(64+2*8), 21)  # load x[18] with B[2][j]
m.asm('mul', 18, 12, 18)          # x[18] := x[11] * x[18] = A[i][2] * B[2][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(64+3*8), 21)  # load x[18] with B[3][j]
m.asm('mul', 18, 13, 18)          # x[18] := x[11] * x[18] = A[i][3] * B[3][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(64+4*8), 21)  # load x[18] with B[4][j]
m.asm('mul', 18, 14, 18)          # x[18] := x[11] * x[18] = A[i][4] * B[4][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(64+5*8), 21)  # load x[18] with B[5][j]
m.asm('mul', 18, 15, 18)          # x[18] := x[11] * x[18] = A[i][5] * B[5][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(64+6*8), 21)  # load x[18] with B[6][j]
m.asm('mul', 18, 16, 18)          # x[18] := x[11] * x[18] = A[i][6] * B[6][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(64+7*8), 21)  # load x[18] with B[7][j]
m.asm('mul', 18, 17, 18)          # x[18] := x[11] * x[18] = A[i][7] * B[7][j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('add', 24, 20, 21)          # calculate base address for result-matrix
m.asm('sw',  19, 4*128, 24)       # store res[i][j] from x[19]

m.asm('bne', 21, 0, 'inner-loop') # branch to 'inner-loop' if x[21] != 0
m.asm('bne', 20, 0, 'outer-loop') # branch to 'outer-loop' if x[20] != 0
m.lbl('end')

# execute program from 'start' to 'end'
m.exe(start='start', end='end')
m.print_perf()

# compare results against golden reference
res = m.read_i32_vec(8*8, 4*128).reshape(8,8)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True

#-------------------------------------------------------------------------------
# Example 4.3: Same as example 4.2, but now use Python for-loops in the assembly
# code to improve readability
m.clear_mem()
m.clear_cpu()

# generate 8x8 matrices A and B and store them in memory
A = np.random.randint(100, size=(8,8))
B = np.random.randint(100, size=(8,8))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*64)  # write matrix B to mem[4*64]

# store assembly program starting at address 4*256
m.pc = 4*256
# here, we decrement the loop variables down to 0 so that we don't need an
# additional register to hold the constant for detecting the end of the loop
# x[20] is 4*8*i (i.e. the outer-loop variable) and is decremented by 32 from 256
# x[21] is 4*j (i.e. the inner-loop variable) and is decremented by 4 from 32
m.lbl('start')
m.asm('addi', 20, 0, 256)           # x[20] := 0 + 256
m.lbl('outer-loop')
m.asm('addi', 20, 20, -32)          # decrement loop-variable: x[20] := x[20] - 32
for k in range(0, 8):
  m.asm('lw', 10+k, k*4, 20)        # load x[10+k] with A[i][k] from mem[k*4 + x[20]]
m.asm('addi', 21, 0, 32)            # reset loop-variable j: x[21] := 0 + 32
m.lbl('inner-loop')
m.asm('addi', 21, 21, -4)           # decrement j: x[21] := x[21] - 4
m.asm('lw',   18, 4*64, 21)         # load x[18] with B[0][j] from mem[4*64 + x[21]]
m.asm('mul',  19, 10, 18)           # x[19] := x[10] * x[18] = A[i][0] * B[0][j]
for k in range(1, 8):
  m.asm('lw',  18, 4*(64+k*8), 21)  # load x[18] with B[k][j]
  m.asm('mul', 18, 10+k, 18)        # x[18] := x[10+k] * x[18] = A[i][k] * B[k][j]
  m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]
m.asm('add', 24, 20, 21)            # calculate base address for result-matrix
m.asm('sw',  19, 4*128, 24)         # store res[i][j] from x[19]
m.asm('bne', 21, 0, 'inner-loop')   # branch to 'inner-loop' if x[21] != 0
m.asm('bne', 20, 0, 'outer-loop')   # branch to 'outer-loop' if x[20] != 0
m.lbl('end')

# execute program from 'start' to 'end'
m.exe(start='start', end='end')
m.print_perf()

# compare results against golden reference
res = m.read_i32_vec(8*8, 4*128).reshape(8,8)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True
