from model import *

# This file shows 3 ways to add two 16-element vectors (elements are INT32):
#  1) pseudo-assembly code using the upper-case instructions (such as 'ADD')
#  2) assembly code without any branch instructions
#  3) assembly code with branch instructions

#------------------------------------------------------------------------------
# golden reference: add two 16-element vectors using numpy
#------------------------------------------------------------------------------
np.random.seed(5)  # fix seed for reproducible results
a = np.random.randint(-1 << 31, (1 << 31)-1, size=16, dtype=np.int32)
b = np.random.randint(-1 << 31, (1 << 31)-1, size=16, dtype=np.int32)
c_gold = a + b

#-------------------------------------------------------------------------------
# 1st way: pseudo-assembly using the upper-case instructions (such as 'ADD')
#-------------------------------------------------------------------------------

# memory map: each 32-bit value occupies 4 byte addresses in mem
#
#  byte address | contents
# ---------------------------------------------------------------
#  0   .. 4*15  | a-vector (elements a[0] to a[15])
# 4*16 .. 4*31  | b-vector (elements b[0] to b[15])
# 4*32 .. 4*47  | output c-vector (elements c[0] to c[15])

# write vectors a[] and b[] to memory
write_i32_vec(s, a, 0)
write_i32_vec(s, b, 4*16)

# pseudo-assembly for adding vectors a[] and b[]
for i in range(0, 16):
  LW (s, a0, 4*i,      x0)
  LW (s, a1, 4*(i+16), x0)
  ADD(s, a2, a0,       a1)
  SW (s, a2, 4*(i+32), x0)

# compare results against golden reference
c_way1 = read_i32_vec(s, 16, 4*32)
print(c_gold - c_way1)

#-------------------------------------------------------------------------------
# 2nd way: assembly code without any branch instructions
#-------------------------------------------------------------------------------
clear_mem(s, start=4*32)
clear_cpu(s)

# store assembly program starting at address 4*48
s.pc = 4*48
for i in range(0, 16):
  enc(s, 'lw',  a0, 4*i,      x0)
  enc(s, 'lw',  a1, 4*(16+i), x0)
  enc(s, 'add', a2, a0,       a1)
  enc(s, 'sw',  a2, 4*(32+i), x0)

# execute program from address 4*48, execute 16*4 instructions and then stop
exe(s, start=4*48, instructions=16*4)

# compare results against golden reference
c_way2 = read_i32_vec(s, 16, 4*32)
print(c_gold - c_way2)

#-------------------------------------------------------------------------------
# 3rd way: assembly code with branch instructions
#-------------------------------------------------------------------------------
clear_mem(s, start=4*32)
clear_cpu(s)

# store assembly program starting at address 4*48
s.pc = 4*48
# a3 is the loop-variable that goes from 0, 4, 8, ... 60
# a4 is the constant for determining the end of the for-loop
enc(s, 'add',  a3, x0,   x0)   # a3 = 0
enc(s, 'addi', a4, x0,   64)   # a4 = 64
# loop:
enc(s, 'lw',   a0, 0,    a3)
enc(s, 'lw',   a1, 4*16, a3)
enc(s, 'add',  a2, a0,   a1)
enc(s, 'sw',   a2, 4*32, a3)
enc(s, 'addi', a3, a3,   4)    # a3 += 4
enc(s, 'bne',  a3, a4,  -5*4)  # jump back by 5 instructions to label 'loop'

# execute program from address 4*48
exe(s, start=4*48, instructions=2+16*6)

# compare results against golden reference
c_way3 = read_i32_vec(s, 16, 4*32)
print(c_gold - c_way3)

# TODO: a more efficient implementation decrements a4 so that branch compare
# against zero, which eliminates the a4 constant
