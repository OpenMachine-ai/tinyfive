from tinyfive import tinyfive
import numpy as np

# Two examples:
#   - Example 1: multiply two numbers
#   - Example 2: add two 8-element vectors
#
# TinyFive can be used in the following three ways:
#   - Option A: Use upper-case instructions such as ADD() and MUL(), see
#     examples 1.1, 1.2, and 2.1 below.
#   - Option B: Use enc() and exe() functions without branch instructions, see
#     examples 1.3 and 2.2 below.
#   - Option C: Use enc() and exe() functions with branch instructions, see
#     example 2.3 below.

#-------------------------------------------------------------------------------
# Example 1: multiply two numbers
#-------------------------------------------------------------------------------

m = tinyfive(mem_size=1000)  # instantiate RISC-V machine with 1KB of memory

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
m.enc('lw',  11, 0,  0)   # load register x[11] from mem[0 + 0]
m.enc('lw',  12, 4,  0)   # load register x[12] from mem[4 + 0]
m.enc('mul', 10, 11, 12)  # x[10] := x[11] * x[12]

# execute program from address 4*20: execute 3 instructions and then stop
m.exe(start=4*20, instructions=3)
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
# Example 2.2: same as example 2.1, but now use enc() and exe() functions without
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
  m.enc('lw',  11, 4*i,      0)   # load x[11] with a[i] from mem[4*i + 0]
  m.enc('lw',  12, 4*(i+8),  0)   # load x[12] with b[i] from mem[4*(i+8) + 0]
  m.enc('add', 10, 11,       12)  # x[10] := x[11] + x[12]
  m.enc('sw',  10, 4*(i+16), 0)   # store results in mem[], starting at address 4*16

# execute program from address 4*48: execute 8*4 instructions and then stop
m.exe(start=4*48, instructions=8*4)

# compare results against golden reference
res = m.read_i32_vec(8, 4*16)  # read result vector from address 4*16
ref = a + b                    # golden reference: simply add a[] + b[]
print(res - ref)               # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]

#-------------------------------------------------------------------------------
# Example 2.3: same as example 2.2, but now use enc() and exe() with branch
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
m.enc('add',  13, 0, 0)      # x[13] := x[0] + x[0] = 0 (because x[0] is always 0)
m.enc('addi', 14, 0, 32)     # x[14] := x[0] + 32 = 32 (because x[0] is always 0)
m.label('loop')              # define label "loop"
m.enc('lw',   11, 0,    13)  # load x[11] with a[] from mem[0 + x[13]]
m.enc('lw',   12, 4*8,  13)  # load x[12] with b[] from mem[4*8 + x[13]]
m.enc('add',  10, 11,   12)  # x[10] := x[11] + x[12]
m.enc('sw',   10, 4*16, 13)  # store x[10] in mem[4*16 + x[13]]
m.enc('addi', 13, 13,   4)   # x[13] := x[13] + 4 (increment x[13] by 4)
m.enc('bne',  13, 14, m.labels('loop'))  # branch to 'loop' if x[13] != x[14]

# execute program from address 4*48: execute 2+8*6 instructions and then stop
m.exe(start=4*48, instructions=2+8*6)

# compare results against golden reference
res = m.read_i32_vec(8, 4*16)  # read result vector from address 4*16
ref = a + b                    # golden reference: simply add a[] + b[]
print(res - ref)               # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]

# TODO: a slightly more efficient implementation decrements x[14] so that
# branch compare against zero, which eliminates the x[14] constant
