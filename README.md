# TinyFive

<a href="https://colab.research.google.com/github/OpenMachine-ai/tinyfive/blob/main/misc/colab.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>
[![Downloads](https://static.pepy.tech/badge/tinyfive)](https://pepy.tech/project/tinyfive)

<!--- view counter is currently commented out
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FOpenMachine-ai%2Ftinyfive&title_bg=%23555555&icon=&title=views+%28today+%2F+total%29&edge_flat=false)](https://hits.seeyoufarm.com)
 --->

TinyFive is a lightweight RISC-V emulator and assembler written entirely in Python:
- Useful for running neural networks on RISC-V: Simulate your RISC-V assembly code along with a neural network in Keras or PyTorch (and without relying on RISC-V toolchains).
- Custom instructions can be added for easy HW/SW codesign in Python (without C++ and compiler toolchains).
- If you want to learn how RISC-V works, TinyFive lets you play with instructions and assembly code in [this colab](https://colab.research.google.com/github/OpenMachine-ai/tinyfive/blob/main/misc/colab.ipynb).
- TinyFive might also be useful for ML scientists who are using ML/RL for compiler optimizations (see e.g. [CompilerGym](https://github.com/facebookresearch/CompilerGym/blob/development/README.md)) or to replace compiler toolchains by AI.
- Can be very fast if you only use the upper-case instructions defined in the [first ~200 lines of machine.py](machine.py#L1-L200).
- [Fewer than 1000 lines](machine.py) of code (w/o tests and examples)
- Uses NumPy for math

## Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Example 1: Multiply two numbers](#example-1-multiply-two-numbers)
  - [Example 2: Add two vectors](#example-2-add-two-vectors)
  - [Example 3: Multiply two matrices](#example-3-multiply-two-matrices)
  - [Example 4: Neural network layers](#example-4-neural-network-layers)
  - [Example 5: MobileNet](#example-5-mobilenet)
- [Running in colab](#running-in-colab)
- [Running without package](#running-without-package)
- [Contribute](#contribute)
- [Latest status](#latest-status)
- [Speed](#speed)
- [Comparison](#comparison)
- [References](#references)
- [Tiny Tech promise](#tiny-tech-promise)

## Installation
```
pip install tinyfive
```

## Usage
TinyFive can be used in the following three ways:
- **Option A:** Use upper-case instructions such as `ADD()` and `MUL()`, see examples 1.1, 1.2, 2.1, and 3.1 below.
- **Option B:** Use `asm()` and `exe()` functions without branch instructions, see examples 1.3 and 2.2 below.
- **Option C:** Use `asm()` and `exe()` functions with branch instructions, see example 2.3, 3.2, and 3.3 below.

For the examples below, import and instantiate a RISC-V machine with at least 4KB of memory as follows:
```python
from tinyfive.machine import machine
m = machine(mem_size=4000)  # instantiate RISC-V machine with 4KB of memory
```

### Example 1: Multiply two numbers
**Example 1.1:** Use upper-case instructions (option A) with back-door loading of registers.
```python
m.x[11] = 6        # manually load '6' into register x[11]
m.x[12] = 7        # manually load '7' into register x[12]
m.MUL(10, 11, 12)  # x[10] := x[11] * x[12]
print(m.x[10])
# Output: 42
```
**Example 1.2:** Same as example 1.1, but now load the data from memory. Specifically, the data values are stored at addresses 0 and 4. Here, each value is 32 bits wide (i.e. 4 bytes wide), which occupies 4 addresses in the byte-wide memory.
```python
m.write_i32(6, 0)  # manually write '6' into mem[0] (memory @ address 0)
m.write_i32(7, 4)  # manually write '7' into mem[4] (memory @ address 4)
m.LW (11, 0,  0)   # load register x[11] from mem[0 + 0]
m.LW (12, 4,  0)   # load register x[12] from mem[4 + 0]
m.MUL(10, 11, 12)  # x[10] := x[11] * x[12]
print(m.x[10])
# Output: 42
```
**Example 1.3:** Same as example 1.2, but now use `asm()` and `exe()` (option B). The assembler function `asm()` function takes an instruction and converts it into machine code and stores it in memory at address `s.pc`. Once the entire assembly program is written into memory `mem[]`, the `exe()` function (aka ISS) can then exectute the machine code stored in memory.
```python
m.write_i32(6, 0)  # manually write '6' into mem[0] (memory @ address 0)
m.write_i32(7, 4)  # manually write '7' into mem[4] (memory @ address 4)

# store assembly program in mem[] starting at address 4*20
m.pc = 4*20
m.asm('lw',  11, 0,  0)   # load register x[11] from mem[0 + 0]
m.asm('lw',  12, 4,  0)   # load register x[12] from mem[4 + 0]
m.asm('mul', 10, 11, 12)  # x[10] := x[11] * x[12]

# execute program from address 4*20: execute 3 instructions and then stop
m.exe(start=4*20, instructions=3)
print(m.x[10])
# Output: 42
```

### Example 2: Add two vectors
We are using the following memory map for adding two 8-element vectors `res[] := a[] + b[]`, where each vector element is 32 bits wide (i.e. each element occupies 4 byte-addresses in memory).
| Byte address | Contents |
| ------------ | -------- |
|  0   .. 4\*7   | a-vector: `a[0]` is at address 0, `a[7]` is at address 4\*7 |
| 4\*8  .. 4\*15 | b-vector: `b[0]` is at address 4\*8, `b[7]` is at address 4\*15 |
| 4\*16 .. 4\*23 | result-vector: `res[0]` is at address 4\*16, `res[7]` is at address 4\*23 |

**Example 2.1:** Use upper-case instructions (option A) with Python for-loop.
```python
# generate 8-element vectors a[] and b[] and store them in memory
a = np.random.randint(100, size=8)
b = np.random.randint(100, size=8)
m.write_i32_vec(a, 0)    # write vector a[] to mem[0]
m.write_i32_vec(b, 4*8)  # write vector b[] to mem[4*8]

# pseudo-assembly for adding vectors a[] and b[] using Python for-loop
for i in range(8):
  m.LW (11, 4*i,      0)   # load x[11] with a[i] from mem[4*i + 0]
  m.LW (12, 4*(i+8),  0)   # load x[12] with b[i] from mem[4*(i+8) + 0]
  m.ADD(10, 11,       12)  # x[10] := x[11] + x[12]
  m.SW (10, 4*(i+16), 0)   # store results in mem[], starting at address 4*16

# compare results against golden reference
res = m.read_i32_vec(4*16, size=8)  # read result vector from address 4*16
ref = a + b                         # golden reference: simply add a[] + b[]
print(res - ref)                    # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]
```
**Example 2.2**: Same as example 2.1, but now use `asm()` and `exe()` functions without branch instructions (option B).
```python
# generate 8-element vectors a[] and b[] and store them in memory
a = np.random.randint(100, size=8)
b = np.random.randint(100, size=8)
m.write_i32_vec(a, 0)    # write vector a[] to mem[0]
m.write_i32_vec(b, 4*8)  # write vector b[] to mem[4*8]

# store assembly program in mem[] starting at address 4*48
m.pc = 4*48
for i in range(8):
  m.asm('lw',  11, 4*i,      0)   # load x[11] with a[i] from mem[4*i + 0]
  m.asm('lw',  12, 4*(i+8),  0)   # load x[12] with b[i] from mem[4*(i+8) + 0]
  m.asm('add', 10, 11,       12)  # x[10] := x[11] + x[12]
  m.asm('sw',  10, 4*(i+16), 0)   # store results in mem[], starting at address 4*16

# execute program from address 4*48: execute 8*4 instructions and then stop
m.exe(start=4*48, instructions=8*4)

# compare results against golden reference
res = m.read_i32_vec(4*16, size=8)  # read result vector from address 4*16
ref = a + b                         # golden reference: simply add a[] + b[]
print(res - ref)                    # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]
```
**Example 2.3:** Same as example 2.2, but now use `asm()` and `exe()` functions with branch instructions (option C). The `lbl()` function defines labels, which are symbolic names that represent memory addresses. These labels improve the readability of branch instructions and mark the start and end of the assembly code executed by the `exe()` function.
```python
# generate 8-element vectors a[] and b[] and store them in memory
a = np.random.randint(100, size=8)
b = np.random.randint(100, size=8)
m.write_i32_vec(a, 0)    # write vector a[] to mem[0]
m.write_i32_vec(b, 4*8)  # write vector b[] to mem[4*8]

# store assembly program starting at address 4*48
m.pc = 4*48
# x[13] is the loop-variable that is incremented by 4: 0, 4, .., 28
# x[14] is the constant 28+4 = 32 for detecting the end of the for-loop
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
res = m.read_i32_vec(4*16, size=8)  # read result vector from address 4*16
ref = a + b                         # golden reference: simply add a[] + b[]
print(res - ref)                    # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]
```
A slightly more efficient implementation would decrement the loop variable `x[13]` (instead of incrementing) so that the branch instruction compares against `x[0] = 0` (instead of the constant stored in `x[14]`), which frees up register `x[14]` and reduces the total number of instructions by 1.

Use `print_perf()` to analyze performance and `dump_state()` to print out the current values of the register files and the the program counter (PC) as follows:
```python
>>> m.print_perf()
Ops counters: {'total': 50, 'load': 16, 'store': 8, 'mul': 0, 'add': 18, 'madd': 0, 'branch': 8}
x[] regfile : 5 out of 31 x-registers are used
f[] regfile : 0 out of 32 f-registers are used
Image size  : 32 Bytes

>>> m.dump_state()
pc   :  224
x[ 0]:    0, x[ 1]:    0, x[ 2]:    0, x[ 3]:    0
x[ 4]:    0, x[ 5]:    0, x[ 6]:    0, x[ 7]:    0
x[ 8]:    0, x[ 9]:    0, x[10]:   34, x[11]:   27
x[12]:    7, x[13]:   32, x[14]:   32, x[15]:    0
x[16]:    0, x[17]:    0, x[18]:    0, x[19]:    0
x[20]:    0, x[21]:    0, x[22]:    0, x[23]:    0
x[24]:    0, x[25]:    0, x[26]:    0, x[27]:    0
x[28]:    0, x[29]:    0, x[30]:    0, x[31]:    0
```

### Example 3: Multiply two matrices
We are using the following memory map for multiplying two 4x4 matrices as `res := np.matmul(A, B)`, where each matrix element is 32 bits wide (i.e. each element occupies 4 byte-addresses in memory).
| Byte address | Contents |
| ------------ | -------- |
|  0    .. 4\*15 | A-matrix in row-major order: `A[0, 0], A[0, 1], ... A[3, 3]` |
| 4\*16 .. 4\*31 | B-matrix in row-major order: `B[i, j]` is at address `4*(16+i*4+j)` |
| 4\*32 .. 4\*47 | result matrix `res[0, 0] ... res[3, 3]` |

**Example 3.1:** Use upper-case instructions (option A) with Python for-loop.
```python
# generate 4x4 matrices A and B and store them in memory
A = np.random.randint(100, size=(4, 4))
B = np.random.randint(100, size=(4, 4))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*16)  # write matrix B to mem[4*16]

# pseudo-assembly for matmul(A, B) using Python for-loops
for i in range(4):
  # load x[10] ... x[13] with row i of A
  for k in range(4):
    m.LW (10+k, 4*(4*i+k), 0)  # load x[10+k] with A[i, k]

  for j in range(4):
    # calculate dot product
    m.LW (18, 4*(16+j), 0)        # load x[18] with B[0, j]
    m.MUL(19, 10, 18)             # x[19] := x[10] * x[18] = A[i, 0] * B[0, j]
    for k in range(1, 4):
      m.LW (18, 4*(16+4*k+j), 0)  # load x[18] with B[k, j]
      m.MUL(18, 10+k, 18)         # x[18] := x[10+k] * x[18] = A[i, k] * B[k, j]
      m.ADD(19, 19, 18)           # x[19] := x[19] + x[18]
    m.SW (19, 4*(32+i*4+j), 0)    # store res[i, j] from x[19]

# compare results against golden reference
res = m.read_i32_vec(4*32, size=4*4).reshape(4, 4)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True
```
**Example 3.2:** Same as example 3.1, but now use `asm()` and `exe()` functions with branch instructions (option C).
```python
# generate 4x4 matrices A and B and store them in memory
A = np.random.randint(100, size=(4, 4))
B = np.random.randint(100, size=(4, 4))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*16)  # write matrix B to mem[4*16]

# store assembly program starting at address 4*128
m.pc = 4*128
# here, we decrement the loop variables down to 0 so that we don't need an
# additional register to hold the constant for detecting the end of the loop:
#  - x[20] is 4*4*i (i.e. the outer-loop variable) and is decremented by 16 from 64
#  - x[21] is 4*j (i.e. the inner-loop variable) and is decremented by 4 from 16
m.lbl('start')
m.asm('addi', 20, 0, 64)          # x[20] := 0 + 64

m.lbl('outer-loop')
m.asm('addi', 20, 20, -16)        # decrement loop-variable: x[20] := x[20] - 16
m.asm('lw',   10, 0,   20)        # load x[10] with A[i, 0] from mem[0 + x[20]]
m.asm('lw',   11, 4,   20)        # load x[11] with A[i, 1] from mem[4 + x[20]]
m.asm('lw',   12, 2*4, 20)        # load x[12] with A[i, 2] from mem[2*4 + x[20]]
m.asm('lw',   13, 3*4, 20)        # load x[13] with A[i, 3] from mem[3*4 + x[20]]
m.asm('addi', 21, 0, 16)          # reset loop-variable j: x[21] := 0 + 16

m.lbl('inner-loop')
m.asm('addi', 21, 21, -4)         # decrement j: x[21] := x[21] - 4

m.asm('lw',  18, 4*16, 21)        # load x[18] with B[0, j] from mem[4*16 + x[21]]
m.asm('mul', 19, 10, 18)          # x[19] := x[10] * x[18] = A[i, 0] * B[0, j]

m.asm('lw',  18, 4*(16+4), 21)    # load x[18] with B[1, j]
m.asm('mul', 18, 11, 18)          # x[18] := x[11] * x[18] = A[i, 1] * B[1, j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(16+2*4), 21)  # load x[18] with B[2, j]
m.asm('mul', 18, 12, 18)          # x[18] := x[11] * x[18] = A[i, 2] * B[2, j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('lw',  18, 4*(16+3*4), 21)  # load x[18] with B[3, j]
m.asm('mul', 18, 13, 18)          # x[18] := x[11] * x[18] = A[i, 3] * B[3, j]
m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]

m.asm('add', 24, 20, 21)          # calculate base address for result-matrix
m.asm('sw',  19, 4*32, 24)        # store res[i, j] from x[19]

m.asm('bne', 21, 0, 'inner-loop') # branch to 'inner-loop' if x[21] != 0
m.asm('bne', 20, 0, 'outer-loop') # branch to 'outer-loop' if x[20] != 0
m.lbl('end')

# execute program from 'start' to 'end'
m.exe(start='start', end='end')

# compare results against golden reference
res = m.read_i32_vec(4*32, size=4*4).reshape(4, 4)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True
```
**Example 3.3:** Same as example 3.2,  but now use Python for-loops in the assembly code to improve readability.
```python
# generate 4x4 matrices A and B and store them in memory
A = np.random.randint(100, size=(4, 4))
B = np.random.randint(100, size=(4, 4))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*16)  # write matrix B to mem[4*16]

# store assembly program starting at address 4*128
m.pc = 4*128
# here, we decrement the loop variables down to 0 so that we don't need an
# additional register to hold the constant for detecting the end of the loop:
#  - x[20] is 4*4*i (i.e. the outer-loop variable) and is decremented by 16 from 64
#  - x[21] is 4*j (i.e. the inner-loop variable) and is decremented by 4 from 16
m.lbl('start')
m.asm('addi', 20, 0, 64)            # x[20] := 0 + 64
m.lbl('outer-loop')
m.asm('addi', 20, 20, -16)          # decrement loop-variable: x[20] := x[20] - 16
for k in range(4):
  m.asm('lw', 10+k, k*4, 20)        # load x[10+k] with A[i, k] from mem[k*4 + x[20]]
m.asm('addi', 21, 0, 16)            # reset loop-variable j: x[21] := 0 + 16
m.lbl('inner-loop')
m.asm('addi', 21, 21, -4)           # decrement j: x[21] := x[21] - 4
m.asm('lw',   18, 4*16, 21)         # load x[18] with B[0, j] from mem[4*16 + x[21]]
m.asm('mul',  19, 10, 18)           # x[19] := x[10] * x[18] = A[i, 0] * B[0, j]
for k in range(1, 4):
  m.asm('lw',  18, 4*(16+k*4), 21)  # load x[18] with B[k, j]
  m.asm('mul', 18, 10+k, 18)        # x[18] := x[10+k] * x[18] = A[i, k] * B[k, j]
  m.asm('add', 19, 19, 18)          # x[19] := x[19] + x[18]
m.asm('add', 24, 20, 21)            # calculate base address for result-matrix
m.asm('sw',  19, 4*32, 24)          # store res[i, j] from x[19]
m.asm('bne', 21, 0, 'inner-loop')   # branch to 'inner-loop' if x[21] != 0
m.asm('bne', 20, 0, 'outer-loop')   # branch to 'outer-loop' if x[20] != 0
m.lbl('end')

# execute program from 'start' to 'end'
m.exe(start='start', end='end')

# compare results against golden reference
res = m.read_i32_vec(4*32, size=4*4).reshape(4, 4)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True
```
Performance numbers for example 3.3:
```python
>>> m.print_perf()
Ops counters: {'total': 269, 'load': 80, 'store': 16, 'mul': 64, 'add': 89, 'madd': 0, 'branch': 20}
x[] regfile : 9 out of 31 x-registers are used
f[] regfile : 0 out of 32 f-registers are used
Image size  : 92 Bytes
```
**Example 3.4:** 4x4 matrix multiplication optimized for runtime at the expense of image size and register file usage. Specifically, we first store the entire B matrix in the register file. And we fully unroll the for-loops to eliminate loop variables and branch instructions at the expense of a larger image size.
```python
# generate 4x4 matrices A and B and store them in memory
A = np.random.randint(100, size=(4, 4))
B = np.random.randint(100, size=(4, 4))
m.write_i32_vec(A.flatten(), 0)     # write matrix A to mem[0]
m.write_i32_vec(B.flatten(), 4*16)  # write matrix B to mem[4*16]

# store assembly program starting at address 4*128
m.pc = 4*128
m.lbl('start')
# load entire B matrix into registers x[16] ... x[31]
for i in range(4):
  for j in range(4):
    m.asm('lw', 16+4*i+j, 4*(16+4*i+j), 0)
# perform matmul in row-major order
for i in range(4):
  for k in range(4):                    # load x[10] ... x[13] with row i of A
    m.asm('lw', 10+k, 4*(4*i+k), 0)     # load x[10+k] with A[i, k]
  for j in range(4):
    m.asm('mul', 15, 10, 16+j)          # x[15] := x[10] * x[16+j] = A[i, 0] * B[0, j]
    for k in range(1, 4):
      m.asm('mul', 14, 10+k, 16+4*k+j)  # x[14] := x[10+k] * x[16+4k+j] = A[i, k] * B[k, j]
      m.asm('add', 15, 15, 14)          # x[15] := x[15] + x[14]
    m.asm('sw', 15, 4*(32+i*4+j), 0)    # store res[i, j] from x[15]
m.lbl('end')

# execute program from 'start' to 'end'
m.exe(start='start', end='end')

# compare results against golden reference
res = m.read_i32_vec(4*32, size=4*4).reshape(4, 4)  # read result matrix
ref = np.matmul(A, B)            # golden reference
print(np.array_equal(res, ref))  # should return 'True'
# Output: True
```
The table below shows a speedup of 1.7 with the following caveats:
- The bit-widths don't make sense for fixed point (in general, multiplying two 32-bit integers produces a 64-bit product; and adding 4 of these products requires up to 66 bits).
- For runtime calculations, we assume that our RISC-V CPU can only perform one instruction per cycle (while many RISC-V cores can perform multiple instructions per cycle).
- We assume all 31 registers can be used, which is unrealistic because we ignore register allocation conventions such as the procedure
calling conventions specified [here](https://github.com/riscv-non-isa/riscv-elf-psabi-doc).

|             | Image | Registers | Load | Store | Mul | Add | Branch | Total ops | Speedup |
|:-----------:|:-----:|:---------:|:----:|:-----:|:---:|:---:|:------:|:---------:|:-------:|
| Example 3.3 | 92B   | 9         | 80   | 16    | 64  | 89  | 20     | 269       | 1       |
| Example 3.4 | 640B  | 22        | 32   | 16    | 64  | 48  | 0      | 160       | 1.7     |

### Example 4: Neural network layers
Coming soon, see [file layer_examples.py](layer_examples.py) for now

### Example 5: MobileNet
Coming soon-ish, see [file mobilenet_v1_0.25.py](mobilenet_v1_0.25.py) for now

## Running in colab
<a href="https://colab.research.google.com/github/OpenMachine-ai/tinyfive/blob/main/misc/colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20">
</a>  This is the quickest way to get started and should work on any machine.

If you have a free Google Drive account, you can make a copy of this colab via the menu `File` -> `Save a copy in Drive`. Now you can edit the code.

Alternatively, start a new colab in your Google Drive as follows: [Go here](https://drive.google.com/drive/my-drive) and click on `New` -> `More` -> `Google Colaboratory`. Then copy below lines into your colab:

```python
!pip install tinyfive
from tinyfive.machine import machine
import numpy as np

m = machine(mem_size=4000)  # instantiate RISC-V machine with 4KB of memory
```

## Running without package
If you don't want to use the TinyFive python package, then you can clone the latest repo and install numpy as follows:
```bash
git clone https://github.com/OpenMachine-ai/tinyfive.git
cd tinyfive
pip install numpy
```
To run the examples, type:
```bash
python3 examples.py
```
To run the test suite, type:
```bash
python3 tests.py
```

If you don't want to run above steps on your local machine, you can run it in a colab as follows: Start a new colab in your Google Drive by [going here](https://drive.google.com/drive/my-drive) and clicking on `New` -> `More` -> `Google Colaboratory`. Then copy below lines into your colab:
```python
!git clone https://github.com/OpenMachine-ai/tinyfive.git
%cd tinyfive

# run examples
!python3 examples.py

# run test suite
!python3 tests.py
```
## Contribute
If you like this project, give it a ⭐ and share it with friends!  And if you are interested in helping make TinyFive better,
I highly welcome you to do so. I thank you in advance for your interest.  If you are unsure of what you could do to improve the project, you may have a look [here](https://github.com/OpenMachine-ai/tinyfive/issues/5).

## Latest status
- TinyFive is still under construction, many things haven't been implemented and tested yet.
- 37 of the 40 base instructions (RV32I), all instructions of the M-extension (RV32M) and the F-extension (RV32F) with the default rounding mode are already implemented, and many of them are tested.  (The three missing RV32I instructions `fence`, `ebreak`, and `ecall` are not applicable here.)
- Remaining work: improve testing, add more extensions. See TODOs in the code for more details.
- Stay updated by following us on [Twitter](https://twitter.com/OpenMachine_AI), [Post.news](https://post.news/@/openmachine), and [LinkedIn](https://www.linkedin.com/in/nilsgraef/).

## Speed
- TinyFive is not optimized for speed (but for ease-of-use and [LOC](https://en.wikipedia.org/wiki/Source_lines_of_code)).
- You might be able to use PyPy or [Codon](https://github.com/exaloop/codon) to speed up TinyFive (see e.g. the [Pydgin paper](https://www.csl.cornell.edu/~berkin/ilbeyi-pydgin-riscv2016.pdf) for details).
- If you only use the upper-case instructions such as `ADD()`, then TinyFive is very fast because there is no instruction decoding. And you should be able to accelerate it on a GPU or TPU.
- If you use the lower-case instructions with `asm()` and `exe()`, then execution of these functions is slow as they involve look-up and string matching with O(n) complexity where "n" is the total number of instructions. The current implementations of `asm()` and `dec()` are optimized for ease-of-use and readability. A faster implementation would collapse multiple look-ups into one look-up, optimize the pattern-matching for the instruction decoding (bits -> instruction), and change the order of the instructions so that more frequently used instructions are at the top of the list. [Here is an older version](https://github.com/OpenMachine-ai/tinyfive/blob/2aa4987391561c9c6692602ed3fccdeaee333e0b/tinyfive.py) of TinyFive with a faster `dec()` function that collapses two look-ups (`bits -> instruction` and `instruction -> uppeer-case instruction`) and doesn't use `fnmatch`.

## Comparison
The table below compares TinyFive with other [ISS](https://en.wikipedia.org/wiki/Instruction_set_simulator) and emulator projects.

| ISS | Author | Language | Mature? | Extensions | LOC |
| --- | ------ | -------- | ------- | ---------- | --- |
| [TinyFive](https://github.com/OpenMachine-ai/tinyfive)             | OpenMachine          | Python    | No               | I, M, some F  | < 1k |
| [Pydgin](https://github.com/cornell-brg/pydgin)                    | Cornell University   | Python, C | Last update 2016 | A, D, F, I, M | |
| [Spike](https://github.com/riscv-software-src/riscv-isa-sim)       | UC Berkeley          | C, C++    | Yes              | All           | |
| [QEMU](https://www.qemu.org/) | [Fabrice Bellard](https://en.wikipedia.org/wiki/Fabrice_Bellard) | C  | Yes              | All           | |
| [TinyEMU](https://bellard.org/tinyemu/) | [Fabrice Bellard](https://en.wikipedia.org/wiki/Fabrice_Bellard) | C  | Yes    | All           | |
| [riscvOVPsim](https://github.com/riscv-ovpsim/imperas-riscv-tests) | Imperas              | C         | Yes              | All           | |
| [Whisper](https://github.com/chipsalliance/SweRV-ISS)              | Western Digital      | C, C++    | Yes | Almost all                 | |
| [Sail Model](https://github.com/riscv/sail-riscv)                  | Cambridge, Edinburgh | Sail, C   | Yes | All                        | |
| [PiMaker/rvc](https://github.com/PiMaker/rvc)                      | PiMaker              | C         |     |                            | |
| [mini-rv32ima](https://github.com/cnlohr/mini-rv32ima)             | Charles Lohr         | C         |     | A, I, M, Zifencei, Zicsr   | < 1k |

## References
- [HuggingFive:raised_hand_with_fingers_splayed:](https://github.com/OpenMachine-ai/HuggingFive)
- Official [RISC-V spec](https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf)
- See [this RISC-V card](https://inst.eecs.berkeley.edu/~cs61c/fa18/img/riscvcard.pdf) for a brief description of most instructions. See also the [RISC-V reference card](http://riscvbook.com/greencard-20181213.pdf).
- Book [The RISC-V Reader: An Open Architecture Atlas](https://www.abebooks.com/book-search/author/patterson-david-waterman-andrew/) by David Patterson and Andrew Waterman. Appendix A of this book defines all instructions. The Spanish version of this book is [available for free](http://riscvbook.com/spanish/guia-practica-de-risc-v-1.0.5.pdf),
other free versions are [available here](http://riscvbook.com).
- Pydgin [paper](https://www.csl.cornell.edu/~berkin/ilbeyi-pydgin-riscv2016.pdf) and [video](https://youtu.be/-p_AGki7Vsk)
- [Online simulator](https://ascslab.org/research/briscv/simulator/simulator.html) for debug

## Tiny Tech promise
Similar to [TinyEMU](https://bellard.org/tinyemu/), [tinygrad](https://github.com/geohot/tinygrad), and other “tiny tech” projects, we believe that core technology should be simple and small (in terms of LOC). Therefore, we will make sure that the core of TinyFive (without tests and examples) will always be below 1000 lines.

Simplicity and size (in terms of number of instructions) is a key feature of [RISC](https://en.wikipedia.org/wiki/Reduced_instruction_set_computer): the "R" in RISC stands for "reduced" (as opposed to complex CISC). Specifically, the ISA manual of RISC-V has only ~200 pages while the ARM-32 manual is over 2000 pages long according to Fig. 1.6 of
the [RISC-V Reader](http://riscvbook.com/spanish/guia-practica-de-risc-v-1.0.5.pdf).

<p align="center">
  <img src="https://github.com/OpenMachine-ai/tinyfive/blob/main/misc/logo.jpg">
</p>
