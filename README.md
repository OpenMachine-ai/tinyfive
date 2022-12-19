# TinyFive

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FOpenMachine-ai%2Ftinyfive&title_bg=%23555555&icon=&title=visitors+%28today+%2F+total%29&edge_flat=false)](https://hits.seeyoufarm.com)

TinyFive is a simple RISC-V emulator, [ISS](https://en.wikipedia.org/wiki/Instruction_set_simulator), and assembler written entirely in Python:
- It's useful for running neural networks on RISC-V: Simulate your RISC-V assembly code along with your neural network in Python (and without relying on RISC-V toolchains). Custom instructions can be easily added.
- TinyFive is also useful for ML scientists who are using ML/RL for compiler optimization (see e.g. [CompilerGym](https://github.com/facebookresearch/CompilerGym/blob/development/README.md)).
- If you want to learn how RISC-V works, TinyFive lets you play with instructions and assembly code.
- Can be very fast if you only use the upper-case instructions defined in the [first ~200 lines of tinyfive.py](tinyfive.py#L1-L200).
- [Fewer than 1000 lines](tinyfive.py) of code (w/o tests and examples)
- Uses NumPy for math

## Table of content
- [Usage](#usage)
  - [Example 1: Multiply two numbers](#example-1-multiply-two-numbers)
  - [Example 2: Add two vectors](#example-2-add-two-vectors)
  - [Example 3: Multiply two matrices](#example-3-multiply-two-matrices)
- [Running in colab notebook](#running-in-colab-notebook)
- [Running on your machine](#running-on-your-machine)
- [Speed](#speed)
- [Latest status](#latest-status)
- [Comparison](#comparison)
- [References](#references)
- [Tiny Tech promise](#tiny-tech-promise)

## Usage
TinyFive can be used in the following three ways:
- **Option A:** Use upper-case instructions such as `ADD()` and `MUL()`, see examples 1.1, 1.2, and 2.1 below.
- **Option B:** Use `asm()` and `exe()` functions without branch instructions, see examples 1.3 and 2.2 below.
- **Option C:** Use `asm()` and `exe()` functions with branch instructions, see example 2.3 below.

For all examples below, we assume that you import the TinyFive module and instantiate a RISC-V machine with at least 1KB of memory as follows:
```python
from tinyfive import tinyfive
m = tinyfive(mem_size=1000)  # instantiate RISC-V machine with 1KB of memory
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
for i in range(0, 8):
  m.asm('lw',  11, 4*i,      0)   # load x[11] with a[i] from mem[4*i + 0]
  m.asm('lw',  12, 4*(i+8),  0)   # load x[12] with b[i] from mem[4*(i+8) + 0]
  m.asm('add', 10, 11,       12)  # x[10] := x[11] + x[12]
  m.asm('sw',  10, 4*(i+16), 0)   # store results in mem[], starting at address 4*16

# execute program from address 4*48: execute 8*4 instructions and then stop
m.exe(start=4*48, instructions=8*4)

# compare results against golden reference
res = m.read_i32_vec(8, 4*16)  # read result vector from address 4*16
ref = a + b                    # golden reference: simply add a[] + b[]
print(res - ref)               # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]
```
**Example 2.3:** Same as example 2.2, but now use `asm()` and `exe()` functions with branch instructions (option C). The `lbl()` function defines labels, which are symbolic names that represent memory addresses. These labels are used for easier readability, especially for branch instructions and to mark the start and end of the assembly code executed by the `exe()` function.
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
res = m.read_i32_vec(8, 4*16)  # read result vector from address 4*16
ref = a + b                    # golden reference: simply add a[] + b[]
print(res - ref)               # print difference (should be all-zero)
# Output: [0 0 0 0 0 0 0 0]
```

### Example 3: Multiply two matrices
Coming soon

## Running in colab notebook
You can run TinyFive in [this colab notebook](https://colab.research.google.com/drive/1KXDPwSJmaOGefh5vAjrediwuiRf3wWa2?usp=sharing). This is the quickest way to get started and should work on any machine.

## Running on your machine
Clone the repo and install packages `numpy` and `fnmatch` as follows:
```bash
git clone https://github.com/OpenMachine-ai/tinyfive.git
cd tinyfive
pip3 install numpy fnmatch
```
To run the examples, type:
```bash
python3 examples.py
```
To run the test suite, type:
```bash
python3 tests.py
```

## Speed
- TinyFive is not optimized for speed (but for ease-of-use and [LOC](https://en.wikipedia.org/wiki/Source_lines_of_code)).
- You could use PyPy to speed it up (see e.g. the [Pydgin paper](https://www.csl.cornell.edu/~berkin/ilbeyi-pydgin-riscv2016.pdf) for details).
- If you only use the upper-case instructions such as `ADD()`, then TinyFive is very fast because there is no instruction decoding. And you should be able to accelerate it on a GPU or TPU.
- If you use the lower-case instructions with `asm()` and `exe()`, then execution of these functions is slow as they involve look-up and string matching with O(n) complexity where "n" is the total number of instructions. The current implementations of `asm()` and `dec()` are optimized for ease-of-use and readability. A faster implementation would collapse multiple look-ups into one look-up, optimize the pattern-matching for the instruction decoding (bits -> instruction), and change the order of the instructions so that more frequently used instructions are at the top of the list. [Here is an older version](https://github.com/OpenMachine-ai/tinyfive/blob/2aa4987391561c9c6692602ed3fccdeaee333e0b/tinyfive.py) of TinyFive with a faster `dec()` function that collapses two look-ups (`bits -> instruction` and `instruction -> uppeer-case instruction`) and doesn't use `fnmatch`.

## Latest status
- TinyFive is still under construction, many things haven't been implemented and tested yet.
- 37 of the 40 base instructions (RV32I), all instructions of the M-extension (RV32M) and the F-extension (RV32F) with the default rounding mode are already implemented, and many of them are tested.  (The three missing RV32I instructions `fence`, `ebreak`, and `ecall` are not applicable here.)
- Remaining work: improve testing, add perhaps more extensions. See TODOs in the code for more details.

## Comparison
The table below compares TinyFive with other ISS and emulator projects.

| ISS | Author | Language | Mature? | Extensions | LOC |
| --- | ------ | -------- | ------- | ---------- | --- |
| [TinyFive](https://github.com/OpenMachine-ai/tinyfive)             | OpenMachine          | Python    | No               | I, M, some F  | < 1k |
| [Pydgin](https://github.com/cornell-brg/pydgin)                    | Cornell University   | Python, C | Last update 2016 | A, D, F, I, M | |
| [Spike](https://github.com/riscv-software-src/riscv-isa-sim)       | UC Berkeley          | C, C++    | Yes              | All           | |
| [QEMU](https://www.qemu.org/) | [Fabrice Bellard](https://en.wikipedia.org/wiki/Fabrice_Bellard) | C  | Yes              | All           | |
| [riscvOVPsim](https://github.com/riscv-ovpsim/imperas-riscv-tests) | Imperas              | C         | Yes              | All           | |
| [Whisper](https://github.com/chipsalliance/SweRV-ISS)              | Western Digital      | C, C++    | Yes | Almost all                 | |
| [Sail Model](https://github.com/riscv/sail-riscv)                  | Cambridge, Edinburgh | Sail, C   | Yes | All                        | |
| [PiMaker/rvc](https://github.com/PiMaker/rvc)                      | PiMaker              | C         |     |                            | |
| [mini-rv32ima](https://github.com/cnlohr/mini-rv32ima)             | Charles Lohr         | C         |     | A, I, M, Zifencei, Zicsr   | < 1k |

## References
- Official [RISC-V spec](https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf)
- See [this RISC-V card](https://inst.eecs.berkeley.edu/~cs61c/fa18/img/riscvcard.pdf) for a brief description of most instructions. See also the [RISC-V reference card](http://riscvbook.com/greencard-20181213.pdf).
- Book "The RISC-V Reader: An Open Architecture Atlas" by David Patterson and Andrew Waterman (2 of the 4 founders of RISC-V). Appendix A of this book defines all instructions. The Spanish version of this book is [available for free](http://riscvbook.com/spanish/guia-practica-de-risc-v-1.0.5.pdf),
other free versions are [available here](http://riscvbook.com).
- Pydgin [paper](https://www.csl.cornell.edu/~berkin/ilbeyi-pydgin-riscv2016.pdf) and [video](https://youtu.be/-p_AGki7Vsk)
- [Online simulator](https://ascslab.org/research/briscv/simulator/simulator.html) for debug

## Tiny Tech promise
Similar to [tinygrad](https://github.com/geohot/tinygrad), [micrograd](https://github.com/karpathy/micrograd), and other “tiny tech” projects, we believe that core technology should be simple and small (in terms of [LOC](https://en.wikipedia.org/wiki/Source_lines_of_code)). Therefore, we will make sure that the core of TinyFive (without tests and examples) will always be below 1000 lines. Keep in mind that simplicity and size (in terms of number of instructions) is a key feature of [RISC](https://en.wikipedia.org/wiki/Reduced_instruction_set_computer): the "R" in RISC stands for "reduced" (as opposed to complex CISC). Specifically, the ISA manual of RISC-V has only ~200 pages while the ARM-32 manual is over 2000 pages long according to Fig. 1.6 of
the [RISC-V Reader](http://riscvbook.com/spanish/guia-practica-de-risc-v-1.0.5.pdf).

<p align="center">
  <img src="https://github.com/OpenMachine-ai/tinyfive/blob/main/logo.jpg">
</p>
