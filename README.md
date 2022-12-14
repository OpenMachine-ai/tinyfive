# TinyFive

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FOpenMachine-ai%2Ftinyfive&title_bg=%23555555&icon=&title=visitors+%28today+%2F+total%29&edge_flat=false)](https://hits.seeyoufarm.com)

TinyFive is a simple RISC-V emulator and
[ISS](https://en.wikipedia.org/wiki/Instruction_set_simulator) written entirely in Python:
- It's useful for running neural networks on RISC-V: Simulate your RISC-V assembly code along with your neural network in Python (and without relying on RISC-V toolchains). Custom instructions can be easily added.
- TinyFive is also useful for ML scientists who are using ML/RL for compiler optimization (see [CompilerGym](https://github.com/facebookresearch/CompilerGym/blob/development/README.md)).
- If you want to learn how RISC-V works, TinyFive lets you play with instructions and assembly code.
- Can be very fast if you only use the upper-case instructions defined in the 
  [first 200 lines of tinyfive.py](https://github.com/OpenMachine-ai/tinyfive/blob/main/tinyfive.py#L1-L200)
- [Fewer than 1000 lines](https://github.com/OpenMachine-ai/tinyfive/blob/main/tinyfive.py) of code (w/o tests and examples)
- Uses NumPy for math

## Table of content
- [Usage](#usage)
  - [Example 1: multiply two numbers](#example-1-multiply-two-numbers)
  - [Example 2: add two 8-element vectors](#example-2-add-two-8-element-vectors)
- [Running in colab notebook](#running-in-colab-notebook)
- [Running on your machine](#running-on-your-machine)
- [Speed](#speed)
- [Latest status](#latest-status)
- [Comparison](#comparison)
- [References](#references)
- [Tiny Tech promise](#tiny-tech-promise)

## Usage
TinyFive can be used in the following three ways:
- **Option A:** Use upper-case instructions such as `ADD()` and `MUL()`, see e.g. examples 1.1 and 1.2 below.
- **Option B:** Use `enc()` and `exe()` without branch instructions, see e.g. example 1.3 below.
- **Option C:** Use `enc()` and `exe()` with branch instructions

### Example 1: multiply two numbers
**Example 1.1:** Use upper-case instructions (option A) with back-door loading of registers
```python
s.x[11] = 6         # manually load '6' into register x[11]
s.x[12] = 7         # manually load '7' into register x[12]
MUL(s, 10, 11, 12)  # x[10] := x[11] * x[12]
print(s.x[10])
# Output: 42
```
**Example 1.2:** Same as example 1.1, but now load the data from memory
```python
write_i32(s, 6, 0)  # manually write '6' into mem[0] (memory @ address 0)
write_i32(s, 7, 4)  # manually write '7' into mem[4] (memory @ address 4)
LW (s, 11, 0,  0)   # load register x[11] from mem[0 + 0]
LW (s, 12, 4,  0)   # load register x[12] from mem[4 + 0]
MUL(s, 10, 11, 12)  # x[10] := x[11] * x[12]
print(s.x[10])
# Output: 42
```
**Example 1.3:** Same as example 1.2, but now use `enc()` and `exe()` (option B)
```python
write_i32(s, 6, 0)  # manually write '6' into mem[0] (memory @ address 0)
write_i32(s, 7, 4)  # manually write '7' into mem[4] (memory @ address 4)

# store assembly program in mem[] starting at address 4*20
s.pc = 4*20
enc(s, 'lw',  11, 0,  0)   # load register x[11] from mem[0 + 0]
enc(s, 'lw',  12, 4,  0)   # load register x[12] from mem[4 + 0]
enc(s, 'mul', 10, 11, 12)  # x[10] := x[11] * x[12]

# execute program from address 4*20, execute 3 instructions and then stop
exe(s, start=4*20, instructions=3)
print(s.x[10])
# Output: 42
```

### Example 2: add two 8-element vectors



## Running in colab notebook
You can run TinyFive in
[this colab notebook](https://colab.research.google.com/drive/1KXDPwSJmaOGefh5vAjrediwuiRf3wWa2?usp=sharing).
This is the quickest way to get started and should work on any machine.

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
- TinyFive is not optimized for speed (but for ease-of-use and LOC).
- You could use PyPy to speed it up (see e.g. the Pydgin paper for details).
- If you only use the upper-case instructions such as `ADD()`, then TinyFive
  is very fast because there is no instruction decoding. And you should be
  able to accelerate it on a GPU or TPU.
- If you use the lower-case instructions with `enc()` and `exe()`, then
  execution of these functions is slow as they involve look-up
  and string matching with O(n) complexity where "n" is the total number of
  instructions. The current implementations of `enc()` and `dec()` are optimized
  for ease-of-use and readability. A faster implementation would collapse
  multiple look-ups into one look-up, optimize the pattern-matching for the
  instruction decoding (bits -> instruction), and change the order of the
  instructions so that more frequently used instructions are at the top of
  the list. 
  [Here is an older version](https://github.com/OpenMachine-ai/tinyfive/blob/2aa4987391561c9c6692602ed3fccdeaee333e0b/tinyfive.py) 
  of TinyFive with a faster `dec()` function that collapses two look-ups
  (`bits -> instruction` and `instruction -> uppeer-case instruction`)
  and doesn't use `fnmatch`.  

## Latest status
- TinyFive is still under construction, many things haven't been implemented and tested yet.
- 37 of the 40 base instructions (RV32I), all instructions of the M-extension (RV32M) and
  the F-extension (RV32F) with the default rounding mode are already implemented, and many
  of them are tested.  (The three missing RV32I instructions `fence`, `ebreak`, and `ecall`
  are not applicable here.)
- Remaining work: improve testing, add perhaps more extensions. See TODOs in the code for
  more details.

## Comparison
The table below compares TinyFive with other ISS projects.

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
- See [this RISC-V card](https://inst.eecs.berkeley.edu/~cs61c/fa18/img/riscvcard.pdf)
 for a brief description of most instructions. See also the
 [RISC-V reference card](http://riscvbook.com/greencard-20181213.pdf).
- Book "The RISC-V Reader: An Open Architecture Atlas" by David Patterson and Andrew Waterman
(2 of the 4 founders of RISC-V). Appendix A of this book defines all instructions.
The Spanish version of this book is
[available for free](http://riscvbook.com/spanish/guia-practica-de-risc-v-1.0.5.pdf),
other free versions are [available here](http://riscvbook.com).
- Pydgin [paper](https://www.csl.cornell.edu/~berkin/ilbeyi-pydgin-riscv2016.pdf) and [video](https://youtu.be/-p_AGki7Vsk)
- [Online simulator](https://ascslab.org/research/briscv/simulator/simulator.html) for debug

## Tiny Tech promise
Similar to [tinygrad](https://github.com/geohot/tinygrad),
[micrograd](https://github.com/karpathy/micrograd), and other “tiny tech” projects,
we believe that core technology should be simple and small (in terms of
[LOC](https://en.wikipedia.org/wiki/Source_lines_of_code)). Therefore, we will make sure
that the core of TinyFive (without tests and examples) will always be below 1000 lines.
Keep in mind that simplicity and size (in terms of number of instructions) is a key feature
of [RISC](https://en.wikipedia.org/wiki/Reduced_instruction_set_computer): the "R" in RISC
stands for "reduced" (as opposed to complex CISC). Specifically, the ISA manual of RISC-V
has only ~200 pages while the ARM-32 manual is over 2000 pages long according to Fig. 1.6 of
the [RISC-V Reader](http://riscvbook.com/spanish/guia-practica-de-risc-v-1.0.5.pdf).

<p align="center">
  <img src="https://github.com/OpenMachine-ai/tinyfive/blob/main/logo.jpg">
</p>
