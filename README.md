# TinyFive

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FOpenMachine-ai%2Ftinyfive&count_bg=%232EF706&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors+%28today+%2F+total%29&edge_flat=false)](https://hits.seeyoufarm.com)

TinyFive is a simple RISC-V simulation model and
[ISS](https://en.wikipedia.org/wiki/Instruction_set_simulator) written entirely in Python:
- TinyFive is useful for running neural networks on RISC-V: Simulate your RISC-V assembly code along with your neural network all in Python (and without relying on RISC-V toolchains). 
- It's also useful for ML scientists who are using ML/RL for compiler optimization (see [CompilerGym](https://github.com/facebookresearch/CompilerGym/blob/development/README.md)).
- If you want to learn how RISC-V works, TinyFive let's you play with instructions and assembler.
- Fewer than 1000 lines of code (w/o tests and examples).

### Table of content
- [Running in colab notebook](#running-in-colab-notebook)
- [Running on your machine](#running-on-your-machine)
- [Latest status](#latest-status)
- [Comparison](#comparison)
- [References](#references)
- [Tiny Tech promise](#tiny-tech-promise)

### Running in colab notebook
You can run TinyFive in
[this colab notebook](https://colab.research.google.com/drive/1KXDPwSJmaOGefh5vAjrediwuiRf3wWa2?usp=sharing).
This is the quickest way to get started and should work on any machine.

### Running on your machine
Clone the repo and install packages `numpy` and `bitstring` as follows:
```
git clone https://github.com/OpenMachine-ai/tinyfive.git
cd tinyfive
pip3 install --upgrade pip
pip3 install numpy bitstring
```

To run the examples, type:
```
python3 examples.py
```

To run the test suite, type:
```
python3 tests.py
```

### Latest status
- TinyFive is still under construction, many things haven't been implemented and tested yet.
- 37 of the 40 base instructions `RV32I`, all instructions of the M-extension `RV32M`, and
  some of the F-extension `RV32F` are already implemented, and many of them are tested. 
  (The three missing instructions `fence`, `ebreak`, and `ecall` are not applicable here.)
- Remaining work: improve testing, add more extensions, add RV64. See TODOs in
  the code for more details.

### Comparison
The table below compares TinyFive with other ISS projects.

| ISS | Author | Language | Mature? | Extensions | LOC |
| --- | ------ | -------- | ------- | ---------- | --- |
| [TinyFive](https://github.com/OpenMachine-ai/tinyfive)             | OpenMachine          | Python    | No               | I, M          | < 1k |
| [Pydgin](https://github.com/cornell-brg/pydgin)                    | Cornell University   | Python, C | Last update 2016 | A, D, F, I, M | |
| [Spike](https://github.com/riscv-software-src/riscv-isa-sim)       | UC Berkeley          | C, C++    | Yes              | All           | |
| [riscvOVPsim](https://github.com/riscv-ovpsim/imperas-riscv-tests) | Imperas              | C         | Yes              | All           | |
| [Whisper](https://github.com/chipsalliance/SweRV-ISS)              | Western Digital      | C, C++    | Yes | Almost all                 | |
| [Sail Model](https://github.com/riscv/sail-riscv)                  | Cambridge, Edinburgh | Sail, C   | Yes | All                        | |
| [PiMaker/rvc](https://github.com/PiMaker/rvc)                      | PiMaker              | C         |  ?  |                            | |
| [mini-rv32ima](https://github.com/cnlohr/mini-rv32ima)             | Charles Lohr         | C         |  ?  | A, I, M, Zifencei, Zicsr   | < 1k | 

### References
- Official [RISC-V spec](https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf)
- See [this RISC-V card](https://inst.eecs.berkeley.edu/~cs61c/fa18/img/riscvcard.pdf)
 for a brief description of most instructions. See also the 
 [RISC-V reference card](http://riscvbook.com/greencard-20181213.pdf).
- Book "The RISC-V Reader: An Open Architecture Atlas" by David Patterson and Andrew Waterman
(2 of the 4 founders of RISC-V). Appendix A of this book defines all instructions.
The Spanish version of this book is
[available for free](http://riscvbook.com/spanish/guia-practica-de-risc-v-1.0.5.pdf),
other free versions are [available here](http://riscvbook.com).
- [Online simulator](https://ascslab.org/research/briscv/simulator/simulator.html) for debug

### Tiny Tech promise
Similar to [tinygrad](https://github.com/geohot/tinygrad),
[micrograd](https://github.com/karpathy/micrograd), and other “tiny tech” projects,
we believe that core technology should be simple and small (in terms of
[LOC](https://en.wikipedia.org/wiki/Source_lines_of_code)). Therefore, we will make sure
that the core of TinyFive (without tests and examples) will always be below 1000 lines.
Keep in mind that simplicity and size (in terms of number of instructions) is a key feature
of [RISC](https://en.wikipedia.org/wiki/Reduced_instruction_set_computer): the "R" in RISC
stands for "reduced" (as opposed to complex CISC).

<p align="center">
  <img src="https://github.com/OpenMachine-ai/tinyfive/blob/main/logo.jpg">
</p>
