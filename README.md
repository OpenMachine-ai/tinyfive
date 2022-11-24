# TinyFive

<p align="center">
  <img src="https://github.com/OpenMachine-ai/tinyfive/blob/main/logo.jpg">
</p>

TinyFive is a simple RISC-V model and
[ISS](https://en.wikipedia.org/wiki/Instruction_set_simulator) written in Python.

It's useful for running neural networks on RISC-V: TinyFive lets you
simulate your RISC-V assembly code along with your neural network, all
in Python (and without relying on RISC-V toolchains).

### Running in colab notebook
If you don’t have a terminal open right now, you can run TinyFive in
[this colab notebook](https://colab.research.google.com/drive/1KXDPwSJmaOGefh5vAjrediwuiRf3wWa2?usp=sharing).
This is the quickest way to get started and should work on any platform.

### Running on your machine
Clone the repo and Install packages `numpy` and `bitstring` as follows:
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
- 37 of the 40 base instructions (rv32i) are implemented and many of them have
  been tested.
- Remaining work: more testing, add extensions M and F. See TODOs in the code
  for more details.
