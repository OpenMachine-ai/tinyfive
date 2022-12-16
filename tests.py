from tinyfive import *
import numpy as np

# this file performs the following tests:
#   1) floating point tests from the official RISC-V unit tests
#   2) ALU instruction tests (arithmetic, logic, shift) derived from the
#      official RISC-V test suite
#   3) a few basic load/store tests
#
# source for (1):
#   The official RISC-V unit tests are here:
#   https://github.com/riscv-software-src/riscv-tests
#   Specifically, the following files and folders have been used:
#     isa/macros/scalar/test_macros.h : macros TEST_FP_OP*_S
#     isa/rv64uf : testcase from various *.s files
#   The original copyright and license notice of the RISC-V
#   unit tests is as follows:
#     Copyright (c) 2012-2015, The Regents of the University of California (Regents).
#     All Rights Reserved.
#
# source for (2):
#   The official RISC-V test-suite is here:
#   https://github.com/riscv-non-isa/riscv-arch-test/blob/main/riscv-test-suite
#   Specifically, the following files and folders have been used:
#     env/arch_test.h : macros TEST_RR_OP and TEST_IMM_OP
#     rv32i_m/I/src : testcases from various *.s files
#   The original copyright and license notice of the official RISC-V
#   test-suite is as follows:
#     // Copyright (c) 2020. RISC-V International. All rights reserved.
#     // SPDX-License-Identifier: BSD-3-Clause

m = tinyfive(mem_size=1000)  # instantiate RISC-V machine with 1KB of memory

#-------------------------------------------------------------------------------
# functions for (1) and (2)
#-------------------------------------------------------------------------------
def i32(x): return np.int32(x)   # convert to 32-bit signed

def check(inst, res, correctval):
  if (res != i32(correctval)):
    print('FAIL ' + inst + ' res = ' + str(res) + ' ref = ' + str(i32(correctval)))
  return int(res != i32(correctval))

def test_rr(m, inst, rd, rs1, rs2, correctval, val1, val2):
  """similar to TEST_RR_OP from file arch_test.h of the official test suite"""
  m.clear_cpu()
  m.x[rs1] = val1
  m.x[rs2] = val2
  m.enc(inst,  rd, rs1, rs2)
  m.exe(start=0, instructions=1)
  return check(inst, m.x[rd], correctval)

def test_imm(m, inst, rd, rs1, correctval, val1, imm):
  """similar to TEST_IMM_OP from arch_test.h of the official test suite"""
  m.clear_cpu()
  m.x[rs1] = val1
  m.enc(inst, rd, rs1, imm)
  m.exe(start=0, instructions=1)
  return check(inst, m.x[rd], correctval)

""" TODO: for debug only, eventually remove it
def debug_test_rr(m, inst, rd, rs1, rs2, correctval, val1, val2):
  m.clear_cpu()
  m.x[rs1] = val1
  m.x[rs2] = val2
  if   inst == 'add' : m.ADD (rd, rs1, rs2)
  elif inst == 'sub' : m.SUB (rd, rs1, rs2)
  elif inst == 'sll' : m.SLL (rd, rs1, rs2)
  elif inst == 'slt' : m.SLT (rd, rs1, rs2)
  elif inst == 'sltu': m.SLTU(rd, rs1, rs2)
  elif inst == 'xor' : m.XOR (rd, rs1, rs2)
  elif inst == 'srl' : m.SRL (rd, rs1, rs2)
  elif inst == 'sra' : m.SRA (rd, rs1, rs2)
  elif inst == 'or'  : m.OR  (rd, rs1, rs2)
  elif inst == 'and' : m.AND (rd, rs1, rs2)
  return check(inst, m.x[rd], correctval)

def debug_test_imm(m, inst, rd, rs1, correctval, val1, imm):
  m.clear_cpu()
  m.x[rs1] = val1
  if   inst == 'addi' : m.ADDI (rd, rs1, imm)
  elif inst == 'slti' : m.SLTI (rd, rs1, imm)
  elif inst == 'sltiu': m.SLTIU(rd, rs1, imm)
  elif inst == 'xori' : m.XORI (rd, rs1, imm)
  elif inst == 'ori'  : m.ORI  (rd, rs1, imm)
  elif inst == 'andi' : m.ANDI (rd, rs1, imm)
  elif inst == 'slli' : m.SLLI (rd, rs1, imm)
  elif inst == 'srli' : m.SRLI (rd, rs1, imm)
  elif inst == 'srai' : m.SRAI (rd, rs1, imm)
  return check(inst, m.x[rd], correctval)
"""

#-------------------------------------------------------------------------------
# functions for floating point

def check_fp(inst, res, correctval):
  error = not ((res == np.float32(correctval)) or
               (np.isnan(res) and np.isnan(correctval)))
  if error:
    print('FAIL ' + inst + ' res = ' + str(res) + ' ref = ' +
          str(np.float32(correctval)))
  return int(error)

def test_fp(m, inst, correctval, val1, val2=0.0, val3=0.0):
  """similar to TEST_FP_OP*_S from isa/macros/scalar/test_macros.h"""
  m.clear_cpu()
  m.f[1] = val1
  m.f[2] = val2
  m.f[3] = val3
  rd = 10
  if inst in ['fmv.w.x', 'fcvt.s.w', 'fcvt.s.wu']:
    m.x[1] = val1
  m.enc(inst, rd, 1, 2, 3)
  m.exe(start=0, instructions=1)
  if inst in ['feq.s', 'fle.s', 'flt.s', 'fcvt.w.s', 'fcvt.wu.s', 'fclass.s']:
    return check(inst, m.x[rd], correctval)
  else:
    return check_fp(inst, m.f[rd], correctval)
  # TODO: add check for floating point flags, as done in TEST_FP_OP*_S

""" TODO: for debug only, eventually remove it
def debug_test_fp(m, inst, correctval, val1, val2=0.0, val3=0.0):
  m.clear_cpu()
  m.f[1] = val1
  m.f[2] = val2
  m.f[3] = val3
  rd = 10
  if inst in ['fmv.w.x', 'fcvt.s.w', 'fcvt.s.wu']:
    m.x[1] = val1
  if   inst == 'fadd.s'   : m.FADD_S   (rd, 1, 2)
  elif inst == 'fsub.s'   : m.FSUB_S   (rd, 1, 2)
  elif inst == 'fmul.s'   : m.FMUL_S   (rd, 1, 2)
  elif inst == 'fdiv.s'   : m.FDIV_S   (rd, 1, 2)
  elif inst == 'fsqrt.s'  : m.FSQRT_S  (rd, 1)
  elif inst == 'fmin.s'   : m.FMIN_S   (rd, 1, 2)
  elif inst == 'fmax.s'   : m.FMAX_S   (rd, 1, 2)
  elif inst == 'fmadd.s'  : m.FMADD_S  (rd, 1, 2, 3)
  elif inst == 'fmsub.s'  : m.FMSUB_S  (rd, 1, 2, 3)
  elif inst == 'fnmadd.s' : m.FNMADD_S (rd, 1, 2, 3)
  elif inst == 'fnmsub.s' : m.FNMSUB_S (rd, 1, 2, 3)
  elif inst == 'feq.s'    : m.FEQ_S    (rd, 1, 2)
  elif inst == 'fle.s'    : m.FLE_S    (rd, 1, 2)
  elif inst == 'flt.s'    : m.FLT_S    (rd, 1, 2)
  elif inst == 'fcvt.s.w' : m.FCVT_S_W (rd, 1)
  elif inst == 'fcvt.s.wu': m.FCVT_S_WU(rd, 1)
  elif inst == 'fcvt.w.s' : m.FCVT_W_S (rd, 1)
  elif inst == 'fcvt.wu.s': m.FCVT_WU_S(rd, 1)
  elif inst == 'fclass.s' : m.FCLASS_S (rd, 1)
  if inst in ['feq.s', 'fle.s', 'flt.s', 'fcvt.w.s', 'fcvt.wu.s', 'fclass.s']:
    return check(inst, m.x[rd], correctval)
  else:
    return check_fp(inst, m.f[rd], correctval)
"""
qNaN = m.b2f(0x7fc00000)
sNaN = m.b2f(0x7f800001)
NaN  = np.nan

#-------------------------------------------------------------------------------
# floating point unit tests
#-------------------------------------------------------------------------------

# fadd, fsub, fmul
err = 0
err += test_fp(m, 'fadd.s',           3.5,        2.5,        1.0)
err += test_fp(m, 'fadd.s',         -1234,    -1235.1,        1.1)
err += test_fp(m, 'fadd.s',    3.14159265, 3.14159265, 0.00000001)
err += test_fp(m, 'fsub.s',           1.5,        2.5,        1.0)
err += test_fp(m, 'fsub.s',         -1234,    -1235.1,       -1.1)
err += test_fp(m, 'fsub.s',    3.14159265, 3.14159265, 0.00000001)
err += test_fp(m, 'fmul.s',           2.5,        2.5,        1.0)
err += test_fp(m, 'fmul.s',       1358.61,    -1235.1,       -1.1)
err += test_fp(m, 'fmul.s', 3.14159265e-8, 3.14159265, 0.00000001)
err += test_fp(m, 'fsub.s',        np.nan,     np.inf,     np.inf)

# fmin, fmax
# TODO: fix the tests that are commented out below
err += test_fp(m, 'fmin.s',        1.0,        2.5,        1.0)
err += test_fp(m, 'fmin.s',    -1235.1,    -1235.1,        1.1)
err += test_fp(m, 'fmin.s',    -1235.1,        1.1,    -1235.1)
#err+= test_fp(m, 'fmin.s',    -1235.1,        NaN,    -1235.1)
err += test_fp(m, 'fmin.s', 0.00000001, 3.14159265, 0.00000001)
err += test_fp(m, 'fmin.s',       -2.0,       -1.0,       -2.0)
err += test_fp(m, 'fmax.s',        2.5,        2.5,        1.0)
err += test_fp(m, 'fmax.s',        1.1,    -1235.1,        1.1)
err += test_fp(m, 'fmax.s',        1.1,        1.1,    -1235.1)
#err+= test_fp(m, 'fmax.s',    -1235.1,        NaN,    -1235.1)
err += test_fp(m, 'fmax.s', 3.14159265, 3.14159265, 0.00000001)
err += test_fp(m, 'fmax.s',       -1.0,       -1.0,       -2.0)
#err+= test_fp(m, 'fmax.s',        1.0,       sNaN,        1.0)
err += test_fp(m, 'fmax.s',       qNaN,        NaN,        NaN)
err += test_fp(m, 'fmin.s',       -0.0,       -0.0,        0.0)
err += test_fp(m, 'fmin.s',       -0.0,        0.0,       -0.0)
err += test_fp(m, 'fmax.s',        0.0,       -0.0,        0.0)
err += test_fp(m, 'fmax.s',        0.0,        0.0,       -0.0)

# fdiv, fsqrt
err += test_fp(m, 'fdiv.s',  1.1557273520668288, 3.14159265, 2.71828182)
err += test_fp(m, 'fdiv.s', -0.9991093838555584,      -1234,     1235.1)
err += test_fp(m, 'fdiv.s',          3.14159265, 3.14159265,        1.0)
err += test_fp(m, 'fsqrt.s', 1.7724538498928541, 3.14159265)
err += test_fp(m, 'fsqrt.s',                100,      10000)
err += test_fp(m, 'fsqrt.s',                NaN,       -1.0)
err += test_fp(m, 'fsqrt.s',          13.076696,      171.0)

# fmadd, fnmadd, fmsub, fnmsub
err += test_fp(m, 'fmadd.s',      3.5,  1.0,        2.5,        1.0)
err += test_fp(m, 'fmadd.s',   1236.2, -1.0,    -1235.1,        1.1)
err += test_fp(m, 'fmadd.s',    -12.0,  2.0,       -5.0,       -2.0)
err += test_fp(m, 'fnmadd.s',    -3.5,  1.0,        2.5,        1.0)
err += test_fp(m, 'fnmadd.s', -1236.2, -1.0,    -1235.1,        1.1)
err += test_fp(m, 'fnmadd.s',    12.0,  2.0,       -5.0,       -2.0)
err += test_fp(m, 'fmsub.s',      1.5,  1.0,        2.5,        1.0)
err += test_fp(m, 'fmsub.s',     1234, -1.0,    -1235.1,        1.1)
err += test_fp(m, 'fmsub.s',     -8.0,  2.0,       -5.0,       -2.0)
err += test_fp(m, 'fnmsub.s',    -1.5,  1.0,        2.5,        1.0)
err += test_fp(m, 'fnmsub.s',   -1234, -1.0,    -1235.1,        1.1)
err += test_fp(m, 'fnmsub.s',     8.0,  2.0,       -5.0,       -2.0)

# flw and fsw (adopted from isa/rv32uf/ldst.S)
def test_flw_fsw(m, tdat, tdat_offset):
  m.clear_cpu()
  m.x[a1] = tdat
  m.enc('flw.s', f1, tdat_offset, a1)
  m.enc('fsw.s', f1, 20, a1)
  m.enc('lw',    a0, 20, a1)
  m.exe(start=0, instructions=3)
  ref = m.read_i32(tdat+tdat_offset)
  return check('flw/fsw', m.x[a0], ref)

tdat = 500  # write testdata to address 500
m.write_i32(0xbf800000, tdat)
m.write_i32(0x40000000, tdat+4)
err += test_flw_fsw(m, tdat, 4)
err += test_flw_fsw(m, tdat, 0)

# feq, fle, flt
err += test_fp(m, 'feq.s', 1, -1.36, -1.36)
err += test_fp(m, 'fle.s', 1, -1.36, -1.36)
err += test_fp(m, 'flt.s', 0, -1.36, -1.36)
err += test_fp(m, 'feq.s', 0, -1.37, -1.36)
err += test_fp(m, 'fle.s', 1, -1.37, -1.36)
err += test_fp(m, 'flt.s', 1, -1.37, -1.36)
err += test_fp(m, 'feq.s', 0, NaN, 0)
err += test_fp(m, 'feq.s', 0, NaN, NaN)
err += test_fp(m, 'feq.s', 0, sNaN, 0)
err += test_fp(m, 'flt.s', 0, NaN, 0)
err += test_fp(m, 'flt.s', 0, NaN, NaN)
err += test_fp(m, 'flt.s', 0, sNaN, 0)
err += test_fp(m, 'fle.s', 0, NaN, 0)
err += test_fp(m, 'fle.s', 0, NaN, NaN)
err += test_fp(m, 'fle.s', 0, sNaN, 0)

# fcvt
err += test_fp(m, 'fcvt.s.w',           2.0,  2)
err += test_fp(m, 'fcvt.s.w',          -2.0, -2)
err += test_fp(m, 'fcvt.s.wu',          2.0,  2)
err += test_fp(m, 'fcvt.s.wu',  4.2949673e9, -2)
err += test_fp(m, 'fcvt.w.s',          -1, -1.1)
err += test_fp(m, 'fcvt.w.s',          -1, -1.0)
err += test_fp(m, 'fcvt.w.s',           0, -0.9)
err += test_fp(m, 'fcvt.w.s',           0,  0.9)
err += test_fp(m, 'fcvt.w.s',           1,  1.0)
err += test_fp(m, 'fcvt.w.s',           1,  1.1)
err += test_fp(m, 'fcvt.w.s',      -1<<31, -3e9)
err += test_fp(m, 'fcvt.w.s',   (1<<31)-1,  3e9)
err += test_fp(m, 'fcvt.wu.s',          0, -3.0)
err += test_fp(m, 'fcvt.wu.s',          0, -1.0)
err += test_fp(m, 'fcvt.wu.s',          0, -0.9)
err += test_fp(m, 'fcvt.wu.s',          0,  0.9)
err += test_fp(m, 'fcvt.wu.s',          1,  1.0)
err += test_fp(m, 'fcvt.wu.s',          1,  1.1)
err += test_fp(m, 'fcvt.wu.s',          0, -3e9)
err += test_fp(m, 'fcvt.wu.s', 3000000000,  3e9)

# fmv, fsgnj
def test_fsgn(inst, new_sign, rs1_sign, rs2_sign):
  ref = 0x12345678 | (-new_sign << 31)
  m.clear_cpu()
  m.x[a1] = (rs1_sign << 31) | 0x12345678
  m.x[a2] = -rs2_sign
  m.enc('fmv.w.x', f1, a1)
  m.enc('fmv.w.x', f2, a2)
  m.enc(inst,      f0, f1, f2)
  m.enc('fmv.x.w', a0, f0)
  m.exe(start=0, instructions=4)
  return check('fmv/fsgnj', m.x[a0], ref)

err += test_fsgn('fsgnj.s', 0, 0, 0)
err += test_fsgn('fsgnj.s', 1, 0, 1)
err += test_fsgn('fsgnj.s', 0, 1, 0)
err += test_fsgn('fsgnj.s', 1, 1, 1)

# fclass
err += test_fp(m, 'fclass.s', 1 << 0, m.b2f(0xff800000))
err += test_fp(m, 'fclass.s', 1 << 1, m.b2f(0xbf800000))
err += test_fp(m, 'fclass.s', 1 << 2, m.b2f(0x807fffff))
err += test_fp(m, 'fclass.s', 1 << 3, m.b2f(0x80000000))
err += test_fp(m, 'fclass.s', 1 << 4, m.b2f(0x00000000))
err += test_fp(m, 'fclass.s', 1 << 5, m.b2f(0x007fffff))
err += test_fp(m, 'fclass.s', 1 << 6, m.b2f(0x3f800000))
err += test_fp(m, 'fclass.s', 1 << 7, m.b2f(0x7f800000))
err += test_fp(m, 'fclass.s', 1 << 8, m.b2f(0x7f800001))
err += test_fp(m, 'fclass.s', 1 << 9, m.b2f(0x7fc00000))

print('FP-tests errors: ' + str(err))

#-------------------------------------------------------------------------------
# test ALU instructions
#-------------------------------------------------------------------------------

# the tests below are derived from the *.s files at rv32i_m/I/src of the
# official test-suite. Specifically, the first 10 or so tests are used
# TODO: use all tests. Note that we are currently only testing the ALU
# instructions here because the other tests are harder to adoot, this is a TODO

# add
err = 0
err += test_rr(m, 'add', x24, x4,  x24, 0x80000000, 0x7fffffff, 0x1)
err += test_rr(m, 'add', x28, x10, x10, 0x40000, 0x20000, 0x20000)
err += test_rr(m, 'add', x21, x21, x21, 0xfdfffffe, -0x1000001, -0x1000001)
err += test_rr(m, 'add', x22, x22, x31, 0x3fffe, -0x2, 0x40000)
err += test_rr(m, 'add', x11, x12, x6,  0xaaaaaaac, 0x55555556, 0x55555556)
err += test_rr(m, 'add', x10, x29, x13, 0x80000002, 0x2, -0x80000000)
err += test_rr(m, 'add', x26, x31, x5,  0xffffffef, -0x11, 0x0)
err += test_rr(m, 'add', x7,  x2,  x1,  0xe6666665, 0x66666666, 0x7fffffff)
err += test_rr(m, 'add', x14, x8,  x25, 0x2aaaaaaa, -0x80000000, -0x55555556)
err += test_rr(m, 'add', x1,  x13, x8,  0xfdffffff, 0x0, -0x2000001)
err += test_rr(m, 'add', x0,  x28, x9,  0, 0x1, 0x800000)
err += test_rr(m, 'add', x20, x14, x4,  0x9, 0x7, 0x2)
err += test_rr(m, 'add', x16, x7,  x19, 0xc, 0x8, 0x4)
err += test_rr(m, 'add', x8,  x23, x29, 0x808, 0x800, 0x8)
err += test_rr(m, 'add', x13, x5,  x27, 0x10, 0x0, 0x10)
err += test_rr(m, 'add', x27, x25, x20, 0x55555576, 0x55555556, 0x20)
err += test_rr(m, 'add', x17, x15, x26, 0x2f, -0x11, 0x40)
err += test_rr(m, 'add', x29, x17, x2,  0x7b, -0x5, 0x80)
err += test_rr(m, 'add', x4,  x24, x17, 0x120, 0x20, 0x100)
err += test_rr(m, 'add', x2,  x16, x11, 0x40000200, 0x40000000, 0x200)

# sub
err += test_rr(m, 'sub', x26, x24, x26, 0x5555554e, 0x55555554, 0x6)
err += test_rr(m, 'sub', x23, x17, x17, 0x0, 0x2000000, 0x2000000)
err += test_rr(m, 'sub', x16, x16, x16, 0x0, -0x7, -0x7)
err += test_rr(m, 'sub', x31, x31, x19, 0x99999998, -0x3, 0x66666665)
err += test_rr(m, 'sub', x8,  x23, x14, 0x0, 0x80000, 0x80000)
err += test_rr(m, 'sub', x18, x13, x24, 0x7bffffff, -0x4000001, -0x80000000)
err += test_rr(m, 'sub', x0,  x12, x4,  0, 0x20, 0x0)
err += test_rr(m, 'sub', x10, x22, x9,  0x60000000, -0x20000001, 0x7fffffff)
err += test_rr(m, 'sub', x25, x10, x27, 0xffff, 0x10000, 0x1)
err += test_rr(m, 'sub', x14, x8,  x3,  0xc0000000, -0x80000000, -0x40000000)
err += test_rr(m, 'sub', x29, x25, x30, 0xffe00000, 0x0, 0x200000)
err += test_rr(m, 'sub', x15, x18, x8,  0x7fffdfff, 0x7fffffff, 0x2000)
err += test_rr(m, 'sub', x3,  x14, x15, 0xfffffff1, 0x1, 0x10)
err += test_rr(m, 'sub', x13, x26, x29, 0xfffffff7, -0x7, 0x2)
err += test_rr(m, 'sub', x19, x21, x31, 0xfffffffe, 0x2, 0x4)
err += test_rr(m, 'sub', x11, x30, x5,  0xfefffff7, -0x1000001, 0x8)
err += test_rr(m, 'sub', x30, x28, x7,  0xaaaaaa8a, -0x55555556, 0x20)
err += test_rr(m, 'sub', x7,  x9,  x10, 0xffffff7f, -0x41, 0x40)
err += test_rr(m, 'sub', x17, x0,  x18, 0xffffff80, 0x0, 0x80)
err += test_rr(m, 'sub', x27, x2,  x12, 0xbfffff00, -0x40000000, 0x100)

# sll
err += test_rr(m, 'sll', x28, x16, x28, 0xfffdfc00, -0x81, 0xa)
err += test_rr(m, 'sll', x0,  x21, x21, 0, 0x5, 0x5)
err += test_rr(m, 'sll', x18, x18, x18, 0x80000000, -0x8001, -0x8001)
err += test_rr(m, 'sll', x5,  x5,  x13, 0x7, 0x7, 0x0)
err += test_rr(m, 'sll', x23, x22, x12, 0x180, 0x6, 0x6)
err += test_rr(m, 'sll', x6,  x19, x0,  0x80000000, -0x80000000, 0x0)
err += test_rr(m, 'sll', x13, x25, x24, 0x0, 0x0, 0x4)
err += test_rr(m, 'sll', x16, x12, x26, 0xffe00000, 0x7fffffff, 0x15)
err += test_rr(m, 'sll', x20, x6,  x14, 0x10, 0x1, 0x4)
err += test_rr(m, 'sll', x22, x14, x1,  0x0, 0x2, 0x1f)
err += test_rr(m, 'sll', x21, x29, x7,  0x10, 0x4, 0x2)
err += test_rr(m, 'sll', x4,  x31, x10, 0x0, 0x8, 0x1f)
err += test_rr(m, 'sll', x7,  x17, x20, 0x0, 0x10, 0x1f)
err += test_rr(m, 'sll', x12, x20, x11, 0x10000000, 0x20, 0x17)
err += test_rr(m, 'sll', x3,  x11, x22, 0x800000, 0x40, 0x11)
err += test_rr(m, 'sll', x24, x0,  x30, 0x0, 0x0, 0x10)
err += test_rr(m, 'sll', x8,  x3,  x31, 0x0, 0x100, 0x1b)
err += test_rr(m, 'sll', x10, x27, x17, 0x200, 0x200, 0x0)
err += test_rr(m, 'sll', x11, x10, x19, 0x0, 0x400, 0x1b)
err += test_rr(m, 'sll', x1,  x8,  x27, 0x20000, 0x800, 0x6)

# slt
err += test_rr(m, 'slt', x26, x18, x26, 0x0, 0x66666667, 0x66666667)
err += test_rr(m, 'slt', x1,  x20, x20, 0x0, 0x33333334, 0x33333334)
err += test_rr(m, 'slt', x21, x21, x21, 0x0, -0x4001, -0x4001)
err += test_rr(m, 'slt', x15, x15, x27, 0x1, -0x201, 0x5)
err += test_rr(m, 'slt', x7,  x5,  x18, 0x0, 0x33333334, -0x80000000)
err += test_rr(m, 'slt', x17, x25, x19, 0x0, 0x8000000, 0x0)
err += test_rr(m, 'slt', x23, x9,  x31, 0x1, 0x20000, 0x7fffffff)
err += test_rr(m, 'slt', x11, x2,  x15, 0x1, -0x20001, 0x1)
err += test_rr(m, 'slt', x22, x28, x13, 0x1, -0x80000000, 0x400)
err += test_rr(m, 'slt', x30, x10, x7,  0x1, 0x0, 0x8)
err += test_rr(m, 'slt', x5,  x24, x22, 0x0, 0x7fffffff, -0x101)
err += test_rr(m, 'slt', x24, x8,  x3,  0x0, 0x1, -0x201)
err += test_rr(m, 'slt', x2,  x26, x9,  0x0, 0x10000000, 0x2)
err += test_rr(m, 'slt', x8,  x4,  x14, 0x0, 0x8000, 0x4)
err += test_rr(m, 'slt', x25, x6,  x4,  0x1, -0x1001, 0x10)
err += test_rr(m, 'slt', x14, x31, x25, 0x0, 0x1000000, 0x20)
err += test_rr(m, 'slt', x20, x14, x8,  0x1, 0x5, 0x40)
err += test_rr(m, 'slt', x19, x7,  x11, 0x1, 0x0, 0x80)
err += test_rr(m, 'slt', x9,  x19, x29, 0x1, 0x5, 0x100)
err += test_rr(m, 'slt', x6,  x3,  x23, 0x0, 0x400000, 0x200)

# sltu
err += test_rr(m, 'sltu', x31, x0,  x31, 0x1, 0x0, 0xfffffffe)
err += test_rr(m, 'sltu', x5,  x19, x19, 0x0, 0x100000, 0x100000)
err += test_rr(m, 'sltu', x25, x25, x25, 0x0, 0x40000000, 0x40000000)
err += test_rr(m, 'sltu', x14, x14, x24, 0x1, 0xfffffffe, 0xffffffff)
err += test_rr(m, 'sltu', x12, x17, x13, 0x0, 0x1, 0x1)
err += test_rr(m, 'sltu', x24, x26, x18, 0x1, 0x0, 0xb)
err += test_rr(m, 'sltu', x19, x5,  x14, 0x0, 0xffffffff, 0x0)
err += test_rr(m, 'sltu', x0,  x3,  x22, 0, 0x4, 0x2)
err += test_rr(m, 'sltu', x20, x23, x29, 0x0, 0xf7ffffff, 0x4)
err += test_rr(m, 'sltu', x10, x4,  x6,  0x0, 0x11, 0x8)
err += test_rr(m, 'sltu', x1,  x12, x17, 0x0, 0x7fffffff, 0x10)
err += test_rr(m, 'sltu', x6,  x30, x8,  0x0, 0x2000000, 0x20)
err += test_rr(m, 'sltu', x3,  x21, x16, 0x0, 0xfffff7ff, 0x40)
err += test_rr(m, 'sltu', x17, x29, x26, 0x0, 0x400, 0x80)
err += test_rr(m, 'sltu', x28, x18, x10, 0x1, 0xd, 0x100)
err += test_rr(m, 'sltu', x11, x2,  x28, 0x1, 0x4, 0x200)
err += test_rr(m, 'sltu', x29, x8,  x30, 0x0, 0xffffffbf, 0x400)
err += test_rr(m, 'sltu', x7,  x22, x11, 0x1, 0x80, 0x800)
err += test_rr(m, 'sltu', x9,  x15, x4,  0x0, 0x200000, 0x1000)
err += test_rr(m, 'sltu', x13, x28, x12, 0x1, 0x80, 0x2000)

# xor
err += test_rr(m, 'xor', x24, x27, x24, 0x66666666, 0x66666665, 0x3)
err += test_rr(m, 'xor', x10, x13, x13, 0x0, 0x5, 0x5)
err += test_rr(m, 'xor', x23, x23, x23, 0x0, -0x4001, -0x4001)
err += test_rr(m, 'xor', x28, x28, x14, 0xffffffb7, -0x41, 0x8)
err += test_rr(m, 'xor', x18, x1,  x2,  0x0, -0x1, -0x1)
err += test_rr(m, 'xor', x19, x5,  x22, 0x80400000, 0x400000, -0x80000000)
err += test_rr(m, 'xor', x13, x26, x12, 0xffffffef, -0x11, 0x0)
err += test_rr(m, 'xor', x4,  x12, x11, 0xd5555555, -0x55555556, 0x7fffffff)
err += test_rr(m, 'xor', x17, x19, x30, 0x0, 0x1, 0x1)
err += test_rr(m, 'xor', x3,  x11, x1,  0x6fffffff, -0x80000000, -0x10000001)
err += test_rr(m, 'xor', x8,  x24, x29, 0xffff4afc, 0x0, -0xb504)
err += test_rr(m, 'xor', x9,  x0,  x18, 0x1000, 0x0, 0x1000)
err += test_rr(m, 'xor', x26, x10, x6,  0x80002, 0x80000, 0x2)
err += test_rr(m, 'xor', x30, x22, x31, 0xffffffdb, -0x21, 0x4)
err += test_rr(m, 'xor', x16, x8,  x0,  0x6, 0x6, 0x0)
err += test_rr(m, 'xor', x15, x16, x27, 0x25, 0x5, 0x20)
err += test_rr(m, 'xor', x31, x3,  x26, 0xffdfffbf, -0x200001, 0x40)
err += test_rr(m, 'xor', x14, x9,  x25, 0xffff7f7f, -0x8001, 0x80)
err += test_rr(m, 'xor', x6,  x30, x10, 0x55555456, 0x55555556, 0x100)
err += test_rr(m, 'xor', x7,  x2,  x9,  0xffffbdff, -0x4001, 0x200)

# srl
err += test_rr(m, 'srl', x11, x26, x11, 0x1ff7f, -0x400001, 0xf)
err += test_rr(m, 'srl', x12, x31, x31, 0x155, 0x55555556, 0x55555556)
err += test_rr(m, 'srl', x7,  x7,  x7,  0x1, -0x1, -0x1)
err += test_rr(m, 'srl', x18, x18, x12, 0x100, 0x100, 0x0)
err += test_rr(m, 'srl', x8,  x14, x3,  0x0, 0x9, 0x9)
err += test_rr(m, 'srl', x20, x21, x22, 0x80000, -0x80000000, 0xc)
err += test_rr(m, 'srl', x30, x4,  x17, 0x0, 0x0, 0xf)
err += test_rr(m, 'srl', x6,  x1,  x4,  0x1, 0x7fffffff, 0x1e)
err += test_rr(m, 'srl', x15, x0,  x21, 0x0, 0x0, 0x1d)
err += test_rr(m, 'srl', x5,  x28, x23, 0x0, 0x2, 0x6)
err += test_rr(m, 'srl', x4,  x9,  x30, 0x0, 0x4, 0xa)
err += test_rr(m, 'srl', x10, x13, x29, 0x2, 0x8, 0x2)
err += test_rr(m, 'srl', x25, x2,  x16, 0x0, 0x10, 0x7)
err += test_rr(m, 'srl', x19, x11, x28, 0x0, 0x20, 0x8)
err += test_rr(m, 'srl', x23, x10, x6,  0x0, 0x40, 0xb)
err += test_rr(m, 'srl', x0,  x3,  x1,  0, 0x80, 0x6)
err += test_rr(m, 'srl', x14, x6,  x8,  0x0, 0x200, 0xe)
err += test_rr(m, 'srl', x9,  x20, x27, 0x0, 0x400, 0x1d)
err += test_rr(m, 'srl', x16, x30, x18, 0x0, 0x800, 0xd)
err += test_rr(m, 'srl', x24, x22, x10, 0x10, 0x1000, 0x8)

# sra
err += test_rr(m, 'sra', x27, x16, x27, -0x800000, -0x80000000, 0x8)
err += test_rr(m, 'sra', x16, x12, x12, 0x2000000, 0x2000000, 0x2000000)
err += test_rr(m, 'sra', x1,  x1,  x1,  -0x1, -0x801, -0x801)
err += test_rr(m, 'sra', x13, x13, x19, 0x33333333, 0x33333333, 0x0)
err += test_rr(m, 'sra', x8,  x28, x2,  0x0, 0x6, 0x6)
err += test_rr(m, 'sra', x19, x26, x31, 0x0, 0x0, 0x3)
err += test_rr(m, 'sra', x29, x14, x28, 0x1fff, 0x7fffffff, 0x12)
err += test_rr(m, 'sra', x12, x10, x26, 0x0, 0x1, 0x2)
err += test_rr(m, 'sra', x15, x30, x16, 0x0, 0x2, 0x4)
err += test_rr(m, 'sra', x6,  x24, x0,  0x4, 0x4, 0x0)
err += test_rr(m, 'sra', x3,  x21, x15, 0x0, 0x8, 0xa)
err += test_rr(m, 'sra', x10, x15, x30, 0x10, 0x10, 0x0)
err += test_rr(m, 'sra', x22, x18, x4,  0x4, 0x20, 0x3)
err += test_rr(m, 'sra', x11, x19, x17, 0x0, 0x40, 0x17)
err += test_rr(m, 'sra', x0,  x5,  x14, 0, 0x80, 0x8)
err += test_rr(m, 'sra', x5,  x6,  x24, 0x10, 0x100, 0x4)
err += test_rr(m, 'sra', x31, x7,  x18, 0x0, 0x200, 0x1f)
err += test_rr(m, 'sra', x30, x31, x21, 0x0, 0x400, 0x10)
err += test_rr(m, 'sra', x2,  x4,  x11, 0x0, 0x800, 0x13)
err += test_rr(m, 'sra', x9,  x22, x25, 0x400, 0x1000, 0x2)

# or
err += test_rr(m, 'or', x26, x8,  x26, 0x100010, 0x100000, 0x10)
err += test_rr(m, 'or', x17, x6,  x6,  0x2, 0x2, 0x2)
err += test_rr(m, 'or', x31, x31, x31, 0xefffffff, -0x10000001, -0x10000001)
err += test_rr(m, 'or', x27, x27, x29, 0xfffff7ff, -0x801, 0x400000)
err += test_rr(m, 'or', x18, x30, x19, 0xffefffff, -0x100001, -0x100001)
err += test_rr(m, 'or', x9,  x21, x14, 0x80020000, 0x20000, -0x80000000)
err += test_rr(m, 'or', x4,  x26, x24, 0xffffdfff, -0x2001, 0x0)
err += test_rr(m, 'or', x30, x9,  x8,  0x7fffffff, 0x0, 0x7fffffff)
err += test_rr(m, 'or', x8,  x23, x7,  0xff7fffff, -0x800001, 0x1)
err += test_rr(m, 'or', x22, x12, x0,  0x80000000, -0x80000000, 0x0)
err += test_rr(m, 'or', x28, x10, x30, 0x7fffffff, 0x7fffffff, 0x40)
err += test_rr(m, 'or', x16, x18, x21, 0x55555555, 0x1, 0x55555554)
err += test_rr(m, 'or', x12, x14, x17, 0x1002, 0x1000, 0x2)
err += test_rr(m, 'or', x15, x19, x16, 0xff7fffff, -0x800001, 0x4)
err += test_rr(m, 'or', x7,  x4,  x2,  0xfffffbff, -0x401, 0x8)
err += test_rr(m, 'or', x11, x2,  x22, 0x7fffffff, 0x7fffffff, 0x20)
err += test_rr(m, 'or', x25, x28, x15, 0xfffffdff, -0x201, 0x80)
err += test_rr(m, 'or', x6,  x25, x1,  0xb504, 0xb504, 0x100)
err += test_rr(m, 'or', x20, x17, x10, 0x204, 0x4, 0x200)
err += test_rr(m, 'or', x5,  x20, x23, 0xffefffff, -0x100001, 0x400)

# and
err += test_rr(m, 'and', x25, x24, x25, 0x0, 0x4000, 0x7)
err += test_rr(m, 'and', x18, x3,  x3,  0x800, 0x800, 0x800)
err += test_rr(m, 'and', x19, x19, x19, 0xfffffffd, -0x3, -0x3)
err += test_rr(m, 'and', x5,  x5,  x14, 0x7fffffff, -0x1, 0x7fffffff)
err += test_rr(m, 'and', x20, x23, x16, 0x5, 0x5, 0x5)
err += test_rr(m, 'and', x30, x20, x2,  0x0, 0x2, -0x80000000)
err += test_rr(m, 'and', x13, x7,  x24, 0x0, 0x33333333, 0x0)
err += test_rr(m, 'and', x10, x30, x27, 0x1, -0x40000001, 0x1)
err += test_rr(m, 'and', x22, x28, x18, 0x0, -0x80000000, 0x800)
err += test_rr(m, 'and', x0,  x2,  x15, 0, 0x0, 0x200)
err += test_rr(m, 'and', x12, x25, x26, 0x55555555, 0x7fffffff, 0x55555555)
err += test_rr(m, 'and', x2,  x1,  x31, 0x0, 0x1, 0x55555554)
err += test_rr(m, 'and', x14, x27, x11, 0x0, 0x40000, 0x2)
err += test_rr(m, 'and', x4,  x31, x23, 0x4, -0x20001, 0x4)
err += test_rr(m, 'and', x27, x21, x9,  0x8, -0x55555555, 0x8)
err += test_rr(m, 'and', x23, x26, x7,  0x0, 0x400, 0x10)
err += test_rr(m, 'and', x24, x9,  x20, 0x20, -0x8, 0x20)
err += test_rr(m, 'and', x26, x15, x13, 0x40, -0x101, 0x40)
err += test_rr(m, 'and', x17, x12, x4,  0x80, -0x2000001, 0x80)
err += test_rr(m, 'and', x8,  x4,  x17, 0x0, 0x66666665, 0x100)

# addi
err += test_imm(m, 'addi', x7,  x20, 0x1ffff800, 0x20000000, -0x800)
err += test_imm(m, 'addi', x3,  x3,  0x400, 0x400, 0x0)
err += test_imm(m, 'addi', x22, x4,  0x5fe, -0x201, 0x7ff)
err += test_imm(m, 'addi', x11, x30, 0x1, 0x0, 0x1)
err += test_imm(m, 'addi', x31, x27, 0x80000010, -0x80000000, 0x10)
err += test_imm(m, 'addi', x30, x17, 0x80000005, 0x7fffffff, 0x6)
err += test_imm(m, 'addi', x28, x18, 0x5, 0x1, 0x4)
err += test_imm(m, 'addi', x6,  x13, 0xa, 0x5, 0x5)
err += test_imm(m, 'addi', x16, x10, 0xaaaaaa8a, -0x55555555, -0x21)
err += test_imm(m, 'addi', x21, x9,  0xfffffff1, -0x11, 0x2)
err += test_imm(m, 'addi', x2,  x7,  0xb50d, 0xb505, 0x8)
err += test_imm(m, 'addi', x18, x22, 0xffff4b1c, -0xb504, 0x20)
err += test_imm(m, 'addi', x0,  x29, 0, -0x200001, 0x40)
err += test_imm(m, 'addi', x13, x25, 0x85, 0x5, 0x80)
err += test_imm(m, 'addi', x29, x11, 0xfe0000ff, -0x2000001, 0x100)
err += test_imm(m, 'addi', x8,  x6,  0x210, 0x10, 0x200)
err += test_imm(m, 'addi', x4,  x19, 0x402, 0x2, 0x400)
err += test_imm(m, 'addi', x10, x12, 0x55555552, 0x55555554, -0x2)
err += test_imm(m, 'addi', x26, x31, 0x55555552, 0x55555555, -0x3)
err += test_imm(m, 'addi', x15, x26, 0x5555554f, 0x55555554, -0x5)

# slti
err += test_imm(m, 'slti', x12, x25, 0x0, -0x81, -0x800)
err += test_imm(m, 'slti', x5,  x5,  0x1, -0x1001, 0x0)
err += test_imm(m, 'slti', x28, x4,  0x1, -0x40000000, 0x7ff)
err += test_imm(m, 'slti', x15, x31, 0x1, -0x11, 0x1)
err += test_imm(m, 'slti', x13, x1,  0x1, -0x80000000, 0x3)
err += test_imm(m, 'slti', x1,  x15, 0x1, 0x0, 0x2)
err += test_imm(m, 'slti', x9,  x16, 0x0, 0x7fffffff, -0x8)
err += test_imm(m, 'slti', x31, x11, 0x0, 0x1, -0x400)
err += test_imm(m, 'slti', x27, x14, 0x0, 0x10, 0x10)
err += test_imm(m, 'slti', x26, x12, 0x0, 0x33333334, 0x4)
err += test_imm(m, 'slti', x4,  x17, 0x0, 0x3fffffff, 0x8)
err += test_imm(m, 'slti', x10, x18, 0x1, -0x2001, 0x20)
err += test_imm(m, 'slti', x21, x27, 0x1, 0x3, 0x40)
err += test_imm(m, 'slti', x8,  x3,  0x0, 0x55555554, 0x80)
err += test_imm(m, 'slti', x0,  x7,  0, 0x55555554, 0x100)
err += test_imm(m, 'slti', x24, x22, 0x1, -0x55555556, 0x200)
err += test_imm(m, 'slti', x18, x24, 0x0, 0x4000, 0x400)
err += test_imm(m, 'slti', x25, x6,  0x1, -0x401, -0x2)
err += test_imm(m, 'slti', x23, x21, 0x0, 0x66666667, -0x3)
err += test_imm(m, 'slti', x7,  x0,  0x0, 0x0, -0x5)

# sltiu
err += test_imm(m, 'sltiu', x28, x23, 0x0, 0x400, 0x0)
err += test_imm(m, 'sltiu', x2,  x2,  0x1, 0x800, 0xfff)
err += test_imm(m, 'sltiu', x25, x3,  0x0, 0x4, 0x1)
err += test_imm(m, 'sltiu', x11, x19, 0x1, 0x0, 0x6)
err += test_imm(m, 'sltiu', x15, x14, 0x0, 0xffffffff, 0x2c)
err += test_imm(m, 'sltiu', x4,  x13, 0x0, 0x1, 0x0)
err += test_imm(m, 'sltiu', x3,  x26, 0x0, 0xd, 0xd)
err += test_imm(m, 'sltiu', x29, x20, 0x0, 0xaaaaaaaa, 0x2)
err += test_imm(m, 'sltiu', x16, x27, 0x0, 0x7fffffff, 0x4)
err += test_imm(m, 'sltiu', x20, x17, 0x0, 0xfeffffff, 0x8)
err += test_imm(m, 'sltiu', x8,  x31, 0x0, 0x800, 0x10)
err += test_imm(m, 'sltiu', x23, x24, 0x1, 0xc, 0x20)
err += test_imm(m, 'sltiu', x26, x25, 0x0, 0x55555555, 0x40)
err += test_imm(m, 'sltiu', x6,  x22, 0x0, 0x80000, 0x80)
err += test_imm(m, 'sltiu', x5,  x12, 0x0, 0xfffffff7, 0x100)
err += test_imm(m, 'sltiu', x1,  x9,  0x0, 0x80000000, 0x200)
err += test_imm(m, 'sltiu', x10, x28, 0x0, 0xfffbffff, 0x400)
err += test_imm(m, 'sltiu', x31, x21, 0x1, 0x0, 0x800)
err += test_imm(m, 'sltiu', x21, x0,  0x1, 0x0, 0xffe)
err += test_imm(m, 'sltiu', x14, x11, 0x1, 0x12, 0xffd)

# xori
err += test_imm(m, 'xori', x10, x24, 0xcccccb34, 0x33333334, -0x800)
err += test_imm(m, 'xori', x18, x18, 0x4, 0x4, 0x0)
err += test_imm(m, 'xori', x24, x15, 0xfffff803, -0x4, 0x7ff)
err += test_imm(m, 'xori', x20, x11, 0x3, 0x2, 0x1)
err += test_imm(m, 'xori', x21, x7,  0x80000554, -0x80000000, 0x554)
err += test_imm(m, 'xori', x27, x17, 0xfffffbff, 0x0, -0x401)
err += test_imm(m, 'xori', x1,  x22, 0x80000009, 0x7fffffff, -0xa)
err += test_imm(m, 'xori', x22, x20, 0x5, 0x1, 0x4)
err += test_imm(m, 'xori', x31, x19, 0x0, -0x201, -0x201)
err += test_imm(m, 'xori', x5,  x9,  0xffffffdd, -0x21, 0x2)
err += test_imm(m, 'xori', x29, x28, 0x80000008, -0x80000000, 0x8)
err += test_imm(m, 'xori', x4,  x30, 0xbfffffef, -0x40000001, 0x10)
err += test_imm(m, 'xori', x8,  x27, 0x7fffffdf, 0x7fffffff, 0x20)
err += test_imm(m, 'xori', x25, x3,  0x66666626, 0x66666666, 0x40)
err += test_imm(m, 'xori', x17, x31, 0xfff7ff7f, -0x80001, 0x80)
err += test_imm(m, 'xori', x16, x29, 0xffff4bfc, -0xb504, 0x100)
err += test_imm(m, 'xori', x6,  x4,  0x200, 0x0, 0x200)
err += test_imm(m, 'xori', x3,  x14, 0xffeffbff, -0x100001, 0x400)
err += test_imm(m, 'xori', x15, x12, 0x7, -0x7, -0x2)
err += test_imm(m, 'xori', x9,  x21, 0xfffffff8, 0x5, -0x3)

# ori
err += test_imm(m, 'ori', x22, x5,  0xfffffdff, -0x201, -0x800)
err += test_imm(m, 'ori', x27, x27, 0x0, 0x0, 0x0)
err += test_imm(m, 'ori', x8,  x17, 0x333337ff, 0x33333334, 0x7ff)
err += test_imm(m, 'ori', x1,  x20, 0xffff4afd, -0xb504, 0x1)
err += test_imm(m, 'ori', x19, x12, 0x8000002d, -0x80000000, 0x2d)
err += test_imm(m, 'ori', x3,  x8,  0x7fffffff, 0x7fffffff, 0x555)
err += test_imm(m, 'ori', x26, x28, 0x667, 0x1, 0x667)
err += test_imm(m, 'ori', x23, x16, 0xffffffff, 0x7, -0x7)
err += test_imm(m, 'ori', x31, x25, 0x40002, 0x40000, 0x2)
err += test_imm(m, 'ori', x11, x23, 0x20000004, 0x20000000, 0x4)
err += test_imm(m, 'ori', x17, x14, 0xfffffdff, -0x201, 0x8)
err += test_imm(m, 'ori', x7,  x31, 0x12, 0x2, 0x10)
err += test_imm(m, 'ori', x4,  x21, 0x8020, 0x8000, 0x20)
err += test_imm(m, 'ori', x5,  x15, 0x840, 0x800, 0x40)
err += test_imm(m, 'ori', x25, x30, 0xfffbffff, -0x40001, 0x80)
err += test_imm(m, 'ori', x30, x11, 0xfffffffb, -0x5, 0x100)
err += test_imm(m, 'ori', x10, x4,  0xfff7ffff, -0x80001, 0x200)
err += test_imm(m, 'ori', x0,  x13, 0, -0x40000001, 0x400)
err += test_imm(m, 'ori', x6,  x26, 0xffffffff, -0x21, -0x2)
err += test_imm(m, 'ori', x18, x19, 0xffffffff, 0xb503, -0x3)

# andi
err += test_imm(m, 'andi', x10, x22, 0xfffff800, -0x2, -0x800)
err += test_imm(m, 'andi', x25, x25, 0x0, -0x1001, 0x0)
err += test_imm(m, 'andi', x17, x16, 0x7ff, -0x2000001, 0x7ff)
err += test_imm(m, 'andi', x8,  x2,  0x1, -0x20001, 0x1)
err += test_imm(m, 'andi', x30, x28, 0x0, -0x80000000, 0x4)
err += test_imm(m, 'andi', x19, x4,  0x0, 0x0, -0x800)
err += test_imm(m, 'andi', x2,  x10, 0x6, 0x7fffffff, 0x6)
err += test_imm(m, 'andi', x13, x7,  0x0, 0x1, 0x554)
err += test_imm(m, 'andi', x9,  x27, 0x80, 0x80, 0x80)
err += test_imm(m, 'andi', x3,  x17, 0x7fffffd4, 0x7fffffff, -0x2c)
err += test_imm(m, 'andi', x26, x0,  0x0, 0x0, 0x2)
err += test_imm(m, 'andi', x21, x23, 0x0, 0x66666666, 0x8)
err += test_imm(m, 'andi', x14, x6,  0x0, 0x0, 0x10)
err += test_imm(m, 'andi', x22, x5,  0x0, 0x100, 0x20)
err += test_imm(m, 'andi', x29, x8,  0x40, -0x5, 0x40)
err += test_imm(m, 'andi', x23, x12, 0x0, 0x1, 0x100)
err += test_imm(m, 'andi', x6,  x15, 0x200, -0x55555555, 0x200)
err += test_imm(m, 'andi', x11, x29, 0x0, 0x0, 0x400)
err += test_imm(m, 'andi', x1,  x20, 0x66666666, 0x66666667, -0x2)
err += test_imm(m, 'andi', x5,  x31, 0xffeffffd, -0x100001, -0x3)

# slli
err += test_imm(m, 'slli', x27, x17, 0xe0000000, -0x40000001, 0x1d)
err += test_imm(m, 'slli', x26, x26, 0x33330000, 0x66666666, 0xf)
err += test_imm(m, 'slli', x11, x22, 0xfffeffff, -0x10001, 0x0)
err += test_imm(m, 'slli', x6,  x15, 0x4, 0x4, 0x0)
err += test_imm(m, 'slli', x16, x9,  0x80000000, -0x400001, 0x1f)
err += test_imm(m, 'slli', x20, x11, 0x0, 0x4, 0x1f)
err += test_imm(m, 'slli', x19, x1,  0x800, 0x8, 0x8)
err += test_imm(m, 'slli', x25, x19, 0x0, -0x80000000, 0x10)
err += test_imm(m, 'slli', x12, x8,  0x0, 0x0, 0xc)
err += test_imm(m, 'slli', x30, x27, 0xffffff00, 0x7fffffff, 0x8)
err += test_imm(m, 'slli', x4,  x2,  0x2, 0x1, 0x1)
err += test_imm(m, 'slli', x14, x31, 0x80, 0x2, 0x6)
err += test_imm(m, 'slli', x17, x24, 0x40000, 0x10, 0xe)
err += test_imm(m, 'slli', x10, x4,  0x100, 0x20, 0x3)
err += test_imm(m, 'slli', x2,  x18, 0x8000000, 0x40, 0x15)
err += test_imm(m, 'slli', x23, x5,  0x10000000, 0x80, 0x15)
err += test_imm(m, 'slli', x8,  x13, 0x200, 0x100, 0x1)
err += test_imm(m, 'slli', x0,  x20, 0, 0x200, 0x0)
err += test_imm(m, 'slli', x9,  x16, 0x1000, 0x400, 0x2)
err += test_imm(m, 'slli', x5,  x21, 0x40000000, 0x800, 0x13)

# srli
err += test_imm(m, 'srli', x8,  x30, 0x3fffd2bf, -0xb504, 0x2)
err += test_imm(m, 'srli', x17, x17, 0x0, 0x7, 0x13)
err += test_imm(m, 'srli', x19, x27, 0xffff4afc, -0xb504, 0x0)
err += test_imm(m, 'srli', x9,  x29, 0x3fffffff, 0x3fffffff, 0x0)
err += test_imm(m, 'srli', x22, x25, 0x1, -0xa, 0x1f)
err += test_imm(m, 'srli', x13, x1,  0x0, 0x200, 0x1f)
err += test_imm(m, 'srli', x0,  x21, 0, 0x3, 0x3)
err += test_imm(m, 'srli', x29, x0,  0x0, 0x0, 0x9)
err += test_imm(m, 'srli', x18, x16, 0x0, 0x0, 0x1)
err += test_imm(m, 'srli', x27, x20, 0x3fff, 0x7fffffff, 0x11)
err += test_imm(m, 'srli', x2,  x31, 0x0, 0x1, 0x12)
err += test_imm(m, 'srli', x31, x7,  0x0, 0x2, 0x1d)
err += test_imm(m, 'srli', x16, x14, 0x0, 0x4, 0xf)
err += test_imm(m, 'srli', x25, x12, 0x0, 0x8, 0x1b)
err += test_imm(m, 'srli', x11, x4,  0x0, 0x10, 0xf)
err += test_imm(m, 'srli', x23, x24, 0x0, 0x20, 0x17)
err += test_imm(m, 'srli', x28, x8,  0x0, 0x40, 0xd)
err += test_imm(m, 'srli', x30, x15, 0x0, 0x80, 0x1e)
err += test_imm(m, 'srli', x20, x18, 0x0, 0x100, 0x1f)
err += test_imm(m, 'srli', x14, x13, 0x0, 0x400, 0x12)

# srai
err += test_imm(m, 'srai', x25, x31, -0x1, -0x9, 0x9)
err += test_imm(m, 'srai', x10, x10, 0x2, 0x5, 0x1)
err += test_imm(m, 'srai', x28, x8,  -0x1000001, -0x1000001, 0x0)
err += test_imm(m, 'srai', x5,  x17, 0x100000, 0x100000, 0x0)
err += test_imm(m, 'srai', x27, x23, -0x1, -0x20001, 0x1f)
err += test_imm(m, 'srai', x20, x13, 0x0, 0x1, 0x1f)
err += test_imm(m, 'srai', x11, x22, 0x0, 0x4, 0x4)
err += test_imm(m, 'srai', x30, x7,  -0x80000000, -0x80000000, 0x0)
err += test_imm(m, 'srai', x14, x18, 0x0, 0x0, 0xe)
err += test_imm(m, 'srai', x19, x3,  0x0, 0x7fffffff, 0x1f)
err += test_imm(m, 'srai', x29, x25, 0x0, 0x2, 0x11)
err += test_imm(m, 'srai', x3,  x30, 0x0, 0x8, 0x11)
err += test_imm(m, 'srai', x22, x2,  0x0, 0x10, 0x12)
err += test_imm(m, 'srai', x2,  x12, 0x0, 0x20, 0xd)
err += test_imm(m, 'srai', x12, x1,  0x0, 0x40, 0x17)
err += test_imm(m, 'srai', x24, x20, 0x0, 0x80, 0x9)
err += test_imm(m, 'srai', x0,  x11, 0, 0x100, 0x10)
err += test_imm(m, 'srai', x8,  x26, 0x1, 0x200, 0x9)
err += test_imm(m, 'srai', x17, x9,  0x0, 0x400, 0x11)
err += test_imm(m, 'srai', x23, x16, 0x0, 0x800, 0x1b)

# M extension, mul
err += test_rr(m, 'mul', x31, x31, x5, 0x90000, 0x9, 0x10000)
err += test_rr(m, 'mul', x8, x21, x21, 0x400, 0x20, 0x20)
err += test_rr(m, 'mul', x23, x11, x23, 0xfd555555, -0x55555555, -0x8000001)
err += test_rr(m, 'mul', x7, x14, x20, 0xfffff9fd, -0x201, 0x3)
err += test_rr(m, 'mul', x15, x15, x15, 0x1, -0x1, -0x1)
err += test_rr(m, 'mul', x16, x25, x29, 0x80000000, -0x40001, -0x80000000)
err += test_rr(m, 'mul', x3, x29, x25, 0x0, 0x66666665, 0x0)
err += test_rr(m, 'mul', x17, x23, x28, 0xffff4afc, 0xb504, 0x7fffffff)
err += test_rr(m, 'mul', x22, x2, x19, 0x2, 0x2, 0x1)
err += test_rr(m, 'mul', x28, x20, x22, 0x80000000, -0x80000000, -0x41)
err += test_rr(m, 'mul', x2, x3, x6, 0x0, 0x0, 0x0)
err += test_rr(m, 'mul', x5, x22, x11, 0x80000081, 0x7fffffff, -0x81)
err += test_rr(m, 'mul', x13, x7, x9, 0xefffffff, 0x1, -0x10000001)
err += test_rr(m, 'mul', x6, x10, x17, 0x100, 0x80, 0x2)
err += test_rr(m, 'mul', x30, x1, x2, 0xfffd2bf4, -0xb503, 0x4)
err += test_rr(m, 'mul', x25, x13, x1, 0xfffffff8, -0x20000001, 0x8)
err += test_rr(m, 'mul', x29, x16, x12, 0xffff7ff0, -0x801, 0x10)
err += test_rr(m, 'mul', x21, x27, x24, 0x80000, 0x4000, 0x20)
err += test_rr(m, 'mul', x14, x19, x10, 0x2d4100, 0xb504, 0x40)
err += test_rr(m, 'mul', x26, x6, x14, 0xfffffd80, -0x5, 0x80)

# mulh
err += test_rr(m, 'mulh', x15, x15, x21, 0x5a8, 0xb504, 0x8000000)
err += test_rr(m, 'mulh', x7, x1, x1, 0xa3d70a3, 0x33333332, 0x33333332)
err += test_rr(m, 'mulh', x6, x7, x6, 0x0, -0x8001, -0x801)
err += test_rr(m, 'mulh', x5, x27, x16, 0xffffffff, -0xb503, 0x40)
err += test_rr(m, 'mulh', x26, x26, x26, 0x0, 0x7, 0x7)
err += test_rr(m, 'mulh', x10, x24, x7, 0x5a81, -0xb503, -0x80000000)
err += test_rr(m, 'mulh', x8, x30, x2, 0x0, -0x2001, 0x0)
err += test_rr(m, 'mulh', x19, x0, x11, 0x0, 0x0, 0x7fffffff)
err += test_rr(m, 'mulh', x24, x21, x17, 0x0, 0x2000, 0x1)
err += test_rr(m, 'mulh', x29, x23, x9, 0x2, -0x80000000, -0x4)
err += test_rr(m, 'mulh', x27, x22, x3, 0x0, 0x0, 0x7)
err += test_rr(m, 'mulh', x28, x2, x22, 0x2, 0x7fffffff, 0x6)
err += test_rr(m, 'mulh', x23, x28, x29, 0x0, 0x1, 0x55555555)
err += test_rr(m, 'mulh', x20, x13, x19, 0x0, 0x66666667, 0x2)
err += test_rr(m, 'mulh', x22, x5, x12, 0x0, 0x0, 0x4)
err += test_rr(m, 'mulh', x25, x10, x18, 0x0, 0x4, 0x8)
err += test_rr(m, 'mulh', x9, x25, x15, 0xffffffff, -0x11, 0x10)
err += test_rr(m, 'mulh', x2, x16, x28, 0x0, 0x5, 0x20)
err += test_rr(m, 'mulh', x1, x9, x10, 0x0, 0x40000, 0x80)
err += test_rr(m, 'mulh', x11, x20, x5, 0xffffffdf, -0x20000001, 0x100)

# mulhsu
err += test_rr(m, 'mulhsu', x22, x22, x10, 0x0, 0x4, 0x3)
err += test_rr(m, 'mulhsu', x2, x23, x23, 0x0, 0x80, 0x80)
err += test_rr(m, 'mulhsu', x19, x2, x19, 0x0, 0xb503, 0x0)
err += test_rr(m, 'mulhsu', x7, x25, x9, 0x3, 0x4, 0xffffffff)
err += test_rr(m, 'mulhsu', x28, x28, x28, 0xe3ffffff, -0x20000001, -0x20000001)
err += test_rr(m, 'mulhsu', x8, x0, x18, 0x0, 0x0, 0xffffffdf)
err += test_rr(m, 'mulhsu', x12, x14, x15, 0x0, 0x0, 0x12)
err += test_rr(m, 'mulhsu', x4, x13, x3, 0x77fffffe, 0x7fffffff, 0xefffffff)
err += test_rr(m, 'mulhsu', x30, x6, x16, 0x0, 0x1, 0x400)
err += test_rr(m, 'mulhsu', x1, x8, x5, 0x0, 0x3, 0x2)
err += test_rr(m, 'mulhsu', x9, x11, x27, 0x0, 0x6, 0x4)
err += test_rr(m, 'mulhsu', x13, x27, x25, 0xfffffffe, -0x40000000, 0x8)
err += test_rr(m, 'mulhsu', x16, x30, x14, 0xffffffff, -0x41, 0x10)
err += test_rr(m, 'mulhsu', x11, x15, x22, 0xffffffff, -0x5, 0x20)
err += test_rr(m, 'mulhsu', x29, x16, x4, 0xffffffff, -0x41, 0x40)
err += test_rr(m, 'mulhsu', x21, x24, x13, 0x8, 0x8000000, 0x100)
err += test_rr(m, 'mulhsu', x26, x19, x6, 0xffffffff, -0x101, 0x200)
err += test_rr(m, 'mulhsu', x18, x9, x31, 0x199, 0x33333334, 0x800)
err += test_rr(m, 'mulhsu', x25, x5, x7, 0x0, 0xb504, 0x1000)
err += test_rr(m, 'mulhsu', x0, x26, x17, 0, 0x400000, 0x2000)

# mulhu
err += test_rr(m, 'mulhu', x30, x30, x23, 0x3, 0x4, 0xfffffff7)
err += test_rr(m, 'mulhu', x15, x20, x20, 0x40000000, 0x80000000, 0x80000000)
err += test_rr(m, 'mulhu', x13, x8, x13, 0x0, 0x0, 0x0)
err += test_rr(m, 'mulhu', x10, x4, x6, 0x7f, 0x80, 0xffffffff)
err += test_rr(m, 'mulhu', x26, x26, x26, 0x28f5c28f, 0x66666666, 0x66666666)
err += test_rr(m, 'mulhu', x20, x21, x4, 0x8, 0xffffffff, 0x9)
err += test_rr(m, 'mulhu', x18, x23, x11, 0x0, 0x1, 0x200)
err += test_rr(m, 'mulhu', x27, x16, x18, 0x0, 0x0, 0x2)
err += test_rr(m, 'mulhu', x22, x7, x5, 0x0, 0x10000000, 0x4)
err += test_rr(m, 'mulhu', x3, x31, x1, 0x0, 0x10000000, 0x8)
err += test_rr(m, 'mulhu', x2, x15, x12, 0x0, 0x200, 0x10)
err += test_rr(m, 'mulhu', x5, x1, x15, 0x0, 0x3, 0x20)
err += test_rr(m, 'mulhu', x8, x28, x9, 0x2a, 0xaaaaaaaa, 0x40)
err += test_rr(m, 'mulhu', x14, x5, x24, 0x33, 0x66666667, 0x80)
err += test_rr(m, 'mulhu', x0, x12, x2, 0, 0x55555555, 0x100)
err += test_rr(m, 'mulhu', x19, x9, x30, 0x0, 0x100, 0x400)
err += test_rr(m, 'mulhu', x17, x6, x22, 0x0, 0x6, 0x800)
err += test_rr(m, 'mulhu', x16, x17, x0, 0x0, 0x2, 0x0)
err += test_rr(m, 'mulhu', x24, x18, x16, 0x1fff, 0xfffffff7, 0x2000)
err += test_rr(m, 'mulhu', x28, x3, x21, 0x0, 0xfffe, 0x4000)

# div
err += test_rr(m, 'div', x0, x0, x26, 0, 0x0, 0x40000)
err += test_rr(m, 'div', x19, x17, x17, 0x1, 0x100, 0x100)
err += test_rr(m, 'div', x11, x26, x11, 0x2001, -0x2001, -0x1)
err += test_rr(m, 'div', x18, x30, x5, 0x0, -0x11, 0x10000)
err += test_rr(m, 'div', x10, x10, x10, 0x1, 0x80000, 0x80000)
err += test_rr(m, 'div', x16, x29, x25, 0x0, -0x101, -0x80000000)
err += test_rr(m, 'div', x17, x4, x0, 0xFFFFFFFF, 0x80, 0x0)
err += test_rr(m, 'div', x12, x31, x20, 0x0, 0x55555555, 0x7fffffff)
err += test_rr(m, 'div', x21, x13, x29, 0xffff4afd, -0xb503, 0x1)
err += test_rr(m, 'div', x31, x20, x30, 0x7ff, -0x80000000, -0x100001)
err += test_rr(m, 'div', x20, x7, x13, 0x0, 0x0, -0x9)
err += test_rr(m, 'div', x4, x5, x14, 0x1, 0x7fffffff, 0x7fffffff)
err += test_rr(m, 'div', x1, x6, x23, 0x0, 0x1, 0x5)
err += test_rr(m, 'div', x24, x22, x6, 0xfffffffe, -0x5, 0x2)
err += test_rr(m, 'div', x7, x27, x24, 0xfffffffe, -0x8, 0x4)
err += test_rr(m, 'div', x29, x25, x9, 0x0, 0x0, 0x8)
err += test_rr(m, 'div', x9, x28, x3, 0x400, 0x4000, 0x10)
err += test_rr(m, 'div', x27, x21, x31, 0x1999999, 0x33333332, 0x20)
err += test_rr(m, 'div', x14, x9, x1, 0x1999999, 0x66666666, 0x40)
err += test_rr(m, 'div', x13, x8, x28, 0x0, 0x10, 0x80)

# divu
err += test_rr(m, 'divu', x31, x31, x30, 0x0, 0x66666666, 0xffbfffff)
err += test_rr(m, 'divu', x4, x16, x16, 0x1, 0xfffe, 0xfffe)
err += test_rr(m, 'divu', x29, x27, x29, 0xFFFFFFFF, 0xaaaaaaaa, 0x0)
err += test_rr(m, 'divu', x8, x12, x5, 0x0, 0x200000, 0xffffffff)
err += test_rr(m, 'divu', x6, x6, x6, 0x1, 0x40000, 0x40000)
err += test_rr(m, 'divu', x21, x30, x25, 0x0, 0x0, 0xfff7ffff)
err += test_rr(m, 'divu', x19, x2, x18, 0x1ffff, 0xffffffff, 0x8000)
err += test_rr(m, 'divu', x30, x20, x0, 0xFFFFFFFF, 0x1, 0x0)
err += test_rr(m, 'divu', x5, x15, x31, 0x7fff7fff, 0xfffeffff, 0x2)
err += test_rr(m, 'divu', x22, x10, x28, 0x10000, 0x40000, 0x4)
err += test_rr(m, 'divu', x15, x7, x20, 0x0, 0x6, 0x8)
err += test_rr(m, 'divu', x28, x25, x27, 0x80, 0x800, 0x10)
err += test_rr(m, 'divu', x12, x8, x26, 0x4000, 0x80000, 0x20)
err += test_rr(m, 'divu', x18, x13, x17, 0x8000, 0x200000, 0x40)
err += test_rr(m, 'divu', x27, x9, x24, 0x1fffeff, 0xffff7fff, 0x80)
err += test_rr(m, 'divu', x25, x3, x1, 0x0, 0x0, 0x100)
err += test_rr(m, 'divu', x13, x18, x12, 0x7ffffb, 0xfffff7ff, 0x200)
err += test_rr(m, 'divu', x7, x19, x3, 0x100000, 0x40000000, 0x400)
err += test_rr(m, 'divu', x3, x5, x22, 0x1ffbff, 0xffdfffff, 0x800)
err += test_rr(m, 'divu', x14, x1, x7, 0x0, 0x2, 0x1000)

# rem
err += test_rr(m, 'rem', x11, x24, x11, 0x0, 0x55555556, 0x2)
err += test_rr(m, 'rem', x2, x2, x2, 0x0, 0x40, 0x40)
err += test_rr(m, 'rem', x13, x3, x3, 0x0, -0x41, -0x41)
err += test_rr(m, 'rem', x25, x5, x1, 0x0, -0x80000000, 0x10000)
err += test_rr(m, 'rem', x22, x22, x17, 0x0, 0xb504, 0xb504)
err += test_rr(m, 'rem', x4, x27, x7, 0x55555554, 0x55555554, -0x80000000)
err += test_rr(m, 'rem', x5, x8, x25, 0x200000, 0x200000, 0x0)
err += test_rr(m, 'rem', x14, x4, x0, -0xb503, -0xb503, 0x0)
err += test_rr(m, 'rem', x24, x9, x13, 0x0, 0x6, 0x1)
err += test_rr(m, 'rem', x30, x28, x18, 0x0, 0x0, 0x4)
err += test_rr(m, 'rem', x19, x13, x16, 0x7, 0x7fffffff, -0xa)
err += test_rr(m, 'rem', x20, x7, x9, 0x1, 0x1, 0xb503)
err += test_rr(m, 'rem', x7, x0, x6, 0x0, 0x0, 0x8)
err += test_rr(m, 'rem', x16, x18, x24, 0x0, 0x200, 0x10)
err += test_rr(m, 'rem', x23, x15, x21, -0x1, -0x101, 0x20)
err += test_rr(m, 'rem', x17, x1, x14, -0x1, -0x10000001, 0x40)
err += test_rr(m, 'rem', x1, x19, x15, -0x3, -0x3, 0x80)
err += test_rr(m, 'rem', x15, x11, x12, -0x1, -0x1000001, 0x100)
err += test_rr(m, 'rem', x3, x6, x20, -0x1, -0x800001, 0x200)
err += test_rr(m, 'rem', x10, x25, x30, -0x1, -0x4001, 0x400)

# remu
err += test_rr(m, 'remu', x7, x7, x21, 0x3ff80000, 0xfff7ffff, 0xbfffffff)
err += test_rr(m, 'remu', x30, x9, x9, 0x0, 0x800000, 0x800000)
err += test_rr(m, 'remu', x6, x26, x6, 0xfffffbff, 0xfffffbff, 0x0)
err += test_rr(m, 'remu', x25, x17, x2, 0x400, 0x400, 0xffffffff)
err += test_rr(m, 'remu', x16, x16, x16, 0x0, 0xaaaaaaaa, 0xaaaaaaaa)
err += test_rr(m, 'remu', x5, x2, x1, 0x0, 0x0, 0x66666666)
err += test_rr(m, 'remu', x15, x1, x26, 0x4, 0xffffffff, 0xfffffffb)
err += test_rr(m, 'remu', x0, x29, x15, 0, 0x1, 0xffff7fff)
err += test_rr(m, 'remu', x28, x15, x31, 0x0, 0x1000, 0x2)
err += test_rr(m, 'remu', x21, x3, x7, 0x0, 0x20000000, 0x4)
err += test_rr(m, 'remu', x20, x5, x11, 0x5, 0xd, 0x8)
err += test_rr(m, 'remu', x31, x30, x8, 0x6, 0x6, 0x10)
err += test_rr(m, 'remu', x9, x12, x5, 0xf, 0xf, 0x20)
err += test_rr(m, 'remu', x11, x31, x4, 0x3f, 0xff7fffff, 0x40)
err += test_rr(m, 'remu', x14, x27, x10, 0x40, 0x40, 0x80)
err += test_rr(m, 'remu', x13, x25, x14, 0x0, 0x100, 0x100)
err += test_rr(m, 'remu', x22, x18, x0, 0xa, 0xa, 0x0)
err += test_rr(m, 'remu', x3, x22, x12, 0x103, 0xb503, 0x400)
err += test_rr(m, 'remu', x24, x19, x3, 0x7fb, 0xfffffffb, 0x800)
err += test_rr(m, 'remu', x19, x0, x17, 0x0, 0x0, 0x1000)

print('ALU-tests errors: ' + str(err))

#-------------------------------------------------------------------------------
# some basic tests for load/store instructions
#-------------------------------------------------------------------------------
#m.mem[7]  = -7;  m.mem[8]  = -8; m.mem[9]  = -9; m.mem[10] = 10
#m.mem[11] = -11; m.mem[12] = 12; m.mem[13] = 13; m.x[9] = 9
#m.LB(1, -1, 9);  print(m.x[1])
#m.LBU(1, -1, 9); print(m.x[1])
#m.LH(1, 1, 9);   print(m.x[1]); print(10 - 256*11)
#m.x[2] = -1023
#m.SH(2, 1, 9); print(m.mem[10]); print(_i8(m.mem[11]))
# TODO: improve these tests, they should be self-checking; or just replace them
# by the official RISC-V test-suite or unit tests

# TODO: execute already compiled hex-files tests from
# https://github.com/ucb-bar/riscv-mini/blob/main/tests/rv32ui-p-add.hex
