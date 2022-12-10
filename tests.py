from tinyfive import *

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

#-------------------------------------------------------------------------------
# functions for (1) and (2)
#-------------------------------------------------------------------------------
def i32(x): return np.int32(x)   # convert to 32-bit signed

def check(s, inst, res, correctval):
  if (res != i32(correctval)):
    print('FAIL ' + inst + ' res = ' + str(res) + ' ref = ' + str(i32(correctval)))
  return int(res != i32(correctval))

def test_rr(s, inst, rd, rs1, rs2, correctval, val1, val2):
  """similar to TEST_RR_OP from file arch_test.h of the official test suite"""
  clear_cpu(s)
  s.x[rs1] = val1
  s.x[rs2] = val2
  enc(s, inst,  rd, rs1, rs2)
  exe(s, start=0, instructions=1)
  return check(s, inst, s.x[rd], correctval)

def test_imm(s, inst, rd, rs1, correctval, val1, imm):
  """similar to TEST_IMM_OP from arch_test.h of the official test suite"""
  clear_cpu(s)
  s.x[rs1] = val1
  enc(s, inst,  rd, rs1, imm)
  exe(s, start=0, instructions=1)
  return check(s, inst, s.x[rd], correctval)

""" TODO: for debug only, eventually remove it
def debug_test_rr(s, inst, rd, rs1, rs2, correctval, val1, val2):
  clear_cpu(s)
  s.x[rs1] = val1
  s.x[rs2] = val2
  if   inst == 'add' : ADD (s, rd, rs1, rs2)
  elif inst == 'sub' : SUB (s, rd, rs1, rs2)
  elif inst == 'sll' : SLL (s, rd, rs1, rs2)
  elif inst == 'slt' : SLT (s, rd, rs1, rs2)
  elif inst == 'sltu': SLTU(s, rd, rs1, rs2)
  elif inst == 'xor' : XOR (s, rd, rs1, rs2)
  elif inst == 'srl' : SRL (s, rd, rs1, rs2)
  elif inst == 'sra' : SRA (s, rd, rs1, rs2)
  elif inst == 'or'  : OR  (s, rd, rs1, rs2)
  elif inst == 'and' : AND (s, rd, rs1, rs2)
  return check(s, inst, s.x[rd], correctval)

def debug_test_imm(s, inst, rd, rs1, correctval, val1, imm):
  clear_cpu(s)
  s.x[rs1] = val1
  if   inst == 'addi' : ADDI (s, rd, rs1, imm)
  elif inst == 'slti' : SLTI (s, rd, rs1, imm)
  elif inst == 'sltiu': SLTIU(s, rd, rs1, imm)
  elif inst == 'xori' : XORI (s, rd, rs1, imm)
  elif inst == 'ori'  : ORI  (s, rd, rs1, imm)
  elif inst == 'andi' : ANDI (s, rd, rs1, imm)
  elif inst == 'slli' : SLLI (s, rd, rs1, imm)
  elif inst == 'srli' : SRLI (s, rd, rs1, imm)
  elif inst == 'srai' : SRAI (s, rd, rs1, imm)
  return check(s, inst, s.x[rd], correctval)
"""

#-------------------------------------------------------------------------------
# functions for floating point

def check_fp(s, inst, res, correctval):
  error = not ((res == np.float32(correctval)) or
               (np.isnan(res) and np.isnan(correctval)))
  if error:
    print('FAIL ' + inst + ' res = ' + str(res) + ' ref = ' +
          str(np.float32(correctval)))
  return int(error)

def test_fp(s, inst, correctval, val1, val2=0.0, val3=0.0):
  """similar to TEST_FP_OP*_S from isa/macros/scalar/test_macros.h"""
  clear_cpu(s)
  s.f[1] = val1
  s.f[2] = val2
  s.f[3] = val3
  rd = 10
  if inst in ['fmv.w.x', 'fcvt.s.w', 'fcvt.s.wu']:
    s.x[1] = val1
  enc(s, inst, rd, 1, 2, 3)
  exe(s, start=0, instructions=1)
  if inst in ['feq.s', 'fle.s', 'flt.s', 'fcvt.w.s', 'fcvt.wu.s']:
    return check   (s, inst, s.x[rd], correctval)
  else:
    return check_fp(s, inst, s.f[rd], correctval)
  # TODO: add check for floating point flags, as done in TEST_FP_OP*_S

""" TODO: for debug only, eventually remove it
def debug_test_fp(s, inst, correctval, val1, val2=0.0, val3=0.0):
  clear_cpu(s)
  s.f[1] = val1
  s.f[2] = val2
  s.f[3] = val3
  rd = 10
  if inst in ['fmv.w.x', 'fcvt.s.w', 'fcvt.s.wu']:
    s.x[1] = val1
  if   inst == 'fadd.s'   : FADD_S   (s, rd, 1, 2)
  elif inst == 'fsub.s'   : FSUB_S   (s, rd, 1, 2)
  elif inst == 'fmul.s'   : FMUL_S   (s, rd, 1, 2)
  elif inst == 'fdiv.s'   : FDIV_S   (s, rd, 1, 2)
  elif inst == 'fsqrt.s'  : FSQRT_S  (s, rd, 1)
  elif inst == 'fmin.s'   : FMIN_S   (s, rd, 1, 2)
  elif inst == 'fmax.s'   : FMAX_S   (s, rd, 1, 2)
  elif inst == 'fmadd.s'  : FMADD_S  (s, rd, 1, 2, 3)
  elif inst == 'fmsub.s'  : FMSUB_S  (s, rd, 1, 2, 3)
  elif inst == 'fnmadd.s' : FNMADD_S (s, rd, 1, 2, 3)
  elif inst == 'fnmsub.s' : FNMSUB_S (s, rd, 1, 2, 3)
  elif inst == 'feq.s'    : FEQ_S    (s, rd, 1, 2)
  elif inst == 'fle.s'    : FLE_S    (s, rd, 1, 2)
  elif inst == 'flt.s'    : FLT_S    (s, rd, 1, 2)
  elif inst == 'fcvt.s.w' : FCVT_S_W (s, rd, 1)
  elif inst == 'fcvt.s.wu': FCVT_S_WU(s, rd, 1)
  elif inst == 'fcvt.w.s' : FCVT_W_S (s, rd, 1)
  elif inst == 'fcvt.wu.s': FCVT_WU_S(s, rd, 1)
  if inst in ['feq.s', 'fle.s', 'flt.s', 'fcvt.w.s', 'fcvt.wu.s']:
    return check   (s, inst, s.x[rd], correctval)
  else:
    return check_fp(s, inst, s.f[rd], correctval)
"""

NaN   = np.nan
qNaNf = b2f(0x7fc00000)
sNaNf = b2f(0x7f800001)

#-------------------------------------------------------------------------------
# floating point unit tests
#-------------------------------------------------------------------------------

# fadd, fsub, fmul
err = 0
err += test_fp(s, 'fadd.s',           3.5,        2.5,        1.0)
err += test_fp(s, 'fadd.s',         -1234,    -1235.1,        1.1)
err += test_fp(s, 'fadd.s',    3.14159265, 3.14159265, 0.00000001)
err += test_fp(s, 'fsub.s',           1.5,        2.5,        1.0)
err += test_fp(s, 'fsub.s',         -1234,    -1235.1,       -1.1)
err += test_fp(s, 'fsub.s',    3.14159265, 3.14159265, 0.00000001)
err += test_fp(s, 'fmul.s',           2.5,        2.5,        1.0)
err += test_fp(s, 'fmul.s',       1358.61,    -1235.1,       -1.1)
err += test_fp(s, 'fmul.s', 3.14159265e-8, 3.14159265, 0.00000001)
err += test_fp(s, 'fsub.s',        np.nan,     np.inf,     np.inf)

# fmin, fmax
# TODO: fix the tests that are commented out below
err += test_fp(s, 'fmin.s',        1.0,        2.5,        1.0)
err += test_fp(s, 'fmin.s',    -1235.1,    -1235.1,        1.1)
err += test_fp(s, 'fmin.s',    -1235.1,        1.1,    -1235.1)
#err+= test_fp(s, 'fmin.s',    -1235.1,        NaN,    -1235.1)
err += test_fp(s, 'fmin.s', 0.00000001, 3.14159265, 0.00000001)
err += test_fp(s, 'fmin.s',       -2.0,       -1.0,       -2.0)
err += test_fp(s, 'fmax.s',        2.5,        2.5,        1.0)
err += test_fp(s, 'fmax.s',        1.1,    -1235.1,        1.1)
err += test_fp(s, 'fmax.s',        1.1,        1.1,    -1235.1)
#err+= test_fp(s, 'fmax.s',    -1235.1,        NaN,    -1235.1)
err += test_fp(s, 'fmax.s', 3.14159265, 3.14159265, 0.00000001)
err += test_fp(s, 'fmax.s',       -1.0,       -1.0,       -2.0)
#err+= test_fp(s, 'fmax.s',        1.0,      sNaNf,        1.0)
err += test_fp(s, 'fmax.s',      qNaNf,        NaN,        NaN)
err += test_fp(s, 'fmin.s',       -0.0,       -0.0,        0.0)
err += test_fp(s, 'fmin.s',       -0.0,        0.0,       -0.0)
err += test_fp(s, 'fmax.s',        0.0,       -0.0,        0.0)
err += test_fp(s, 'fmax.s',        0.0,        0.0,       -0.0)

# fdiv, fsqrt
err += test_fp(s, 'fdiv.s',  1.1557273520668288, 3.14159265, 2.71828182)
err += test_fp(s, 'fdiv.s', -0.9991093838555584,      -1234,     1235.1)
err += test_fp(s, 'fdiv.s',          3.14159265, 3.14159265,        1.0)
err += test_fp(s, 'fsqrt.s', 1.7724538498928541, 3.14159265)
err += test_fp(s, 'fsqrt.s',                100,      10000)
err += test_fp(s, 'fsqrt.s',                NaN,       -1.0)
err += test_fp(s, 'fsqrt.s',          13.076696,      171.0)

# fmadd, fnmadd, fmsub, fnmsub
err += test_fp(s, 'fmadd.s',      3.5,  1.0,        2.5,        1.0)
err += test_fp(s, 'fmadd.s',   1236.2, -1.0,    -1235.1,        1.1)
err += test_fp(s, 'fmadd.s',    -12.0,  2.0,       -5.0,       -2.0)
err += test_fp(s, 'fnmadd.s',    -3.5,  1.0,        2.5,        1.0)
err += test_fp(s, 'fnmadd.s', -1236.2, -1.0,    -1235.1,        1.1)
err += test_fp(s, 'fnmadd.s',    12.0,  2.0,       -5.0,       -2.0)
err += test_fp(s, 'fmsub.s',      1.5,  1.0,        2.5,        1.0)
err += test_fp(s, 'fmsub.s',     1234, -1.0,    -1235.1,        1.1)
err += test_fp(s, 'fmsub.s',     -8.0,  2.0,       -5.0,       -2.0)
err += test_fp(s, 'fnmsub.s',    -1.5,  1.0,        2.5,        1.0)
err += test_fp(s, 'fnmsub.s',   -1234, -1.0,    -1235.1,        1.1)
err += test_fp(s, 'fnmsub.s',     8.0,  2.0,       -5.0,       -2.0)

# flw and fsw (adopted from isa/rv32uf/ldst.S)
def test_flw_fsw(s, tdat, tdat_offset):
  clear_cpu(s)
  s.x[a1] = tdat
  enc(s, 'flw.s', f1, tdat_offset, a1)
  enc(s, 'fsw.s', f1, 20, a1)
  enc(s, 'lw',    a0, 20, a1)
  exe(s, start=0, instructions=3)
  ref = read_i32(s, tdat+tdat_offset)
  return check(s, 'flw/fsw', s.x[a0], ref)

tdat = 500  # write testdata to address 500
write_i32(s, 0xbf800000, tdat)
write_i32(s, 0x40000000, tdat+4)
err += test_flw_fsw(s, tdat, 4)
err += test_flw_fsw(s, tdat, 0)

# feq, fle, flt
err += test_fp(s, 'feq.s', 1, -1.36, -1.36)
err += test_fp(s, 'fle.s', 1, -1.36, -1.36)
err += test_fp(s, 'flt.s', 0, -1.36, -1.36)
err += test_fp(s, 'feq.s', 0, -1.37, -1.36)
err += test_fp(s, 'fle.s', 1, -1.37, -1.36)
err += test_fp(s, 'flt.s', 1, -1.37, -1.36)
err += test_fp(s, 'feq.s', 0, NaN, 0)
err += test_fp(s, 'feq.s', 0, NaN, NaN)
err += test_fp(s, 'feq.s', 0, sNaNf, 0)
err += test_fp(s, 'flt.s', 0, NaN, 0)
err += test_fp(s, 'flt.s', 0, NaN, NaN)
err += test_fp(s, 'flt.s', 0, sNaNf, 0)
err += test_fp(s, 'fle.s', 0, NaN, 0)
err += test_fp(s, 'fle.s', 0, NaN, NaN)
err += test_fp(s, 'fle.s', 0, sNaNf, 0)

# fcvt
err += test_fp(s, 'fcvt.s.w',           2.0,  2)
err += test_fp(s, 'fcvt.s.w',          -2.0, -2)
err += test_fp(s, 'fcvt.s.wu',          2.0,  2)
err += test_fp(s, 'fcvt.s.wu',  4.2949673e9, -2)
err += test_fp(s, 'fcvt.w.s',          -1, -1.1)
err += test_fp(s, 'fcvt.w.s',          -1, -1.0)
err += test_fp(s, 'fcvt.w.s',           0, -0.9)
err += test_fp(s, 'fcvt.w.s',           0,  0.9)
err += test_fp(s, 'fcvt.w.s',           1,  1.0)
err += test_fp(s, 'fcvt.w.s',           1,  1.1)
err += test_fp(s, 'fcvt.w.s',      -1<<31, -3e9)
err += test_fp(s, 'fcvt.w.s',   (1<<31)-1,  3e9)
err += test_fp(s, 'fcvt.wu.s',          0, -3.0)
err += test_fp(s, 'fcvt.wu.s',          0, -1.0)
err += test_fp(s, 'fcvt.wu.s',          0, -0.9)
err += test_fp(s, 'fcvt.wu.s',          0,  0.9)
err += test_fp(s, 'fcvt.wu.s',          1,  1.0)
err += test_fp(s, 'fcvt.wu.s',          1,  1.1)
err += test_fp(s, 'fcvt.wu.s',          0, -3e9)
err += test_fp(s, 'fcvt.wu.s', 3000000000,  3e9)

# fmv, fsgnj
def test_fsgn(inst, new_sign, rs1_sign, rs2_sign):
  ref = 0x12345678 | (-new_sign << 31)
  clear_cpu(s)
  s.x[a1] = (rs1_sign << 31) | 0x12345678
  s.x[a2] = -rs2_sign
  enc(s, 'fmv.w.x', f1, a1)
  enc(s, 'fmv.w.x', f2, a2)
  enc(s, inst,      f0, f1, f2)
  enc(s, 'fmv.x.w', a0, f0)
  exe(s, start=0, instructions=4)
  return check(s, 'fmv/fsgnj', s.x[a0], ref)

err += test_fsgn('fsgnj.s', 0, 0, 0)
err += test_fsgn('fsgnj.s', 1, 0, 1)
err += test_fsgn('fsgnj.s', 0, 1, 0)
err += test_fsgn('fsgnj.s', 1, 1, 1)

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
err += test_rr(s, 'add', x24, x4,  x24, 0x80000000, 0x7fffffff, 0x1)
err += test_rr(s, 'add', x28, x10, x10, 0x40000, 0x20000, 0x20000)
err += test_rr(s, 'add', x21, x21, x21, 0xfdfffffe, -0x1000001, -0x1000001)
err += test_rr(s, 'add', x22, x22, x31, 0x3fffe, -0x2, 0x40000)
err += test_rr(s, 'add', x11, x12, x6,  0xaaaaaaac, 0x55555556, 0x55555556)
err += test_rr(s, 'add', x10, x29, x13, 0x80000002, 0x2, -0x80000000)
err += test_rr(s, 'add', x26, x31, x5,  0xffffffef, -0x11, 0x0)
err += test_rr(s, 'add', x7,  x2,  x1,  0xe6666665, 0x66666666, 0x7fffffff)
err += test_rr(s, 'add', x14, x8,  x25, 0x2aaaaaaa, -0x80000000, -0x55555556)
err += test_rr(s, 'add', x1,  x13, x8,  0xfdffffff, 0x0, -0x2000001)
err += test_rr(s, 'add', x0,  x28, x9,  0, 0x1, 0x800000)
err += test_rr(s, 'add', x20, x14, x4,  0x9, 0x7, 0x2)
err += test_rr(s, 'add', x16, x7,  x19, 0xc, 0x8, 0x4)
err += test_rr(s, 'add', x8,  x23, x29, 0x808, 0x800, 0x8)
err += test_rr(s, 'add', x13, x5,  x27, 0x10, 0x0, 0x10)
err += test_rr(s, 'add', x27, x25, x20, 0x55555576, 0x55555556, 0x20)
err += test_rr(s, 'add', x17, x15, x26, 0x2f, -0x11, 0x40)
err += test_rr(s, 'add', x29, x17, x2,  0x7b, -0x5, 0x80)
err += test_rr(s, 'add', x4,  x24, x17, 0x120, 0x20, 0x100)
err += test_rr(s, 'add', x2,  x16, x11, 0x40000200, 0x40000000, 0x200)

# sub
err += test_rr(s, 'sub', x26, x24, x26, 0x5555554e, 0x55555554, 0x6)
err += test_rr(s, 'sub', x23, x17, x17, 0x0, 0x2000000, 0x2000000)
err += test_rr(s, 'sub', x16, x16, x16, 0x0, -0x7, -0x7)
err += test_rr(s, 'sub', x31, x31, x19, 0x99999998, -0x3, 0x66666665)
err += test_rr(s, 'sub', x8,  x23, x14, 0x0, 0x80000, 0x80000)
err += test_rr(s, 'sub', x18, x13, x24, 0x7bffffff, -0x4000001, -0x80000000)
err += test_rr(s, 'sub', x0,  x12, x4,  0, 0x20, 0x0)
err += test_rr(s, 'sub', x10, x22, x9,  0x60000000, -0x20000001, 0x7fffffff)
err += test_rr(s, 'sub', x25, x10, x27, 0xffff, 0x10000, 0x1)
err += test_rr(s, 'sub', x14, x8,  x3,  0xc0000000, -0x80000000, -0x40000000)
err += test_rr(s, 'sub', x29, x25, x30, 0xffe00000, 0x0, 0x200000)
err += test_rr(s, 'sub', x15, x18, x8,  0x7fffdfff, 0x7fffffff, 0x2000)
err += test_rr(s, 'sub', x3,  x14, x15, 0xfffffff1, 0x1, 0x10)
err += test_rr(s, 'sub', x13, x26, x29, 0xfffffff7, -0x7, 0x2)
err += test_rr(s, 'sub', x19, x21, x31, 0xfffffffe, 0x2, 0x4)
err += test_rr(s, 'sub', x11, x30, x5,  0xfefffff7, -0x1000001, 0x8)
err += test_rr(s, 'sub', x30, x28, x7,  0xaaaaaa8a, -0x55555556, 0x20)
err += test_rr(s, 'sub', x7,  x9,  x10, 0xffffff7f, -0x41, 0x40)
err += test_rr(s, 'sub', x17, x0,  x18, 0xffffff80, 0x0, 0x80)
err += test_rr(s, 'sub', x27, x2,  x12, 0xbfffff00, -0x40000000, 0x100)

# sll
err += test_rr(s, 'sll', x28, x16, x28, 0xfffdfc00, -0x81, 0xa)
err += test_rr(s, 'sll', x0,  x21, x21, 0, 0x5, 0x5)
err += test_rr(s, 'sll', x18, x18, x18, 0x80000000, -0x8001, -0x8001)
err += test_rr(s, 'sll', x5,  x5,  x13, 0x7, 0x7, 0x0)
err += test_rr(s, 'sll', x23, x22, x12, 0x180, 0x6, 0x6)
err += test_rr(s, 'sll', x6,  x19, x0,  0x80000000, -0x80000000, 0x0)
err += test_rr(s, 'sll', x13, x25, x24, 0x0, 0x0, 0x4)
err += test_rr(s, 'sll', x16, x12, x26, 0xffe00000, 0x7fffffff, 0x15)
err += test_rr(s, 'sll', x20, x6,  x14, 0x10, 0x1, 0x4)
err += test_rr(s, 'sll', x22, x14, x1,  0x0, 0x2, 0x1f)
err += test_rr(s, 'sll', x21, x29, x7,  0x10, 0x4, 0x2)
err += test_rr(s, 'sll', x4,  x31, x10, 0x0, 0x8, 0x1f)
err += test_rr(s, 'sll', x7,  x17, x20, 0x0, 0x10, 0x1f)
err += test_rr(s, 'sll', x12, x20, x11, 0x10000000, 0x20, 0x17)
err += test_rr(s, 'sll', x3,  x11, x22, 0x800000, 0x40, 0x11)
err += test_rr(s, 'sll', x24, x0,  x30, 0x0, 0x0, 0x10)
err += test_rr(s, 'sll', x8,  x3,  x31, 0x0, 0x100, 0x1b)
err += test_rr(s, 'sll', x10, x27, x17, 0x200, 0x200, 0x0)
err += test_rr(s, 'sll', x11, x10, x19, 0x0, 0x400, 0x1b)
err += test_rr(s, 'sll', x1,  x8,  x27, 0x20000, 0x800, 0x6)

# slt
err += test_rr(s, 'slt', x26, x18, x26, 0x0, 0x66666667, 0x66666667)
err += test_rr(s, 'slt', x1,  x20, x20, 0x0, 0x33333334, 0x33333334)
err += test_rr(s, 'slt', x21, x21, x21, 0x0, -0x4001, -0x4001)
err += test_rr(s, 'slt', x15, x15, x27, 0x1, -0x201, 0x5)
err += test_rr(s, 'slt', x7,  x5,  x18, 0x0, 0x33333334, -0x80000000)
err += test_rr(s, 'slt', x17, x25, x19, 0x0, 0x8000000, 0x0)
err += test_rr(s, 'slt', x23, x9,  x31, 0x1, 0x20000, 0x7fffffff)
err += test_rr(s, 'slt', x11, x2,  x15, 0x1, -0x20001, 0x1)
err += test_rr(s, 'slt', x22, x28, x13, 0x1, -0x80000000, 0x400)
err += test_rr(s, 'slt', x30, x10, x7,  0x1, 0x0, 0x8)
err += test_rr(s, 'slt', x5,  x24, x22, 0x0, 0x7fffffff, -0x101)
err += test_rr(s, 'slt', x24, x8,  x3,  0x0, 0x1, -0x201)
err += test_rr(s, 'slt', x2,  x26, x9,  0x0, 0x10000000, 0x2)
err += test_rr(s, 'slt', x8,  x4,  x14, 0x0, 0x8000, 0x4)
err += test_rr(s, 'slt', x25, x6,  x4,  0x1, -0x1001, 0x10)
err += test_rr(s, 'slt', x14, x31, x25, 0x0, 0x1000000, 0x20)
err += test_rr(s, 'slt', x20, x14, x8,  0x1, 0x5, 0x40)
err += test_rr(s, 'slt', x19, x7,  x11, 0x1, 0x0, 0x80)
err += test_rr(s, 'slt', x9,  x19, x29, 0x1, 0x5, 0x100)
err += test_rr(s, 'slt', x6,  x3,  x23, 0x0, 0x400000, 0x200)

# sltu
err += test_rr(s, 'sltu', x31, x0,  x31, 0x1, 0x0, 0xfffffffe)
err += test_rr(s, 'sltu', x5,  x19, x19, 0x0, 0x100000, 0x100000)
err += test_rr(s, 'sltu', x25, x25, x25, 0x0, 0x40000000, 0x40000000)
err += test_rr(s, 'sltu', x14, x14, x24, 0x1, 0xfffffffe, 0xffffffff)
err += test_rr(s, 'sltu', x12, x17, x13, 0x0, 0x1, 0x1)
err += test_rr(s, 'sltu', x24, x26, x18, 0x1, 0x0, 0xb)
err += test_rr(s, 'sltu', x19, x5,  x14, 0x0, 0xffffffff, 0x0)
err += test_rr(s, 'sltu', x0,  x3,  x22, 0, 0x4, 0x2)
err += test_rr(s, 'sltu', x20, x23, x29, 0x0, 0xf7ffffff, 0x4)
err += test_rr(s, 'sltu', x10, x4,  x6,  0x0, 0x11, 0x8)
err += test_rr(s, 'sltu', x1,  x12, x17, 0x0, 0x7fffffff, 0x10)
err += test_rr(s, 'sltu', x6,  x30, x8,  0x0, 0x2000000, 0x20)
err += test_rr(s, 'sltu', x3,  x21, x16, 0x0, 0xfffff7ff, 0x40)
err += test_rr(s, 'sltu', x17, x29, x26, 0x0, 0x400, 0x80)
err += test_rr(s, 'sltu', x28, x18, x10, 0x1, 0xd, 0x100)
err += test_rr(s, 'sltu', x11, x2,  x28, 0x1, 0x4, 0x200)
err += test_rr(s, 'sltu', x29, x8,  x30, 0x0, 0xffffffbf, 0x400)
err += test_rr(s, 'sltu', x7,  x22, x11, 0x1, 0x80, 0x800)
err += test_rr(s, 'sltu', x9,  x15, x4,  0x0, 0x200000, 0x1000)
err += test_rr(s, 'sltu', x13, x28, x12, 0x1, 0x80, 0x2000)

# xor
err += test_rr(s, 'xor', x24, x27, x24, 0x66666666, 0x66666665, 0x3)
err += test_rr(s, 'xor', x10, x13, x13, 0x0, 0x5, 0x5)
err += test_rr(s, 'xor', x23, x23, x23, 0x0, -0x4001, -0x4001)
err += test_rr(s, 'xor', x28, x28, x14, 0xffffffb7, -0x41, 0x8)
err += test_rr(s, 'xor', x18, x1,  x2,  0x0, -0x1, -0x1)
err += test_rr(s, 'xor', x19, x5,  x22, 0x80400000, 0x400000, -0x80000000)
err += test_rr(s, 'xor', x13, x26, x12, 0xffffffef, -0x11, 0x0)
err += test_rr(s, 'xor', x4,  x12, x11, 0xd5555555, -0x55555556, 0x7fffffff)
err += test_rr(s, 'xor', x17, x19, x30, 0x0, 0x1, 0x1)
err += test_rr(s, 'xor', x3,  x11, x1,  0x6fffffff, -0x80000000, -0x10000001)
err += test_rr(s, 'xor', x8,  x24, x29, 0xffff4afc, 0x0, -0xb504)
err += test_rr(s, 'xor', x9,  x0,  x18, 0x1000, 0x0, 0x1000)
err += test_rr(s, 'xor', x26, x10, x6,  0x80002, 0x80000, 0x2)
err += test_rr(s, 'xor', x30, x22, x31, 0xffffffdb, -0x21, 0x4)
err += test_rr(s, 'xor', x16, x8,  x0,  0x6, 0x6, 0x0)
err += test_rr(s, 'xor', x15, x16, x27, 0x25, 0x5, 0x20)
err += test_rr(s, 'xor', x31, x3,  x26, 0xffdfffbf, -0x200001, 0x40)
err += test_rr(s, 'xor', x14, x9,  x25, 0xffff7f7f, -0x8001, 0x80)
err += test_rr(s, 'xor', x6,  x30, x10, 0x55555456, 0x55555556, 0x100)
err += test_rr(s, 'xor', x7,  x2,  x9,  0xffffbdff, -0x4001, 0x200)

# srl
err += test_rr(s, 'srl', x11, x26, x11, 0x1ff7f, -0x400001, 0xf)
err += test_rr(s, 'srl', x12, x31, x31, 0x155, 0x55555556, 0x55555556)
err += test_rr(s, 'srl', x7,  x7,  x7,  0x1, -0x1, -0x1)
err += test_rr(s, 'srl', x18, x18, x12, 0x100, 0x100, 0x0)
err += test_rr(s, 'srl', x8,  x14, x3,  0x0, 0x9, 0x9)
err += test_rr(s, 'srl', x20, x21, x22, 0x80000, -0x80000000, 0xc)
err += test_rr(s, 'srl', x30, x4,  x17, 0x0, 0x0, 0xf)
err += test_rr(s, 'srl', x6,  x1,  x4,  0x1, 0x7fffffff, 0x1e)
err += test_rr(s, 'srl', x15, x0,  x21, 0x0, 0x0, 0x1d)
err += test_rr(s, 'srl', x5,  x28, x23, 0x0, 0x2, 0x6)
err += test_rr(s, 'srl', x4,  x9,  x30, 0x0, 0x4, 0xa)
err += test_rr(s, 'srl', x10, x13, x29, 0x2, 0x8, 0x2)
err += test_rr(s, 'srl', x25, x2,  x16, 0x0, 0x10, 0x7)
err += test_rr(s, 'srl', x19, x11, x28, 0x0, 0x20, 0x8)
err += test_rr(s, 'srl', x23, x10, x6,  0x0, 0x40, 0xb)
err += test_rr(s, 'srl', x0,  x3,  x1,  0, 0x80, 0x6)
err += test_rr(s, 'srl', x14, x6,  x8,  0x0, 0x200, 0xe)
err += test_rr(s, 'srl', x9,  x20, x27, 0x0, 0x400, 0x1d)
err += test_rr(s, 'srl', x16, x30, x18, 0x0, 0x800, 0xd)
err += test_rr(s, 'srl', x24, x22, x10, 0x10, 0x1000, 0x8)

# sra
err += test_rr(s, 'sra', x27, x16, x27, -0x800000, -0x80000000, 0x8)
err += test_rr(s, 'sra', x16, x12, x12, 0x2000000, 0x2000000, 0x2000000)
err += test_rr(s, 'sra', x1,  x1,  x1,  -0x1, -0x801, -0x801)
err += test_rr(s, 'sra', x13, x13, x19, 0x33333333, 0x33333333, 0x0)
err += test_rr(s, 'sra', x8,  x28, x2,  0x0, 0x6, 0x6)
err += test_rr(s, 'sra', x19, x26, x31, 0x0, 0x0, 0x3)
err += test_rr(s, 'sra', x29, x14, x28, 0x1fff, 0x7fffffff, 0x12)
err += test_rr(s, 'sra', x12, x10, x26, 0x0, 0x1, 0x2)
err += test_rr(s, 'sra', x15, x30, x16, 0x0, 0x2, 0x4)
err += test_rr(s, 'sra', x6,  x24, x0,  0x4, 0x4, 0x0)
err += test_rr(s, 'sra', x3,  x21, x15, 0x0, 0x8, 0xa)
err += test_rr(s, 'sra', x10, x15, x30, 0x10, 0x10, 0x0)
err += test_rr(s, 'sra', x22, x18, x4,  0x4, 0x20, 0x3)
err += test_rr(s, 'sra', x11, x19, x17, 0x0, 0x40, 0x17)
err += test_rr(s, 'sra', x0,  x5,  x14, 0, 0x80, 0x8)
err += test_rr(s, 'sra', x5,  x6,  x24, 0x10, 0x100, 0x4)
err += test_rr(s, 'sra', x31, x7,  x18, 0x0, 0x200, 0x1f)
err += test_rr(s, 'sra', x30, x31, x21, 0x0, 0x400, 0x10)
err += test_rr(s, 'sra', x2,  x4,  x11, 0x0, 0x800, 0x13)
err += test_rr(s, 'sra', x9,  x22, x25, 0x400, 0x1000, 0x2)

# or
err += test_rr(s, 'or', x26, x8,  x26, 0x100010, 0x100000, 0x10)
err += test_rr(s, 'or', x17, x6,  x6,  0x2, 0x2, 0x2)
err += test_rr(s, 'or', x31, x31, x31, 0xefffffff, -0x10000001, -0x10000001)
err += test_rr(s, 'or', x27, x27, x29, 0xfffff7ff, -0x801, 0x400000)
err += test_rr(s, 'or', x18, x30, x19, 0xffefffff, -0x100001, -0x100001)
err += test_rr(s, 'or', x9,  x21, x14, 0x80020000, 0x20000, -0x80000000)
err += test_rr(s, 'or', x4,  x26, x24, 0xffffdfff, -0x2001, 0x0)
err += test_rr(s, 'or', x30, x9,  x8,  0x7fffffff, 0x0, 0x7fffffff)
err += test_rr(s, 'or', x8,  x23, x7,  0xff7fffff, -0x800001, 0x1)
err += test_rr(s, 'or', x22, x12, x0,  0x80000000, -0x80000000, 0x0)
err += test_rr(s, 'or', x28, x10, x30, 0x7fffffff, 0x7fffffff, 0x40)
err += test_rr(s, 'or', x16, x18, x21, 0x55555555, 0x1, 0x55555554)
err += test_rr(s, 'or', x12, x14, x17, 0x1002, 0x1000, 0x2)
err += test_rr(s, 'or', x15, x19, x16, 0xff7fffff, -0x800001, 0x4)
err += test_rr(s, 'or', x7,  x4,  x2,  0xfffffbff, -0x401, 0x8)
err += test_rr(s, 'or', x11, x2,  x22, 0x7fffffff, 0x7fffffff, 0x20)
err += test_rr(s, 'or', x25, x28, x15, 0xfffffdff, -0x201, 0x80)
err += test_rr(s, 'or', x6,  x25, x1,  0xb504, 0xb504, 0x100)
err += test_rr(s, 'or', x20, x17, x10, 0x204, 0x4, 0x200)
err += test_rr(s, 'or', x5,  x20, x23, 0xffefffff, -0x100001, 0x400)

# and
err += test_rr(s, 'and', x25, x24, x25, 0x0, 0x4000, 0x7)
err += test_rr(s, 'and', x18, x3,  x3,  0x800, 0x800, 0x800)
err += test_rr(s, 'and', x19, x19, x19, 0xfffffffd, -0x3, -0x3)
err += test_rr(s, 'and', x5,  x5,  x14, 0x7fffffff, -0x1, 0x7fffffff)
err += test_rr(s, 'and', x20, x23, x16, 0x5, 0x5, 0x5)
err += test_rr(s, 'and', x30, x20, x2,  0x0, 0x2, -0x80000000)
err += test_rr(s, 'and', x13, x7,  x24, 0x0, 0x33333333, 0x0)
err += test_rr(s, 'and', x10, x30, x27, 0x1, -0x40000001, 0x1)
err += test_rr(s, 'and', x22, x28, x18, 0x0, -0x80000000, 0x800)
err += test_rr(s, 'and', x0,  x2,  x15, 0, 0x0, 0x200)
err += test_rr(s, 'and', x12, x25, x26, 0x55555555, 0x7fffffff, 0x55555555)
err += test_rr(s, 'and', x2,  x1,  x31, 0x0, 0x1, 0x55555554)
err += test_rr(s, 'and', x14, x27, x11, 0x0, 0x40000, 0x2)
err += test_rr(s, 'and', x4,  x31, x23, 0x4, -0x20001, 0x4)
err += test_rr(s, 'and', x27, x21, x9,  0x8, -0x55555555, 0x8)
err += test_rr(s, 'and', x23, x26, x7,  0x0, 0x400, 0x10)
err += test_rr(s, 'and', x24, x9,  x20, 0x20, -0x8, 0x20)
err += test_rr(s, 'and', x26, x15, x13, 0x40, -0x101, 0x40)
err += test_rr(s, 'and', x17, x12, x4,  0x80, -0x2000001, 0x80)
err += test_rr(s, 'and', x8,  x4,  x17, 0x0, 0x66666665, 0x100)

# addi
err += test_imm(s, 'addi', x7,  x20, 0x1ffff800, 0x20000000, -0x800)
err += test_imm(s, 'addi', x3,  x3,  0x400, 0x400, 0x0)
err += test_imm(s, 'addi', x22, x4,  0x5fe, -0x201, 0x7ff)
err += test_imm(s, 'addi', x11, x30, 0x1, 0x0, 0x1)
err += test_imm(s, 'addi', x31, x27, 0x80000010, -0x80000000, 0x10)
err += test_imm(s, 'addi', x30, x17, 0x80000005, 0x7fffffff, 0x6)
err += test_imm(s, 'addi', x28, x18, 0x5, 0x1, 0x4)
err += test_imm(s, 'addi', x6,  x13, 0xa, 0x5, 0x5)
err += test_imm(s, 'addi', x16, x10, 0xaaaaaa8a, -0x55555555, -0x21)
err += test_imm(s, 'addi', x21, x9,  0xfffffff1, -0x11, 0x2)
err += test_imm(s, 'addi', x2,  x7,  0xb50d, 0xb505, 0x8)
err += test_imm(s, 'addi', x18, x22, 0xffff4b1c, -0xb504, 0x20)
err += test_imm(s, 'addi', x0,  x29, 0, -0x200001, 0x40)
err += test_imm(s, 'addi', x13, x25, 0x85, 0x5, 0x80)
err += test_imm(s, 'addi', x29, x11, 0xfe0000ff, -0x2000001, 0x100)
err += test_imm(s, 'addi', x8,  x6,  0x210, 0x10, 0x200)
err += test_imm(s, 'addi', x4,  x19, 0x402, 0x2, 0x400)
err += test_imm(s, 'addi', x10, x12, 0x55555552, 0x55555554, -0x2)
err += test_imm(s, 'addi', x26, x31, 0x55555552, 0x55555555, -0x3)
err += test_imm(s, 'addi', x15, x26, 0x5555554f, 0x55555554, -0x5)

# slti
err += test_imm(s, 'slti', x12, x25, 0x0, -0x81, -0x800)
err += test_imm(s, 'slti', x5,  x5,  0x1, -0x1001, 0x0)
err += test_imm(s, 'slti', x28, x4,  0x1, -0x40000000, 0x7ff)
err += test_imm(s, 'slti', x15, x31, 0x1, -0x11, 0x1)
err += test_imm(s, 'slti', x13, x1,  0x1, -0x80000000, 0x3)
err += test_imm(s, 'slti', x1,  x15, 0x1, 0x0, 0x2)
err += test_imm(s, 'slti', x9,  x16, 0x0, 0x7fffffff, -0x8)
err += test_imm(s, 'slti', x31, x11, 0x0, 0x1, -0x400)
err += test_imm(s, 'slti', x27, x14, 0x0, 0x10, 0x10)
err += test_imm(s, 'slti', x26, x12, 0x0, 0x33333334, 0x4)
err += test_imm(s, 'slti', x4,  x17, 0x0, 0x3fffffff, 0x8)
err += test_imm(s, 'slti', x10, x18, 0x1, -0x2001, 0x20)
err += test_imm(s, 'slti', x21, x27, 0x1, 0x3, 0x40)
err += test_imm(s, 'slti', x8,  x3,  0x0, 0x55555554, 0x80)
err += test_imm(s, 'slti', x0,  x7,  0, 0x55555554, 0x100)
err += test_imm(s, 'slti', x24, x22, 0x1, -0x55555556, 0x200)
err += test_imm(s, 'slti', x18, x24, 0x0, 0x4000, 0x400)
err += test_imm(s, 'slti', x25, x6,  0x1, -0x401, -0x2)
err += test_imm(s, 'slti', x23, x21, 0x0, 0x66666667, -0x3)
err += test_imm(s, 'slti', x7,  x0,  0x0, 0x0, -0x5)

# sltiu
err += test_imm(s, 'sltiu', x28, x23, 0x0, 0x400, 0x0)
err += test_imm(s, 'sltiu', x2,  x2,  0x1, 0x800, 0xfff)
err += test_imm(s, 'sltiu', x25, x3,  0x0, 0x4, 0x1)
err += test_imm(s, 'sltiu', x11, x19, 0x1, 0x0, 0x6)
err += test_imm(s, 'sltiu', x15, x14, 0x0, 0xffffffff, 0x2c)
err += test_imm(s, 'sltiu', x4,  x13, 0x0, 0x1, 0x0)
err += test_imm(s, 'sltiu', x3,  x26, 0x0, 0xd, 0xd)
err += test_imm(s, 'sltiu', x29, x20, 0x0, 0xaaaaaaaa, 0x2)
err += test_imm(s, 'sltiu', x16, x27, 0x0, 0x7fffffff, 0x4)
err += test_imm(s, 'sltiu', x20, x17, 0x0, 0xfeffffff, 0x8)
err += test_imm(s, 'sltiu', x8,  x31, 0x0, 0x800, 0x10)
err += test_imm(s, 'sltiu', x23, x24, 0x1, 0xc, 0x20)
err += test_imm(s, 'sltiu', x26, x25, 0x0, 0x55555555, 0x40)
err += test_imm(s, 'sltiu', x6,  x22, 0x0, 0x80000, 0x80)
err += test_imm(s, 'sltiu', x5,  x12, 0x0, 0xfffffff7, 0x100)
err += test_imm(s, 'sltiu', x1,  x9,  0x0, 0x80000000, 0x200)
err += test_imm(s, 'sltiu', x10, x28, 0x0, 0xfffbffff, 0x400)
err += test_imm(s, 'sltiu', x31, x21, 0x1, 0x0, 0x800)
err += test_imm(s, 'sltiu', x21, x0,  0x1, 0x0, 0xffe)
err += test_imm(s, 'sltiu', x14, x11, 0x1, 0x12, 0xffd)

# xori
err += test_imm(s, 'xori', x10, x24, 0xcccccb34, 0x33333334, -0x800)
err += test_imm(s, 'xori', x18, x18, 0x4, 0x4, 0x0)
err += test_imm(s, 'xori', x24, x15, 0xfffff803, -0x4, 0x7ff)
err += test_imm(s, 'xori', x20, x11, 0x3, 0x2, 0x1)
err += test_imm(s, 'xori', x21, x7,  0x80000554, -0x80000000, 0x554)
err += test_imm(s, 'xori', x27, x17, 0xfffffbff, 0x0, -0x401)
err += test_imm(s, 'xori', x1,  x22, 0x80000009, 0x7fffffff, -0xa)
err += test_imm(s, 'xori', x22, x20, 0x5, 0x1, 0x4)
err += test_imm(s, 'xori', x31, x19, 0x0, -0x201, -0x201)
err += test_imm(s, 'xori', x5,  x9,  0xffffffdd, -0x21, 0x2)
err += test_imm(s, 'xori', x29, x28, 0x80000008, -0x80000000, 0x8)
err += test_imm(s, 'xori', x4,  x30, 0xbfffffef, -0x40000001, 0x10)
err += test_imm(s, 'xori', x8,  x27, 0x7fffffdf, 0x7fffffff, 0x20)
err += test_imm(s, 'xori', x25, x3,  0x66666626, 0x66666666, 0x40)
err += test_imm(s, 'xori', x17, x31, 0xfff7ff7f, -0x80001, 0x80)
err += test_imm(s, 'xori', x16, x29, 0xffff4bfc, -0xb504, 0x100)
err += test_imm(s, 'xori', x6,  x4,  0x200, 0x0, 0x200)
err += test_imm(s, 'xori', x3,  x14, 0xffeffbff, -0x100001, 0x400)
err += test_imm(s, 'xori', x15, x12, 0x7, -0x7, -0x2)
err += test_imm(s, 'xori', x9,  x21, 0xfffffff8, 0x5, -0x3)

# ori
err += test_imm(s, 'ori', x22, x5,  0xfffffdff, -0x201, -0x800)
err += test_imm(s, 'ori', x27, x27, 0x0, 0x0, 0x0)
err += test_imm(s, 'ori', x8,  x17, 0x333337ff, 0x33333334, 0x7ff)
err += test_imm(s, 'ori', x1,  x20, 0xffff4afd, -0xb504, 0x1)
err += test_imm(s, 'ori', x19, x12, 0x8000002d, -0x80000000, 0x2d)
err += test_imm(s, 'ori', x3,  x8,  0x7fffffff, 0x7fffffff, 0x555)
err += test_imm(s, 'ori', x26, x28, 0x667, 0x1, 0x667)
err += test_imm(s, 'ori', x23, x16, 0xffffffff, 0x7, -0x7)
err += test_imm(s, 'ori', x31, x25, 0x40002, 0x40000, 0x2)
err += test_imm(s, 'ori', x11, x23, 0x20000004, 0x20000000, 0x4)
err += test_imm(s, 'ori', x17, x14, 0xfffffdff, -0x201, 0x8)
err += test_imm(s, 'ori', x7,  x31, 0x12, 0x2, 0x10)
err += test_imm(s, 'ori', x4,  x21, 0x8020, 0x8000, 0x20)
err += test_imm(s, 'ori', x5,  x15, 0x840, 0x800, 0x40)
err += test_imm(s, 'ori', x25, x30, 0xfffbffff, -0x40001, 0x80)
err += test_imm(s, 'ori', x30, x11, 0xfffffffb, -0x5, 0x100)
err += test_imm(s, 'ori', x10, x4,  0xfff7ffff, -0x80001, 0x200)
err += test_imm(s, 'ori', x0,  x13, 0, -0x40000001, 0x400)
err += test_imm(s, 'ori', x6,  x26, 0xffffffff, -0x21, -0x2)
err += test_imm(s, 'ori', x18, x19, 0xffffffff, 0xb503, -0x3)

# andi
err += test_imm(s, 'andi', x10, x22, 0xfffff800, -0x2, -0x800)
err += test_imm(s, 'andi', x25, x25, 0x0, -0x1001, 0x0)
err += test_imm(s, 'andi', x17, x16, 0x7ff, -0x2000001, 0x7ff)
err += test_imm(s, 'andi', x8,  x2,  0x1, -0x20001, 0x1)
err += test_imm(s, 'andi', x30, x28, 0x0, -0x80000000, 0x4)
err += test_imm(s, 'andi', x19, x4,  0x0, 0x0, -0x800)
err += test_imm(s, 'andi', x2,  x10, 0x6, 0x7fffffff, 0x6)
err += test_imm(s, 'andi', x13, x7,  0x0, 0x1, 0x554)
err += test_imm(s, 'andi', x9,  x27, 0x80, 0x80, 0x80)
err += test_imm(s, 'andi', x3,  x17, 0x7fffffd4, 0x7fffffff, -0x2c)
err += test_imm(s, 'andi', x26, x0,  0x0, 0x0, 0x2)
err += test_imm(s, 'andi', x21, x23, 0x0, 0x66666666, 0x8)
err += test_imm(s, 'andi', x14, x6,  0x0, 0x0, 0x10)
err += test_imm(s, 'andi', x22, x5,  0x0, 0x100, 0x20)
err += test_imm(s, 'andi', x29, x8,  0x40, -0x5, 0x40)
err += test_imm(s, 'andi', x23, x12, 0x0, 0x1, 0x100)
err += test_imm(s, 'andi', x6,  x15, 0x200, -0x55555555, 0x200)
err += test_imm(s, 'andi', x11, x29, 0x0, 0x0, 0x400)
err += test_imm(s, 'andi', x1,  x20, 0x66666666, 0x66666667, -0x2)
err += test_imm(s, 'andi', x5,  x31, 0xffeffffd, -0x100001, -0x3)

# slli
err += test_imm(s, 'slli', x27, x17, 0xe0000000, -0x40000001, 0x1d)
err += test_imm(s, 'slli', x26, x26, 0x33330000, 0x66666666, 0xf)
err += test_imm(s, 'slli', x11, x22, 0xfffeffff, -0x10001, 0x0)
err += test_imm(s, 'slli', x6,  x15, 0x4, 0x4, 0x0)
err += test_imm(s, 'slli', x16, x9,  0x80000000, -0x400001, 0x1f)
err += test_imm(s, 'slli', x20, x11, 0x0, 0x4, 0x1f)
err += test_imm(s, 'slli', x19, x1,  0x800, 0x8, 0x8)
err += test_imm(s, 'slli', x25, x19, 0x0, -0x80000000, 0x10)
err += test_imm(s, 'slli', x12, x8,  0x0, 0x0, 0xc)
err += test_imm(s, 'slli', x30, x27, 0xffffff00, 0x7fffffff, 0x8)
err += test_imm(s, 'slli', x4,  x2,  0x2, 0x1, 0x1)
err += test_imm(s, 'slli', x14, x31, 0x80, 0x2, 0x6)
err += test_imm(s, 'slli', x17, x24, 0x40000, 0x10, 0xe)
err += test_imm(s, 'slli', x10, x4,  0x100, 0x20, 0x3)
err += test_imm(s, 'slli', x2,  x18, 0x8000000, 0x40, 0x15)
err += test_imm(s, 'slli', x23, x5,  0x10000000, 0x80, 0x15)
err += test_imm(s, 'slli', x8,  x13, 0x200, 0x100, 0x1)
err += test_imm(s, 'slli', x0,  x20, 0, 0x200, 0x0)
err += test_imm(s, 'slli', x9,  x16, 0x1000, 0x400, 0x2)
err += test_imm(s, 'slli', x5,  x21, 0x40000000, 0x800, 0x13)

# srli
err += test_imm(s, 'srli', x8,  x30, 0x3fffd2bf, -0xb504, 0x2)
err += test_imm(s, 'srli', x17, x17, 0x0, 0x7, 0x13)
err += test_imm(s, 'srli', x19, x27, 0xffff4afc, -0xb504, 0x0)
err += test_imm(s, 'srli', x9,  x29, 0x3fffffff, 0x3fffffff, 0x0)
err += test_imm(s, 'srli', x22, x25, 0x1, -0xa, 0x1f)
err += test_imm(s, 'srli', x13, x1,  0x0, 0x200, 0x1f)
err += test_imm(s, 'srli', x0,  x21, 0, 0x3, 0x3)
err += test_imm(s, 'srli', x29, x0,  0x0, 0x0, 0x9)
err += test_imm(s, 'srli', x18, x16, 0x0, 0x0, 0x1)
err += test_imm(s, 'srli', x27, x20, 0x3fff, 0x7fffffff, 0x11)
err += test_imm(s, 'srli', x2,  x31, 0x0, 0x1, 0x12)
err += test_imm(s, 'srli', x31, x7,  0x0, 0x2, 0x1d)
err += test_imm(s, 'srli', x16, x14, 0x0, 0x4, 0xf)
err += test_imm(s, 'srli', x25, x12, 0x0, 0x8, 0x1b)
err += test_imm(s, 'srli', x11, x4,  0x0, 0x10, 0xf)
err += test_imm(s, 'srli', x23, x24, 0x0, 0x20, 0x17)
err += test_imm(s, 'srli', x28, x8,  0x0, 0x40, 0xd)
err += test_imm(s, 'srli', x30, x15, 0x0, 0x80, 0x1e)
err += test_imm(s, 'srli', x20, x18, 0x0, 0x100, 0x1f)
err += test_imm(s, 'srli', x14, x13, 0x0, 0x400, 0x12)

# srai
err += test_imm(s, 'srai', x25, x31, -0x1, -0x9, 0x9)
err += test_imm(s, 'srai', x10, x10, 0x2, 0x5, 0x1)
err += test_imm(s, 'srai', x28, x8,  -0x1000001, -0x1000001, 0x0)
err += test_imm(s, 'srai', x5,  x17, 0x100000, 0x100000, 0x0)
err += test_imm(s, 'srai', x27, x23, -0x1, -0x20001, 0x1f)
err += test_imm(s, 'srai', x20, x13, 0x0, 0x1, 0x1f)
err += test_imm(s, 'srai', x11, x22, 0x0, 0x4, 0x4)
err += test_imm(s, 'srai', x30, x7,  -0x80000000, -0x80000000, 0x0)
err += test_imm(s, 'srai', x14, x18, 0x0, 0x0, 0xe)
err += test_imm(s, 'srai', x19, x3,  0x0, 0x7fffffff, 0x1f)
err += test_imm(s, 'srai', x29, x25, 0x0, 0x2, 0x11)
err += test_imm(s, 'srai', x3,  x30, 0x0, 0x8, 0x11)
err += test_imm(s, 'srai', x22, x2,  0x0, 0x10, 0x12)
err += test_imm(s, 'srai', x2,  x12, 0x0, 0x20, 0xd)
err += test_imm(s, 'srai', x12, x1,  0x0, 0x40, 0x17)
err += test_imm(s, 'srai', x24, x20, 0x0, 0x80, 0x9)
err += test_imm(s, 'srai', x0,  x11, 0, 0x100, 0x10)
err += test_imm(s, 'srai', x8,  x26, 0x1, 0x200, 0x9)
err += test_imm(s, 'srai', x17, x9,  0x0, 0x400, 0x11)
err += test_imm(s, 'srai', x23, x16, 0x0, 0x800, 0x1b)

# M extension, mul
err += test_rr(s, 'mul', x31, x31, x5, 0x90000, 0x9, 0x10000)
err += test_rr(s, 'mul', x8, x21, x21, 0x400, 0x20, 0x20)
err += test_rr(s, 'mul', x23, x11, x23, 0xfd555555, -0x55555555, -0x8000001)
err += test_rr(s, 'mul', x7, x14, x20, 0xfffff9fd, -0x201, 0x3)
err += test_rr(s, 'mul', x15, x15, x15, 0x1, -0x1, -0x1)
err += test_rr(s, 'mul', x16, x25, x29, 0x80000000, -0x40001, -0x80000000)
err += test_rr(s, 'mul', x3, x29, x25, 0x0, 0x66666665, 0x0)
err += test_rr(s, 'mul', x17, x23, x28, 0xffff4afc, 0xb504, 0x7fffffff)
err += test_rr(s, 'mul', x22, x2, x19, 0x2, 0x2, 0x1)
err += test_rr(s, 'mul', x28, x20, x22, 0x80000000, -0x80000000, -0x41)
err += test_rr(s, 'mul', x2, x3, x6, 0x0, 0x0, 0x0)
err += test_rr(s, 'mul', x5, x22, x11, 0x80000081, 0x7fffffff, -0x81)
err += test_rr(s, 'mul', x13, x7, x9, 0xefffffff, 0x1, -0x10000001)
err += test_rr(s, 'mul', x6, x10, x17, 0x100, 0x80, 0x2)
err += test_rr(s, 'mul', x30, x1, x2, 0xfffd2bf4, -0xb503, 0x4)
err += test_rr(s, 'mul', x25, x13, x1, 0xfffffff8, -0x20000001, 0x8)
err += test_rr(s, 'mul', x29, x16, x12, 0xffff7ff0, -0x801, 0x10)
err += test_rr(s, 'mul', x21, x27, x24, 0x80000, 0x4000, 0x20)
err += test_rr(s, 'mul', x14, x19, x10, 0x2d4100, 0xb504, 0x40)
err += test_rr(s, 'mul', x26, x6, x14, 0xfffffd80, -0x5, 0x80)

# mulh
err += test_rr(s, 'mulh', x15, x15, x21, 0x5a8, 0xb504, 0x8000000)
err += test_rr(s, 'mulh', x7, x1, x1, 0xa3d70a3, 0x33333332, 0x33333332)
err += test_rr(s, 'mulh', x6, x7, x6, 0x0, -0x8001, -0x801)
err += test_rr(s, 'mulh', x5, x27, x16, 0xffffffff, -0xb503, 0x40)
err += test_rr(s, 'mulh', x26, x26, x26, 0x0, 0x7, 0x7)
err += test_rr(s, 'mulh', x10, x24, x7, 0x5a81, -0xb503, -0x80000000)
err += test_rr(s, 'mulh', x8, x30, x2, 0x0, -0x2001, 0x0)
err += test_rr(s, 'mulh', x19, x0, x11, 0x0, 0x0, 0x7fffffff)
err += test_rr(s, 'mulh', x24, x21, x17, 0x0, 0x2000, 0x1)
err += test_rr(s, 'mulh', x29, x23, x9, 0x2, -0x80000000, -0x4)
err += test_rr(s, 'mulh', x27, x22, x3, 0x0, 0x0, 0x7)
err += test_rr(s, 'mulh', x28, x2, x22, 0x2, 0x7fffffff, 0x6)
err += test_rr(s, 'mulh', x23, x28, x29, 0x0, 0x1, 0x55555555)
err += test_rr(s, 'mulh', x20, x13, x19, 0x0, 0x66666667, 0x2)
err += test_rr(s, 'mulh', x22, x5, x12, 0x0, 0x0, 0x4)
err += test_rr(s, 'mulh', x25, x10, x18, 0x0, 0x4, 0x8)
err += test_rr(s, 'mulh', x9, x25, x15, 0xffffffff, -0x11, 0x10)
err += test_rr(s, 'mulh', x2, x16, x28, 0x0, 0x5, 0x20)
err += test_rr(s, 'mulh', x1, x9, x10, 0x0, 0x40000, 0x80)
err += test_rr(s, 'mulh', x11, x20, x5, 0xffffffdf, -0x20000001, 0x100)

# mulhsu
err += test_rr(s, 'mulhsu', x22, x22, x10, 0x0, 0x4, 0x3)
err += test_rr(s, 'mulhsu', x2, x23, x23, 0x0, 0x80, 0x80)
err += test_rr(s, 'mulhsu', x19, x2, x19, 0x0, 0xb503, 0x0)
err += test_rr(s, 'mulhsu', x7, x25, x9, 0x3, 0x4, 0xffffffff)
err += test_rr(s, 'mulhsu', x28, x28, x28, 0xe3ffffff, -0x20000001, -0x20000001)
err += test_rr(s, 'mulhsu', x8, x0, x18, 0x0, 0x0, 0xffffffdf)
err += test_rr(s, 'mulhsu', x12, x14, x15, 0x0, 0x0, 0x12)
err += test_rr(s, 'mulhsu', x4, x13, x3, 0x77fffffe, 0x7fffffff, 0xefffffff)
err += test_rr(s, 'mulhsu', x30, x6, x16, 0x0, 0x1, 0x400)
err += test_rr(s, 'mulhsu', x1, x8, x5, 0x0, 0x3, 0x2)
err += test_rr(s, 'mulhsu', x9, x11, x27, 0x0, 0x6, 0x4)
err += test_rr(s, 'mulhsu', x13, x27, x25, 0xfffffffe, -0x40000000, 0x8)
err += test_rr(s, 'mulhsu', x16, x30, x14, 0xffffffff, -0x41, 0x10)
err += test_rr(s, 'mulhsu', x11, x15, x22, 0xffffffff, -0x5, 0x20)
err += test_rr(s, 'mulhsu', x29, x16, x4, 0xffffffff, -0x41, 0x40)
err += test_rr(s, 'mulhsu', x21, x24, x13, 0x8, 0x8000000, 0x100)
err += test_rr(s, 'mulhsu', x26, x19, x6, 0xffffffff, -0x101, 0x200)
err += test_rr(s, 'mulhsu', x18, x9, x31, 0x199, 0x33333334, 0x800)
err += test_rr(s, 'mulhsu', x25, x5, x7, 0x0, 0xb504, 0x1000)
err += test_rr(s, 'mulhsu', x0, x26, x17, 0, 0x400000, 0x2000)

# mulhu
err += test_rr(s, 'mulhu', x30, x30, x23, 0x3, 0x4, 0xfffffff7)
err += test_rr(s, 'mulhu', x15, x20, x20, 0x40000000, 0x80000000, 0x80000000)
err += test_rr(s, 'mulhu', x13, x8, x13, 0x0, 0x0, 0x0)
err += test_rr(s, 'mulhu', x10, x4, x6, 0x7f, 0x80, 0xffffffff)
err += test_rr(s, 'mulhu', x26, x26, x26, 0x28f5c28f, 0x66666666, 0x66666666)
err += test_rr(s, 'mulhu', x20, x21, x4, 0x8, 0xffffffff, 0x9)
err += test_rr(s, 'mulhu', x18, x23, x11, 0x0, 0x1, 0x200)
err += test_rr(s, 'mulhu', x27, x16, x18, 0x0, 0x0, 0x2)
err += test_rr(s, 'mulhu', x22, x7, x5, 0x0, 0x10000000, 0x4)
err += test_rr(s, 'mulhu', x3, x31, x1, 0x0, 0x10000000, 0x8)
err += test_rr(s, 'mulhu', x2, x15, x12, 0x0, 0x200, 0x10)
err += test_rr(s, 'mulhu', x5, x1, x15, 0x0, 0x3, 0x20)
err += test_rr(s, 'mulhu', x8, x28, x9, 0x2a, 0xaaaaaaaa, 0x40)
err += test_rr(s, 'mulhu', x14, x5, x24, 0x33, 0x66666667, 0x80)
err += test_rr(s, 'mulhu', x0, x12, x2, 0, 0x55555555, 0x100)
err += test_rr(s, 'mulhu', x19, x9, x30, 0x0, 0x100, 0x400)
err += test_rr(s, 'mulhu', x17, x6, x22, 0x0, 0x6, 0x800)
err += test_rr(s, 'mulhu', x16, x17, x0, 0x0, 0x2, 0x0)
err += test_rr(s, 'mulhu', x24, x18, x16, 0x1fff, 0xfffffff7, 0x2000)
err += test_rr(s, 'mulhu', x28, x3, x21, 0x0, 0xfffe, 0x4000)

# div
err += test_rr(s, 'div', x0, x0, x26, 0, 0x0, 0x40000)
err += test_rr(s, 'div', x19, x17, x17, 0x1, 0x100, 0x100)
err += test_rr(s, 'div', x11, x26, x11, 0x2001, -0x2001, -0x1)
err += test_rr(s, 'div', x18, x30, x5, 0x0, -0x11, 0x10000)
err += test_rr(s, 'div', x10, x10, x10, 0x1, 0x80000, 0x80000)
err += test_rr(s, 'div', x16, x29, x25, 0x0, -0x101, -0x80000000)
err += test_rr(s, 'div', x17, x4, x0, 0xFFFFFFFF, 0x80, 0x0)
err += test_rr(s, 'div', x12, x31, x20, 0x0, 0x55555555, 0x7fffffff)
err += test_rr(s, 'div', x21, x13, x29, 0xffff4afd, -0xb503, 0x1)
err += test_rr(s, 'div', x31, x20, x30, 0x7ff, -0x80000000, -0x100001)
err += test_rr(s, 'div', x20, x7, x13, 0x0, 0x0, -0x9)
err += test_rr(s, 'div', x4, x5, x14, 0x1, 0x7fffffff, 0x7fffffff)
err += test_rr(s, 'div', x1, x6, x23, 0x0, 0x1, 0x5)
err += test_rr(s, 'div', x24, x22, x6, 0xfffffffe, -0x5, 0x2)
err += test_rr(s, 'div', x7, x27, x24, 0xfffffffe, -0x8, 0x4)
err += test_rr(s, 'div', x29, x25, x9, 0x0, 0x0, 0x8)
err += test_rr(s, 'div', x9, x28, x3, 0x400, 0x4000, 0x10)
err += test_rr(s, 'div', x27, x21, x31, 0x1999999, 0x33333332, 0x20)
err += test_rr(s, 'div', x14, x9, x1, 0x1999999, 0x66666666, 0x40)
err += test_rr(s, 'div', x13, x8, x28, 0x0, 0x10, 0x80)

# divu
err += test_rr(s, 'divu', x31, x31, x30, 0x0, 0x66666666, 0xffbfffff)
err += test_rr(s, 'divu', x4, x16, x16, 0x1, 0xfffe, 0xfffe)
err += test_rr(s, 'divu', x29, x27, x29, 0xFFFFFFFF, 0xaaaaaaaa, 0x0)
err += test_rr(s, 'divu', x8, x12, x5, 0x0, 0x200000, 0xffffffff)
err += test_rr(s, 'divu', x6, x6, x6, 0x1, 0x40000, 0x40000)
err += test_rr(s, 'divu', x21, x30, x25, 0x0, 0x0, 0xfff7ffff)
err += test_rr(s, 'divu', x19, x2, x18, 0x1ffff, 0xffffffff, 0x8000)
err += test_rr(s, 'divu', x30, x20, x0, 0xFFFFFFFF, 0x1, 0x0)
err += test_rr(s, 'divu', x5, x15, x31, 0x7fff7fff, 0xfffeffff, 0x2)
err += test_rr(s, 'divu', x22, x10, x28, 0x10000, 0x40000, 0x4)
err += test_rr(s, 'divu', x15, x7, x20, 0x0, 0x6, 0x8)
err += test_rr(s, 'divu', x28, x25, x27, 0x80, 0x800, 0x10)
err += test_rr(s, 'divu', x12, x8, x26, 0x4000, 0x80000, 0x20)
err += test_rr(s, 'divu', x18, x13, x17, 0x8000, 0x200000, 0x40)
err += test_rr(s, 'divu', x27, x9, x24, 0x1fffeff, 0xffff7fff, 0x80)
err += test_rr(s, 'divu', x25, x3, x1, 0x0, 0x0, 0x100)
err += test_rr(s, 'divu', x13, x18, x12, 0x7ffffb, 0xfffff7ff, 0x200)
err += test_rr(s, 'divu', x7, x19, x3, 0x100000, 0x40000000, 0x400)
err += test_rr(s, 'divu', x3, x5, x22, 0x1ffbff, 0xffdfffff, 0x800)
err += test_rr(s, 'divu', x14, x1, x7, 0x0, 0x2, 0x1000)

# rem
err += test_rr(s, 'rem', x11, x24, x11, 0x0, 0x55555556, 0x2)
err += test_rr(s, 'rem', x2, x2, x2, 0x0, 0x40, 0x40)
err += test_rr(s, 'rem', x13, x3, x3, 0x0, -0x41, -0x41)
err += test_rr(s, 'rem', x25, x5, x1, 0x0, -0x80000000, 0x10000)
err += test_rr(s, 'rem', x22, x22, x17, 0x0, 0xb504, 0xb504)
err += test_rr(s, 'rem', x4, x27, x7, 0x55555554, 0x55555554, -0x80000000)
err += test_rr(s, 'rem', x5, x8, x25, 0x200000, 0x200000, 0x0)
err += test_rr(s, 'rem', x14, x4, x0, -0xb503, -0xb503, 0x0)
err += test_rr(s, 'rem', x24, x9, x13, 0x0, 0x6, 0x1)
err += test_rr(s, 'rem', x30, x28, x18, 0x0, 0x0, 0x4)
err += test_rr(s, 'rem', x19, x13, x16, 0x7, 0x7fffffff, -0xa)
err += test_rr(s, 'rem', x20, x7, x9, 0x1, 0x1, 0xb503)
err += test_rr(s, 'rem', x7, x0, x6, 0x0, 0x0, 0x8)
err += test_rr(s, 'rem', x16, x18, x24, 0x0, 0x200, 0x10)
err += test_rr(s, 'rem', x23, x15, x21, -0x1, -0x101, 0x20)
err += test_rr(s, 'rem', x17, x1, x14, -0x1, -0x10000001, 0x40)
err += test_rr(s, 'rem', x1, x19, x15, -0x3, -0x3, 0x80)
err += test_rr(s, 'rem', x15, x11, x12, -0x1, -0x1000001, 0x100)
err += test_rr(s, 'rem', x3, x6, x20, -0x1, -0x800001, 0x200)
err += test_rr(s, 'rem', x10, x25, x30, -0x1, -0x4001, 0x400)

# remu
err += test_rr(s, 'remu', x7, x7, x21, 0x3ff80000, 0xfff7ffff, 0xbfffffff)
err += test_rr(s, 'remu', x30, x9, x9, 0x0, 0x800000, 0x800000)
err += test_rr(s, 'remu', x6, x26, x6, 0xfffffbff, 0xfffffbff, 0x0)
err += test_rr(s, 'remu', x25, x17, x2, 0x400, 0x400, 0xffffffff)
err += test_rr(s, 'remu', x16, x16, x16, 0x0, 0xaaaaaaaa, 0xaaaaaaaa)
err += test_rr(s, 'remu', x5, x2, x1, 0x0, 0x0, 0x66666666)
err += test_rr(s, 'remu', x15, x1, x26, 0x4, 0xffffffff, 0xfffffffb)
err += test_rr(s, 'remu', x0, x29, x15, 0, 0x1, 0xffff7fff)
err += test_rr(s, 'remu', x28, x15, x31, 0x0, 0x1000, 0x2)
err += test_rr(s, 'remu', x21, x3, x7, 0x0, 0x20000000, 0x4)
err += test_rr(s, 'remu', x20, x5, x11, 0x5, 0xd, 0x8)
err += test_rr(s, 'remu', x31, x30, x8, 0x6, 0x6, 0x10)
err += test_rr(s, 'remu', x9, x12, x5, 0xf, 0xf, 0x20)
err += test_rr(s, 'remu', x11, x31, x4, 0x3f, 0xff7fffff, 0x40)
err += test_rr(s, 'remu', x14, x27, x10, 0x40, 0x40, 0x80)
err += test_rr(s, 'remu', x13, x25, x14, 0x0, 0x100, 0x100)
err += test_rr(s, 'remu', x22, x18, x0, 0xa, 0xa, 0x0)
err += test_rr(s, 'remu', x3, x22, x12, 0x103, 0xb503, 0x400)
err += test_rr(s, 'remu', x24, x19, x3, 0x7fb, 0xfffffffb, 0x800)
err += test_rr(s, 'remu', x19, x0, x17, 0x0, 0x0, 0x1000)

print('ALU-tests errors: ' + str(err))

#-------------------------------------------------------------------------------
# some basic tests for load/store instructions
#-------------------------------------------------------------------------------
#s.mem[7]  = -7;  s.mem[8]  = -8; s.mem[9]  = -9; s.mem[10] = 10
#s.mem[11] = -11; s.mem[12] = 12; s.mem[13] = 13; s.x[9] = 9
#LB(s, 1, -1, 9);  print(s.x[1])
#LBU(s, 1, -1, 9); print(s.x[1])
#LH(s, 1, 1, 9);   print(s.x[1]); print(10 - 256*11)
#s.x[2] = -1023
#SH(s, 2, 1, 9); print(s.mem[10]); print(_i8(s.mem[11]))
# TODO: improve these tests, they should be self-checking; or just replace them
# by the official RISC-V test-suite or unit tests

# TODO: execute already compiled hex-files tests from
# https://github.com/ucb-bar/riscv-mini/blob/main/tests/rv32ui-p-add.hex
