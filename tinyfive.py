import numpy as np
from bitstring import Bits, pack

# This file is divided into two parts:
#   1. Part I defines all state and instructions of RISC-V (without bit-encodings)
#   2. Part II implements encoding and decoding functions around the instructions
#      defined in part I
# Part I is sufficient for emulating RISC-V. Part II is only needed if you want
# to emulate the instruction encoding of RISC-V.

#-------------------------------------------------------------------------------
# Part I
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# create state of CPU: memory 'mem[]', register file 'x[]', program counter 'pc'
#-------------------------------------------------------------------------------
class State(object): pass
s = State()

mem_size = 1000  # size of memory in bytes
s.mem = np.zeros(mem_size, dtype=np.uint8)  # memory 'mem[]' is unsigned int8
s.x   = np.zeros(32, dtype=np.int32)    # register file 'x[]' is signed int32
s.pc  = 0

#-------------------------------------------------------------------------------
# Base instructions (RV32I)
#-------------------------------------------------------------------------------
def _i8(x): return np.int8(x)    # convert to 8-bit signed
def _u(x):  return np.uint32(x)  # convert to 32-bit unsigned

# increment pc by 4 and make sure that x[0] is always 0
def _pc(s):
  s.x[0] = 0
  s.pc += 4

# arithmetic
def ADD (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] + s.x[rs2]; _pc(s)
def ADDI(s,rd,rs1,imm): s.x[rd] = s.x[rs1] + imm;      _pc(s)
def SUB (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] - s.x[rs2]; _pc(s)

# bitwise logical
def XOR (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] ^ s.x[rs2]; _pc(s)
def XORI(s,rd,rs1,imm): s.x[rd] = s.x[rs1] ^ imm;      _pc(s)
def OR  (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] | s.x[rs2]; _pc(s)
def ORI (s,rd,rs1,imm): s.x[rd] = s.x[rs1] | imm;      _pc(s)
def AND (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] & s.x[rs2]; _pc(s)
def ANDI(s,rd,rs1,imm): s.x[rd] = s.x[rs1] & imm;      _pc(s)

# shift (note: 0x1f ensures that only the 5 LSBs are used as shift-amount)
# (For rv64, we will need to use the 6 LSBs so 0x3f)
def SLL (s,rd,rs1,rs2): s.x[rd] = s.x[rs1]     << (s.x[rs2] & 0x1f); _pc(s)
def SRA (s,rd,rs1,rs2): s.x[rd] = s.x[rs1]     >> (s.x[rs2] & 0x1f); _pc(s)
def SRL (s,rd,rs1,rs2): s.x[rd] = _u(s.x[rs1]) >> (s.x[rs2] & 0x1f); _pc(s)
def SLLI(s,rd,rs1,imm): s.x[rd] = s.x[rs1]     << imm;               _pc(s)
def SRAI(s,rd,rs1,imm): s.x[rd] = s.x[rs1]     >> imm;               _pc(s)
def SRLI(s,rd,rs1,imm): s.x[rd] = _u(s.x[rs1]) >> imm;               _pc(s)

# set to 1 if less than
def SLT  (s,rd,rs1,rs2): s.x[rd] = 1 if s.x[rs1]     < s.x[rs2]     else 0; _pc(s)
def SLTI (s,rd,rs1,imm): s.x[rd] = 1 if s.x[rs1]     < imm          else 0; _pc(s)
def SLTU (s,rd,rs1,rs2): s.x[rd] = 1 if _u(s.x[rs1]) < _u(s.x[rs2]) else 0; _pc(s)
def SLTIU(s,rd,rs1,imm): s.x[rd] = 1 if _u(s.x[rs1]) < _u(imm)      else 0; _pc(s)

# branch
def BEQ (s,rs1,rs2,imm): s.pc += imm if s.x[rs1] == s.x[rs2]         else 4
def BNE (s,rs1,rs2,imm): s.pc += imm if s.x[rs1] != s.x[rs2]         else 4
def BLT (s,rs1,rs2,imm): s.pc += imm if s.x[rs1] <  s.x[rs2]         else 4
def BGE (s,rs1,rs2,imm): s.pc += imm if s.x[rs1] >= s.x[rs2]         else 4
def BLTU(s,rs1,rs2,imm): s.pc += imm if _u(s.x[rs1]) <  _u(s.x[rs2]) else 4
def BGEU(s,rs1,rs2,imm): s.pc += imm if _u(s.x[rs1]) >= _u(s.x[rs2]) else 4

# jump
def JAL (s,rd,imm): s.x[rd] = s.pc + 4; s.pc += imm
def JALR(s,rd,rs1,imm): t = s.pc + 4; s.pc = (s.x[rs1] + imm) & ~1; s.x[rd] = t

# load immediate
def LUI  (s,rd,imm): s.x[rd] = imm << 12;          _pc(s)
def AUIPC(s,rd,imm): s.x[rd] = s.pc + (imm << 12); _pc(s)

# load, note the different argument order, example: 'lb rd, offset(rs1)'
def LB (s,rd,imm,rs1): s.x[rd] =  _i8(s.mem[s.x[rs1] + imm]);  _pc(s)
def LBU(s,rd,imm,rs1): s.x[rd] =      s.mem[s.x[rs1] + imm];   _pc(s)
def LH (s,rd,imm,rs1): s.x[rd] = (_i8(s.mem[s.x[rs1] + imm + 1]) << 8) + \
                                      s.mem[s.x[rs1] + imm];   _pc(s)
def LHU(s,rd,imm,rs1): s.x[rd] =     (s.mem[s.x[rs1] + imm + 1]  << 8) + \
                                      s.mem[s.x[rs1] + imm];   _pc(s)
def LW (s,rd,imm,rs1): s.x[rd] = (_i8(s.mem[s.x[rs1] + imm + 3]) << 24) + \
                                     (s.mem[s.x[rs1] + imm + 2]  << 16) + \
                                     (s.mem[s.x[rs1] + imm + 1]  << 8)  + \
                                      s.mem[s.x[rs1] + imm];   _pc(s)

# store, note the different argument order, example: 'sb rs2, offset(rs1)'
def SB(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm] = s.x[rs2] & 0xff;  _pc(s)
def SH(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm] = s.x[rs2] & 0xff;  _pc(s); \
                       s.mem[s.x[rs1] + imm + 1] = (s.x[rs2] >> 8) & 0xff
def SW(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm] = s.x[rs2] & 0xff;  _pc(s); \
                       s.mem[s.x[rs1] + imm + 1] = (s.x[rs2] >> 8)  & 0xff; \
                       s.mem[s.x[rs1] + imm + 2] = (s.x[rs2] >> 16) & 0xff; \
                       s.mem[s.x[rs1] + imm + 3] = (s.x[rs2] >> 24) & 0xff

# Note: the 3 missing instructions FENCE, ECALL, EBREAK are not needed here

#-------------------------------------------------------------------------------
# M-extension (RV32M)
#-------------------------------------------------------------------------------
def _muls(a,b): return np.multiply(   a,     b,  dtype=np.int64)
def _mulu(a,b): return np.multiply(_u(a), _u(b), dtype=np.uint64)
def MUL   (s,rd,rs1,rs2): s.x[rd] = _muls(s.x[rs1],  s.x[rs2]);         _pc(s)
def MULH  (s,rd,rs1,rs2): s.x[rd] = _muls(s.x[rs1],  s.x[rs2])   >> 32; _pc(s)
def MULHSU(s,rd,rs1,rs2): s.x[rd] = _muls(s.x[rs1],_u(s.x[rs2])) >> 32; _pc(s)
def MULHU (s,rd,rs1,rs2): s.x[rd] = _mulu(s.x[rs1],s.x[rs2]) >> _u(32); _pc(s)
# TODO: why is Python integer division '//' and remainder '%' not exactly the same
# as RISC-V 'div' and 'rem'? # For efficient mapping of Python code to RISC-V,
# these basic instructions should be exactly the same
def _div(a,b): return np.fix(a/b).astype(int)
def _rem(a,b): return a - b * _div(a, b)
def DIV (s,rd,rs1,rs2): s.x[rd] = _div(   s.x[rs1],    s.x[rs2]);  _pc(s)
def DIVU(s,rd,rs1,rs2): s.x[rd] = _div(_u(s.x[rs1]),_u(s.x[rs2])); _pc(s)
def REM (s,rd,rs1,rs2): s.x[rd] = _rem(   s.x[rs1],    s.x[rs2]);  _pc(s)
def REMU(s,rd,rs1,rs2): s.x[rd] = _rem(_u(s.x[rs1]),_u(s.x[rs2])); _pc(s)

#-------------------------------------------------------------------------------
# F-extension (RV32F)
#-------------------------------------------------------------------------------
s.f = np.zeros(32, dtype=np.float32)  # register file 'f[]' for F-extension

def FADD_S (s,rd,rs1,rs2): s.f[rd] = s.f[rs1] + s.f[rs2];     _pc(s)
def FSUB_S (s,rd,rs1,rs2): s.f[rd] = s.f[rs1] - s.f[rs2];     _pc(s)
def FMUL_S (s,rd,rs1,rs2): s.f[rd] = s.f[rs1] * s.f[rs2];     _pc(s)
def FDIV_S (s,rd,rs1,rs2): s.f[rd] = s.f[rs1] / s.f[rs2];     _pc(s)
def FSQRT_S(s,rd,rs1)    : s.f[rd] = np.sqrt(s.f[rs1]);       _pc(s)
def FMIN_S (s,rd,rs1,rs2): s.f[rd] = min(s.f[rs1], s.f[rs2]); _pc(s)
def FMAX_S (s,rd,rs1,rs2): s.f[rd] = max(s.f[rs1], s.f[rs2]); _pc(s)

def FMADD_S (s,rd,rs1,rs2,rs3): s.f[rd] =  s.f[rs1] * s.f[rs2] + s.f[rs3]; _pc(s)
def FMSUB_S (s,rd,rs1,rs2,rs3): s.f[rd] =  s.f[rs1] * s.f[rs2] - s.f[rs3]; _pc(s)
def FNMADD_S(s,rd,rs1,rs2,rs3): s.f[rd] = -s.f[rs1] * s.f[rs2] - s.f[rs3]; _pc(s)
def FNMSUB_S(s,rd,rs1,rs2,rs3): s.f[rd] = -s.f[rs1] * s.f[rs2] + s.f[rs3]; _pc(s)

# TODOs:
#   - add missing instructions
#   - add rounding mode (rm) argument. Only rm = 0 is implemented right now.
#     See experimental/rounding_modes.py for more details.
#   - add floating point flags and CSR for config

#-------------------------------------------------------------------------------
# Part II
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# useful functions for accessing state and memory
#-------------------------------------------------------------------------------
def clear_cpu(s):
  s.x = np.zeros(32, dtype=np.int32)
  s.pc = 0

def clear_mem(s, start=0, end=mem_size):
  """clear memory from address 'start' to 'end' (excluding 'end')"""
  for i in range(start, end): s.mem[i] = 0

def write_i32(s, x, addr):
  """write 32-bit int to memory (takes 4 byte-addresses)"""
  for i in range(0, 4): s.mem[addr + i] = (x >> (8*i)) & 0xff

def read_i32(s, addr):
  """"read 32-bit int from memory"""
  ret = _i8(s.mem[addr + 3]) << 3*8
  for i in range(0, 3):
    ret += s.mem[addr + i] << i*8
  return ret

def write_i32_vec(s, vec, start):
  """write i32-vector to memory address 'start'"""
  for i in range(0, np.size(vec)):
    write_i32(s, vec[i], start + 4*i)

def read_i32_vec(s, size, start):
  """read i32-vector of size 'size' from memory address 'start'"""
  ret = np.empty(size, dtype=np.int32)
  for i in range(0, size):
    ret[i] = read_i32(s, start + 4*i)
  return ret

def mem_dump(s, start, size):
  """for debug: dump memory from byteaddress 'start', dump out 'size' bytes"""
  for i in range(start, start + size, 4):
    print('%08x' % _u(read_i32(s, i)))

def dump_state(s):
  print("pc   : %4d" % s.pc)
  for i in range(0, 32):
    print("x[%2d]: %4d" % (i, s.x[i]))

#-------------------------------------------------------------------------------
# decode instruction
#-------------------------------------------------------------------------------
def field(bits, hi, lo):
  """extract bitfields from a bit-array using Verilog bit-indexing order,
  so [0] is the right-most bit (which is opposite order than bitstring),
  and [1:0] are the 2 least significant bits, etc."""
  return bits[len(bits) - 1 - hi : len(bits) - lo]

def dec(inst):
  """decode instruction"""
  opcode = field(inst, 6, 0).bin
  rd     = field(inst, 11, 7).uint
  f3     = field(inst, 14, 12).bin
  rs1    = field(inst, 19, 15).uint
  rs2    = field(inst, 24, 20).uint
  rs3    = field(inst, 31, 27).uint
  f2     = field(inst, 26, 25).bin
  f7     = field(inst, 31, 25).bin
  shamt  = rs2
  imm_i  = field(inst, 31, 20).int                                 # I-type
  imm_s = (field(inst, 31, 25) + field(inst, 11, 7)).int           # S-type
  imm_b = (field(inst, 31, 31) + field(inst, 7, 7) +
           field(inst, 30, 25) + field(inst, 11, 8) + '0b0').int   # B-type
  imm_u =  field(inst, 31, 12).int                                 # U-type
  imm_j = (field(inst, 31, 31) + field(inst, 19, 12) +
           field(inst, 20, 20) + field(inst, 30, 21) + '0b0').int  # J-type

  # below is copied from table 24.2 of RISC-V ISA Spec Vol I
  f3_opc = f3 + '_' + opcode
  f7_f3_ = f7 + '_' + f3_opc
  f2_f3_ = f2 + '_' + f3_opc
  if   opcode ==     '0110111': LUI  (s, rd, imm_u)
  elif opcode ==     '0010111': AUIPC(s, rd, imm_u)
  elif opcode ==     '1101111': JAL  (s, rd, imm_j)
  elif f3_opc == '000_1100111': JALR (s, rd, rs1, imm_i)

  elif f3_opc == '000_1100011': BEQ (s, rs1, rs2, imm_b)
  elif f3_opc == '001_1100011': BNE (s, rs1, rs2, imm_b)
  elif f3_opc == '100_1100011': BLT (s, rs1, rs2, imm_b)
  elif f3_opc == '101_1100011': BGE (s, rs1, rs2, imm_b)
  elif f3_opc == '110_1100011': BLTU(s, rs1, rs2, imm_b)
  elif f3_opc == '111_1100011': BGEU(s, rs1, rs2, imm_b)

  elif f3_opc == '000_0000011': LB (s, rd, imm_i, rs1)
  elif f3_opc == '001_0000011': LH (s, rd, imm_i, rs1)
  elif f3_opc == '010_0000011': LW (s, rd, imm_i, rs1)
  elif f3_opc == '100_0000011': LBU(s, rd, imm_i, rs1)
  elif f3_opc == '101_0000011': LHU(s, rd, imm_i, rs1)

  elif f3_opc == '000_0100011': SB(s, rs2, imm_s, rs1)
  elif f3_opc == '001_0100011': SH(s, rs2, imm_s, rs1)
  elif f3_opc == '010_0100011': SW(s, rs2, imm_s, rs1)

  elif f3_opc == '000_0010011': ADDI (s, rd, rs1, imm_i)
  elif f3_opc == '010_0010011': SLTI (s, rd, rs1, imm_i)
  elif f3_opc == '011_0010011': SLTIU(s, rd, rs1, imm_i)
  elif f3_opc == '100_0010011': XORI (s, rd, rs1, imm_i)
  elif f3_opc == '110_0010011': ORI  (s, rd, rs1, imm_i)
  elif f3_opc == '111_0010011': ANDI (s, rd, rs1, imm_i)

  elif f7_f3_ == '0000000_001_0010011': SLLI(s, rd, rs1, shamt)
  elif f7_f3_ == '0000000_101_0010011': SRLI(s, rd, rs1, shamt)
  elif f7_f3_ == '0100000_101_0010011': SRAI(s, rd, rs1, shamt)

  elif f7_f3_ == '0000000_000_0110011': ADD (s, rd, rs1, rs2)
  elif f7_f3_ == '0100000_000_0110011': SUB (s, rd, rs1, rs2)
  elif f7_f3_ == '0000000_001_0110011': SLL (s, rd, rs1, rs2)
  elif f7_f3_ == '0000000_010_0110011': SLT (s, rd, rs1, rs2)
  elif f7_f3_ == '0000000_011_0110011': SLTU(s, rd, rs1, rs2)
  elif f7_f3_ == '0000000_100_0110011': XOR (s, rd, rs1, rs2)
  elif f7_f3_ == '0000000_101_0110011': SRL (s, rd, rs1, rs2)
  elif f7_f3_ == '0100000_101_0110011': SRA (s, rd, rs1, rs2)
  elif f7_f3_ == '0000000_110_0110011': OR  (s, rd, rs1, rs2)
  elif f7_f3_ == '0000000_111_0110011': AND (s, rd, rs1, rs2)

  # M-extension
  elif f7_f3_ == '0000001_000_0110011': MUL   (s, rd, rs1, rs2)
  elif f7_f3_ == '0000001_001_0110011': MULH  (s, rd, rs1, rs2)
  elif f7_f3_ == '0000001_010_0110011': MULHSU(s, rd, rs1, rs2)
  elif f7_f3_ == '0000001_011_0110011': MULHU (s, rd, rs1, rs2)
  elif f7_f3_ == '0000001_100_0110011': DIV   (s, rd, rs1, rs2)
  elif f7_f3_ == '0000001_101_0110011': DIVU  (s, rd, rs1, rs2)
  elif f7_f3_ == '0000001_110_0110011': REM   (s, rd, rs1, rs2)
  elif f7_f3_ == '0000001_111_0110011': REMU  (s, rd, rs1, rs2)

  # F-extension
  elif f7_f3_ == '0000000_000_1010011': FADD_S (s, rd, rs1, rs2)
  elif f7_f3_ == '0000100_000_1010011': FSUB_S (s, rd, rs1, rs2)
  elif f7_f3_ == '0001000_000_1010011': FMUL_S (s, rd, rs1, rs2)
  elif f7_f3_ == '0001100_000_1010011': FDIV_S (s, rd, rs1, rs2)
  elif f7_f3_ == '0101100_000_1010011': FSQRT_S(s, rd, rs1)
  elif f7_f3_ == '0010100_000_1010011': FMIN_S (s, rd, rs1, rs2)
  elif f7_f3_ == '0010100_001_1010011': FMAX_S (s, rd, rs1, rs2)
  elif f2_f3_ == '00_000_1000011': FMADD_S (s, rd, rs1, rs2, rs3)
  elif f2_f3_ == '00_000_1000111': FMSUB_S (s, rd, rs1, rs2, rs3)
  elif f2_f3_ == '00_000_1001011': FNMSUB_S(s, rd, rs1, rs2, rs3)
  elif f2_f3_ == '00_000_1001111': FNMADD_S(s, rd, rs1, rs2, rs3)

  else:
    print('ERROR: this instruction is not supported: ' + str(inst))

#-------------------------------------------------------------------------------
# encode instruction (this is the inverse function of the dec() function above)
#-------------------------------------------------------------------------------
def r_type(f7, f3, opcode, rd, rs1, rs2):
  return pack('bin:7, uint:5, uint:5, bin:3, uint:5, bin:7',
               f7, rs2, rs1, f3, rd, opcode)

def r4_type(f2, f3, opcode, rd, rs1, rs2, rs3):
  return pack('uint:5, bin:2, uint:5, uint:5, bin:3, uint:5, bin:7',
               rs3, f2, rs2, rs1, f3, rd, opcode)

def i_type(f3, opcode, rd, rs1, imm):
  return pack('int:12, uint:5, bin:3, uint:5, bin:7', imm, rs1, f3, rd, opcode)

def i_ty_u(f3, opcode, rd, rs1, imm):
  """same as i_type, but imm is unsigned (only needed for sltiu)"""
  return pack('uint:12, uint:5, bin:3, uint:5, bin:7', imm, rs1, f3, rd, opcode)

def s_type(f3, opcode, rs2, imm, rs1):
  im = Bits(int=imm, length=12)
  imm11_5 = field(im, 11, 5).uint
  imm4_0  = field(im,  4, 0).uint
  return pack('uint:7, uint:5, uint:5, bin:3, uint:5, bin:7',
              imm11_5, rs2, rs1, f3, imm4_0, opcode)

def b_type(f3, opcode, rs1, rs2, imm):
  im = Bits(int=imm, length=13)
  imm12   = field(im, 12, 12).uint
  imm10_5 = field(im, 10,  5).uint
  imm4_1  = field(im,  4,  1).uint
  imm11   = field(im, 11, 11).uint
  return pack('uint:1, uint:6, uint:5, uint:5, bin:3, uint:4, uint:1, bin:7',
              imm12, imm10_5, rs2, rs1, f3, imm4_1, imm11, opcode)

def u_type(opcode, rd, imm):
  return pack('int:20, uint:5, bin:7', imm, rd, opcode)

def j_type(opcode, rd, imm):
  im = Bits(int=imm, length=21)
  imm20    = field(im, 20, 20).uint
  imm10_1  = field(im, 10,  1).uint
  imm11    = field(im, 11, 11).uint
  imm19_12 = field(im, 19, 12).uint
  return pack('uint:1, uint:10, uint:1, uint:8, uint:5, bin:7',
              imm20, imm10_1, imm11, imm19_12, rd, opcode)

# TODO: consider rewriting the bitstring packing/unpacking code, there
# is perhaps a better way than bitstring (numpy's 'packbits' only works
# on uint8)

def enc(s, inst, arg1, arg2, arg3=0, arg4=0):
  """encode instruction and write into mem[]"""
  if   inst == 'lui'  : st = u_type('0110111', arg1, arg2)
  elif inst == 'auipc': st = u_type('0010111', arg1, arg2)
  elif inst == 'jal'  : st = j_type('1101111', arg1, arg2)
  elif inst == 'jalr' : st = i_type('000', '1100111', arg1, arg2, arg3)

  elif inst == 'beq' : st = b_type('000', '1100011', arg1, arg2, arg3)
  elif inst == 'bne' : st = b_type('001', '1100011', arg1, arg2, arg3)
  elif inst == 'blt' : st = b_type('100', '1100011', arg1, arg2, arg3)
  elif inst == 'bge' : st = b_type('101', '1100011', arg1, arg2, arg3)
  elif inst == 'bltu': st = b_type('110', '1100011', arg1, arg2, arg3)
  elif inst == 'bgeu': st = b_type('111', '1100011', arg1, arg2, arg3)

  # note that arg2 and arg3 for i_type are swapped here
  elif inst == 'lb'  : st = i_type('000', '0000011', arg1, arg3, arg2)
  elif inst == 'lh'  : st = i_type('001', '0000011', arg1, arg3, arg2)
  elif inst == 'lw'  : st = i_type('010', '0000011', arg1, arg3, arg2)
  elif inst == 'lbu' : st = i_type('100', '0000011', arg1, arg3, arg2)
  elif inst == 'lhu' : st = i_type('101', '0000011', arg1, arg3, arg2)

  elif inst == 'sb'  : st = s_type('000', '0100011', arg1, arg2, arg3)
  elif inst == 'sh'  : st = s_type('001', '0100011', arg1, arg2, arg3)
  elif inst == 'sw'  : st = s_type('010', '0100011', arg1, arg2, arg3)

  elif inst == 'addi' : st = i_type('000', '0010011', arg1, arg2, arg3)
  elif inst == 'slti' : st = i_type('010', '0010011', arg1, arg2, arg3)
  elif inst == 'sltiu': st = i_ty_u('011', '0010011', arg1, arg2, arg3)
  elif inst == 'xori' : st = i_type('100', '0010011', arg1, arg2, arg3)
  elif inst == 'ori'  : st = i_type('110', '0010011', arg1, arg2, arg3)
  elif inst == 'andi' : st = i_type('111', '0010011', arg1, arg2, arg3)

  # use r-type (instead of i-type) for below 3 instructions
  elif inst == 'slli': st = r_type('0000000', '001', '0010011', arg1, arg2, arg3)
  elif inst == 'srli': st = r_type('0000000', '101', '0010011', arg1, arg2, arg3)
  elif inst == 'srai': st = r_type('0100000', '101', '0010011', arg1, arg2, arg3)

  elif inst == 'add' : st = r_type('0000000', '000', '0110011', arg1, arg2, arg3)
  elif inst == 'sub' : st = r_type('0100000', '000', '0110011', arg1, arg2, arg3)
  elif inst == 'sll' : st = r_type('0000000', '001', '0110011', arg1, arg2, arg3)
  elif inst == 'slt' : st = r_type('0000000', '010', '0110011', arg1, arg2, arg3)
  elif inst == 'sltu': st = r_type('0000000', '011', '0110011', arg1, arg2, arg3)
  elif inst == 'xor' : st = r_type('0000000', '100', '0110011', arg1, arg2, arg3)
  elif inst == 'srl' : st = r_type('0000000', '101', '0110011', arg1, arg2, arg3)
  elif inst == 'sra' : st = r_type('0100000', '101', '0110011', arg1, arg2, arg3)
  elif inst == 'or'  : st = r_type('0000000', '110', '0110011', arg1, arg2, arg3)
  elif inst == 'and' : st = r_type('0000000', '111', '0110011', arg1, arg2, arg3)

  # M-extension
  elif inst == 'mul'   : st = r_type('0000001', '000', '0110011', arg1, arg2, arg3)
  elif inst == 'mulh'  : st = r_type('0000001', '001', '0110011', arg1, arg2, arg3)
  elif inst == 'mulhsu': st = r_type('0000001', '010', '0110011', arg1, arg2, arg3)
  elif inst == 'mulhu' : st = r_type('0000001', '011', '0110011', arg1, arg2, arg3)
  elif inst == 'div'   : st = r_type('0000001', '100', '0110011', arg1, arg2, arg3)
  elif inst == 'divu'  : st = r_type('0000001', '101', '0110011', arg1, arg2, arg3)
  elif inst == 'rem'   : st = r_type('0000001', '110', '0110011', arg1, arg2, arg3)
  elif inst == 'remu'  : st = r_type('0000001', '111', '0110011', arg1, arg2, arg3)

  # F-extension
  elif inst == 'fadd.s' : st = r_type('0000000', '000', '1010011', arg1, arg2, arg3)
  elif inst == 'fsub.s' : st = r_type('0000100', '000', '1010011', arg1, arg2, arg3)
  elif inst == 'fmul.s' : st = r_type('0001000', '000', '1010011', arg1, arg2, arg3)
  elif inst == 'fdiv.s' : st = r_type('0001100', '000', '1010011', arg1, arg2, arg3)
  elif inst == 'fsqrt.s': st = r_type('0101100', '000', '1010011', arg1, arg2, 0)
  elif inst == 'fmin.s' : st = r_type('0010100', '000', '1010011', arg1, arg2, arg3)
  elif inst == 'fmax.s' : st = r_type('0010100', '001', '1010011', arg1, arg2, arg3)
  elif inst == 'fmadd.s' : st = r4_type('00', '000', '1000011', arg1,arg2,arg3,arg4)
  elif inst == 'fmsub.s' : st = r4_type('00', '000', '1000111', arg1,arg2,arg3,arg4)
  elif inst == 'fnmsub.s': st = r4_type('00', '000', '1001011', arg1,arg2,arg3,arg4)
  elif inst == 'fnmadd.s': st = r4_type('00', '000', '1001111', arg1,arg2,arg3,arg4)

  else:
    print('ERROR: this instruction is not supported ' + inst)

  # write instruction into memory at address 's.pc'
  write_i32(s, st.int, s.pc)
  _pc(s)

#-------------------------------------------------------------------------------
# execute code from address 'start', stop execution after n instructions
#-------------------------------------------------------------------------------
def exe(s, start, instructions):
  s.pc = start
  for i in range(0, instructions):
    inst = read_i32(s, s.pc)
    dec(Bits(uint=int(_u(inst)), length=32))

#-------------------------------------------------------------------------------
# assembler mnemonics
#-------------------------------------------------------------------------------
x0  = 0;  x1  = 1;  x2  = 2;  x3  = 3;  x4  = 4;  x5  = 5;  x6  = 6;  x7  = 7
x8  = 8;  x9  = 9;  x10 = 10; x11 = 11; x12 = 12; x13 = 13; x14 = 14; x15 = 15
x16 = 16; x17 = 17; x18 = 18; x19 = 19; x20 = 20; x21 = 21; x22 = 22; x23 = 23
x24 = 24; x25 = 25; x26 = 26; x27 = 27; x28 = 28; x29 = 29; x30 = 30; x31 = 31
zero= 0;  ra  = 1;  sp  = 2;  gp  = 3;  tp  = 4;  t0  = 5;  t1  = 6;  t2  = 7; fp = 8
s0  = 8;  s1  = 9;  a0  = 10; a1  = 11; a2  = 12; a3  = 13; a4  = 14; a5  = 15
a6  = 16; a7  = 17; s2  = 18; s3  = 19; s4  = 20; s5  = 21; s6  = 22; s7  = 23
s8  = 24; s9  = 25; s10 = 26; s11 = 27; t3  = 28; t4  = 29; t5  = 30; t6  = 31

# for F-extension only
f0  = 0;  f1  = 1;  f2   = 2;  f3   = 3;  f4  = 4;  f5  = 5;  f6   = 6;  f7   = 7
f8  = 8;  f9  = 9;  f10  = 10; f11  = 11; f12 = 12; f13 = 13; f14  = 14; f15  = 15
f16 = 16; f17 = 17; f18  = 18; f19  = 19; f20 = 20; f21 = 21; f22  = 22; f23  = 23
f24 = 24; f25 = 25; f26  = 26; f27  = 27; f28 = 28; f29 = 29; f30  = 30; f31  = 31
ft0 = 0;  ft1 = 1;  ft2  = 2;  ft3  = 3;  ft4 = 4;  ft5 = 5;  ft6  = 6;  ft7  = 7
fs0 = 8;  fs1 = 9;  fa0  = 10; fa1  = 11; fa2 = 12; fa3 = 13; fa4  = 14; fa5  = 15
fa6 = 16; fa7 = 17; fs2  = 18; fs3  = 19; fs4 = 20; fs5 = 21; fs6  = 22; fs7  = 23
fs8 = 24; fs9 = 25; fs10 = 26; fs11 = 27; ft8 = 28; ft9 = 29; ft10 = 30; ft11 = 31

#-------------------------------------------------------------------------------
# pseudoinstructions
#-------------------------------------------------------------------------------
def LI(s, rd, imm):
  LUI (s, rd, imm >> 12)
  ADDI(s, rd, rd, imm & 0xfff)

# TODO: add more pseudoinstructions
