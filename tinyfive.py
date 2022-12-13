import numpy as np
import fnmatch

# TinyFive has two parts:
#   - Part I defines all state and instructions of RISC-V (without the
#     instruction bit-encodings).
#   - Part II implements encoding and decoding functions around the instructions
#     defined in part I.
# Part I is sufficient for emulating RISC-V. Part II is only needed if you want
# to emulate the instruction encoding of RISC-V.

# Notes on speed:
#  - TinyFive is not optimized for speed (but for ease-of-use and LOC).
#  - You could use PyPy to speed it up (see e.g. the Pydgin paper for details).
#  - If you only use the upper-case instructions such as ADD(), then TinyFive
#    is very fast because there is no instruction decoding. And you should be
#    able to accelerate it on a GPU or TPU like any other Python code that uses
#    NumPy.
#  - If you use the lower-case instructions with enc() and exe(), then
#    execution of the enc() and dec() functions is slow as they involve look-up
#    and string matching with O(n) complexity where 'n' is the total number of
#    instructions. The current implementations of enc() and dec() are optimized
#    for ease-of-use and readability. A faster implementation would collapse
#    multiple look-ups into one look-up, optimize the pattern-matching for the
#    instruction decoding (bits -> instruction), and change the order of the
#    instructions so that more frequently used instructions are at the top of
#    the list. Here is an older version of TinyFive with a faster dec() function
#    that doesn't use fnmatch and that collapses two look-ups
#    (bits -> instruction and instruction -> uppeer-case instruction):
#    https://github.com/OpenMachine-ai/tinyfive/blob/2aa4987391561c9c6692602ed3fccdeaee333e0b/tinyfive.py

#-------------------------------------------------------------------------------
# Part I
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# create state of CPU: memory 'mem[]', register file 'x[]', program counter 'pc'
class State(object): pass
s = State()

mem_size = 10**6  # memory is 1MB large (TODO: make memory-size variable)
s.mem = np.zeros(mem_size, dtype=np.uint8)  # memory 'mem[]' is unsigned int8
s.x   = np.zeros(32, dtype=np.int32)    # register file 'x[]' is signed int32
s.pc  = 0

#-------------------------------------------------------------------------------
# Base instructions (RV32I)
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
def LH (s,rd,imm,rs1): s.x[rd] = (_i8(s.mem[s.x[rs1] + imm+1]) << 8) + \
                                      s.mem[s.x[rs1] + imm];   _pc(s)
def LHU(s,rd,imm,rs1): s.x[rd] =     (s.mem[s.x[rs1] + imm+1]  << 8) + \
                                      s.mem[s.x[rs1] + imm];   _pc(s)
def LW (s,rd,imm,rs1): s.x[rd] = (_i8(s.mem[s.x[rs1] + imm+3]) << 24) + \
                                     (s.mem[s.x[rs1] + imm+2]  << 16) + \
                                     (s.mem[s.x[rs1] + imm+1]  << 8)  + \
                                      s.mem[s.x[rs1] + imm];   _pc(s)

# store, note the different argument order, example: 'sb rs2, offset(rs1)'
def SB(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm]   =  s.x[rs2] & 0xff; _pc(s)
def SH(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm]   =  s.x[rs2] & 0xff; _pc(s); \
                       s.mem[s.x[rs1] + imm+1] = (s.x[rs2] >> 8)  & 0xff
def SW(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm]   =  s.x[rs2] & 0xff; _pc(s); \
                       s.mem[s.x[rs1] + imm+1] = (s.x[rs2] >> 8)  & 0xff; \
                       s.mem[s.x[rs1] + imm+2] = (s.x[rs2] >> 16) & 0xff; \
                       s.mem[s.x[rs1] + imm+3] = (s.x[rs2] >> 24) & 0xff

# Note: the 3 missing instructions FENCE, ECALL, EBREAK are not needed here

#-------------------------------------------------------------------------------
# M-extension (RV32M)
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
s.f = np.zeros(32, dtype=np.float32)  # register file 'f[]' for F-extension

def f2b(x): return (s.f[x]).view(np.uint32)       # float-to-bits
def b2f(x): return np.uint32(x).view(np.float32)  # bits-to-float

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

def FEQ_S(s,rd,rs1,rs2): s.x[rd] = int(s.f[rs1] == s.f[rs2]); _pc(s)
def FLT_S(s,rd,rs1,rs2): s.x[rd] = int(s.f[rs1] <  s.f[rs2]); _pc(s)
def FLE_S(s,rd,rs1,rs2): s.x[rd] = int(s.f[rs1] <= s.f[rs2]); _pc(s)

def FLW_S(s,rd,imm,rs1): s.f[rd] = b2f((s.mem[s.x[rs1] + imm+3] << 24) + \
                                       (s.mem[s.x[rs1] + imm+2] << 16) + \
                                       (s.mem[s.x[rs1] + imm+1] << 8)  + \
                                        s.mem[s.x[rs1] + imm]);   _pc(s)

def FSW_S(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm] = f2b(rs2) & 0xff;   _pc(s); \
                          s.mem[s.x[rs1] + imm+1] = (f2b(rs2) >> 8)  & 0xff; \
                          s.mem[s.x[rs1] + imm+2] = (f2b(rs2) >> 16) & 0xff; \
                          s.mem[s.x[rs1] + imm+3] = (f2b(rs2) >> 24) & 0xff

def _fsgn(s,rs1,msb): return b2f((msb & 0x80000000) | (f2b(rs1) & 0x7fffffff))
def FSGNJ_S (s,rd,rs1,rs2): s.f[rd] = _fsgn(s, rs1,  f2b(rs2));           _pc(s)
def FSGNJN_S(s,rd,rs1,rs2): s.f[rd] = _fsgn(s, rs1, ~f2b(rs2));           _pc(s)
def FSGNJX_S(s,rd,rs1,rs2): s.f[rd] = _fsgn(s, rs1, f2b(rs2) ^ f2b(rs1)); _pc(s)

def FCVT_S_W (s,rd,rs1): s.f[rd] = np.float32(   s.x[rs1]);   _pc(s)
def FCVT_S_WU(s,rd,rs1): s.f[rd] = np.float32(_u(s.x[rs1]));  _pc(s)
def FCVT_W_S (s,rd,rs1): s.x[rd] = np.int32  (   s.f[rs1]);   _pc(s)
def FCVT_WU_S(s,rd,rs1): s.x[rd] = np.uint32 (   s.f[rs1]);   _pc(s)
def FMV_W_X  (s,rd,rs1): s.f[rd] = b2f(s.x[rs1]);             _pc(s)
def FMV_X_W  (s,rd,rs1): s.x[rd] = f2b(rs1);                  _pc(s)
def FCLASS_S (s,rd,rs1):
  if   np.isneginf(s.f[rs1])    : s.x[rd] = 1       # negative infinity
  elif np.isposinf(s.f[rs1])    : s.x[rd] = 1 << 7  # positive infinity
  elif f2b(rs1) == 0x80000000   : s.x[rd] = 1 << 3  # -0
  elif f2b(rs1) == 0            : s.x[rd] = 1 << 4  # +0
  elif f2b(rs1) == 0x7f800001   : s.x[rd] = 1 << 8  # signaling NaN
  elif f2b(rs1) == 0x7fc00000   : s.x[rd] = 1 << 9  # quiet NaN
  elif (f2b(rs1) >> 23) == 0    : s.x[rd] = 1 << 5  # positive subnormal
  elif (f2b(rs1) >> 23) == 0x100: s.x[rd] = 1 << 2  # negative subnormal
  elif s.f[rs1] < 0.0           : s.x[rd] = 1 << 1  # negative normal
  else                          : s.x[rd] = 1 << 6  # positive normal
  _pc(s)

# TODOs:
#   - add rounding mode (rm) argument. Only rm = 0 is implemented right now.
#     See experimental/rounding_modes.py for more details.
#   - add floating point CSR register

#-------------------------------------------------------------------------------
# Part II
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# decode instruction

dec_dict = {  # below decoder dictionary is copied from table 24.2 of the spec
  '???????_?????_???_0110111': ['U',  'lui'      ],
  '???????_?????_???_0010111': ['U',  'auipc'    ],
  '???????_?????_???_1101111': ['J',  'jal'      ],
  '???????_?????_000_1100111': ['I',  'jalr'     ],
  '???????_?????_000_1100011': ['B',  'beq'      ],
  '???????_?????_001_1100011': ['B',  'bne'      ],
  '???????_?????_100_1100011': ['B',  'blt'      ],
  '???????_?????_101_1100011': ['B',  'bge'      ],
  '???????_?????_110_1100011': ['B',  'bltu'     ],
  '???????_?????_111_1100011': ['B',  'bgeu'     ],
  '???????_?????_000_0000011': ['IL', 'lb'       ],
  '???????_?????_001_0000011': ['IL', 'lh'       ],
  '???????_?????_010_0000011': ['IL', 'lw'       ],
  '???????_?????_100_0000011': ['IL', 'lbu'      ],
  '???????_?????_101_0000011': ['IL', 'lhu'      ],
  '???????_?????_000_0100011': ['S',  'sb'       ],
  '???????_?????_001_0100011': ['S',  'sh'       ],
  '???????_?????_010_0100011': ['S',  'sw'       ],
  '???????_?????_000_0010011': ['I',  'addi'     ],
  '???????_?????_010_0010011': ['I',  'slti'     ],
  '???????_?????_011_0010011': ['I',  'sltiu'    ],
  '???????_?????_100_0010011': ['I',  'xori'     ],
  '???????_?????_110_0010011': ['I',  'ori'      ],
  '???????_?????_111_0010011': ['I',  'andi'     ],
  '0000000_?????_001_0010011': ['R',  'slli'     ],  # use R-type (not I-type)
  '0000000_?????_101_0010011': ['R',  'srli'     ],  # use R-type (not I-type)
  '0100000_?????_101_0010011': ['R',  'srai'     ],  # use R-type (not I-type)
  '0000000_?????_000_0110011': ['R',  'add'      ],
  '0100000_?????_000_0110011': ['R',  'sub'      ],
  '0000000_?????_001_0110011': ['R',  'sll'      ],
  '0000000_?????_010_0110011': ['R',  'slt'      ],
  '0000000_?????_011_0110011': ['R',  'sltu'     ],
  '0000000_?????_100_0110011': ['R',  'xor'      ],
  '0000000_?????_101_0110011': ['R',  'srl'      ],
  '0100000_?????_101_0110011': ['R',  'sra'      ],
  '0000000_?????_110_0110011': ['R',  'or'       ],
  '0000000_?????_111_0110011': ['R',  'and'      ],
  # M-extension
  '0000001_?????_000_0110011': ['R',  'mul'      ],
  '0000001_?????_001_0110011': ['R',  'mulh'     ],
  '0000001_?????_010_0110011': ['R',  'mulhsu'   ],
  '0000001_?????_011_0110011': ['R',  'mulhu'    ],
  '0000001_?????_100_0110011': ['R',  'div'      ],
  '0000001_?????_101_0110011': ['R',  'divu'     ],
  '0000001_?????_110_0110011': ['R',  'rem'      ],
  '0000001_?????_111_0110011': ['R',  'remu'     ],
  # F-extension
  '???????_?????_010_0000111': ['IL', 'flw.s'    ],
  '???????_?????_010_0100111': ['S',  'fsw.s'    ],
  '?????00_?????_000_1000011': ['R4', 'fmadd.s'  ],
  '?????00_?????_000_1000111': ['R4', 'fmsub.s'  ],
  '?????00_?????_000_1001011': ['R4', 'fnmsub.s' ],
  '?????00_?????_000_1001111': ['R4', 'fnmadd.s' ],
  '0000000_?????_000_1010011': ['R',  'fadd.s'   ],
  '0000100_?????_000_1010011': ['R',  'fsub.s'   ],
  '0001000_?????_000_1010011': ['R',  'fmul.s'   ],
  '0001100_?????_000_1010011': ['R',  'fdiv.s'   ],
  '0101100_00000_000_1010011': ['R2', 'fsqrt.s'  ],
  '0010000_?????_000_1010011': ['R',  'fsgnj.s'  ],
  '0010000_?????_001_1010011': ['R',  'fsgnjn.s' ],
  '0010000_?????_010_1010011': ['R',  'fsgnjx.s' ],
  '0010100_?????_000_1010011': ['R',  'fmin.s'   ],
  '0010100_?????_001_1010011': ['R',  'fmax.s'   ],
  '1100000_00000_000_1010011': ['R2', 'fcvt.w.s' ],
  '1100000_00001_000_1010011': ['R2', 'fcvt.wu.s'],
  '1110000_00000_000_1010011': ['R2', 'fmv.x.w'  ],
  '1010000_?????_010_1010011': ['R',  'feq.s'    ],
  '1010000_?????_001_1010011': ['R',  'flt.s'    ],
  '1010000_?????_000_1010011': ['R',  'fle.s'    ],
  '1110000_00000_001_1010011': ['R2', 'fclass.s' ],
  '1101000_00000_000_1010011': ['R2', 'fcvt.s.w' ],
  '1101000_00001_000_1010011': ['R2', 'fcvt.s.wu'],
  '1111000_00000_000_1010011': ['R2', 'fmv.w.x'  ]}

def field(bits, hi, lo):
  """extract bitfields from a bit-array using Verilog bit-indexing order,
  so [0] is the right-most bit (which is opposite order than bitstring),
  and [1:0] are the 2 least significant bits, etc."""
  return bits[len(bits) - 1 - hi : len(bits) - lo]

def bits2u(bits):
  """convert bit-string to unsigned int"""
  return int(bits, 2)

def bits2i(bits):
  """convert bit-string to signed int (subtract constant if msb is 1)"""
  return bits2u(bits) - (1 << len(bits) if bits[0] == '1' else 0)

def dec(bits):
  """decode instruction"""
  opcode = field(bits,  6,  0)
  f3     = field(bits, 14, 12)
  rs2c   = field(bits, 24, 20)  # rs2 code
  f2     = field(bits, 26, 25)
  f7     = field(bits, 31, 25)
  rd     = bits2u(field(bits, 11,  7))
  rs1    = bits2u(field(bits, 19, 15))
  rs2    = bits2u(field(bits, 24, 20))
  rs3    = bits2u(field(bits, 31, 27))
  imm_i  = bits2i(field(bits, 31, 20))                              # I-type
  imm_s  = bits2i(field(bits, 31, 25) + field(bits, 11,  7))        # S-type
  imm_b  = bits2i(field(bits, 31, 31) + field(bits,  7,  7) +
                  field(bits, 30, 25) + field(bits, 11,  8) + '0')  # B-type
  imm_u  = bits2i(field(bits, 31, 12))                              # U-type
  imm_j  = bits2i(field(bits, 31, 31) + field(bits, 19, 12) +
                  field(bits, 20, 20) + field(bits, 30, 21) + '0')  # J-type
  opcode_bits = f7 + '_' + rs2c + '_' + f3 + '_' + opcode

  # decode instruction (opcode_bits -> inst)
  inst = 0
  for k in dec_dict:
    if fnmatch.fnmatch(opcode_bits, k):
      inst = dec_dict[k][1]
      break  # to speed up run-time
  if inst == 0:
    print('ERROR: this instruction is not supported: ' + bits)

  if   inst == 'lui'      : LUI      (s, rd,  imm_u)
  elif inst == 'auipc'    : AUIPC    (s, rd,  imm_u)
  elif inst == 'jal'      : JAL      (s, rd,  imm_j)
  elif inst == 'jalr'     : JALR     (s, rd,  rs1,   imm_i)
  elif inst == 'beq'      : BEQ      (s, rs1, rs2,   imm_b)
  elif inst == 'bne'      : BNE      (s, rs1, rs2,   imm_b)
  elif inst == 'blt'      : BLT      (s, rs1, rs2,   imm_b)
  elif inst == 'bge'      : BGE      (s, rs1, rs2,   imm_b)
  elif inst == 'bltu'     : BLTU     (s, rs1, rs2,   imm_b)
  elif inst == 'bgeu'     : BGEU     (s, rs1, rs2,   imm_b)
  elif inst == 'lb'       : LB       (s, rd,  imm_i, rs1)
  elif inst == 'lh'       : LH       (s, rd,  imm_i, rs1)
  elif inst == 'lw'       : LW       (s, rd,  imm_i, rs1)
  elif inst == 'lbu'      : LBU      (s, rd,  imm_i, rs1)
  elif inst == 'lhu'      : LHU      (s, rd,  imm_i, rs1)
  elif inst == 'sb'       : SB       (s, rs2, imm_s, rs1)
  elif inst == 'sh'       : SH       (s, rs2, imm_s, rs1)
  elif inst == 'sw'       : SW       (s, rs2, imm_s, rs1)
  elif inst == 'addi'     : ADDI     (s, rd,  rs1,   imm_i)
  elif inst == 'slti'     : SLTI     (s, rd,  rs1,   imm_i)
  elif inst == 'sltiu'    : SLTIU    (s, rd,  rs1,   imm_i)
  elif inst == 'xori'     : XORI     (s, rd,  rs1,   imm_i)
  elif inst == 'ori'      : ORI      (s, rd,  rs1,   imm_i)
  elif inst == 'andi'     : ANDI     (s, rd,  rs1,   imm_i)
  elif inst == 'slli'     : SLLI     (s, rd,  rs1,   rs2)
  elif inst == 'srli'     : SRLI     (s, rd,  rs1,   rs2)
  elif inst == 'srai'     : SRAI     (s, rd,  rs1,   rs2)
  elif inst == 'add'      : ADD      (s, rd,  rs1,   rs2)
  elif inst == 'sub'      : SUB      (s, rd,  rs1,   rs2)
  elif inst == 'sll'      : SLL      (s, rd,  rs1,   rs2)
  elif inst == 'slt'      : SLT      (s, rd,  rs1,   rs2)
  elif inst == 'sltu'     : SLTU     (s, rd,  rs1,   rs2)
  elif inst == 'xor'      : XOR      (s, rd,  rs1,   rs2)
  elif inst == 'srl'      : SRL      (s, rd,  rs1,   rs2)
  elif inst == 'sra'      : SRA      (s, rd,  rs1,   rs2)
  elif inst == 'or'       : OR       (s, rd,  rs1,   rs2)
  elif inst == 'and'      : AND      (s, rd,  rs1,   rs2)
  # M-extension
  elif inst == 'mul'      : MUL      (s, rd,  rs1,   rs2)
  elif inst == 'mulh'     : MULH     (s, rd,  rs1,   rs2)
  elif inst == 'mulhsu'   : MULHSU   (s, rd,  rs1,   rs2)
  elif inst == 'mulhu'    : MULHU    (s, rd,  rs1,   rs2)
  elif inst == 'div'      : DIV      (s, rd,  rs1,   rs2)
  elif inst == 'divu'     : DIVU     (s, rd,  rs1,   rs2)
  elif inst == 'rem'      : REM      (s, rd,  rs1,   rs2)
  elif inst == 'remu'     : REMU     (s, rd,  rs1,   rs2)
  # F-extension
  elif inst == 'flw.s'    : FLW_S    (s, rd,  imm_i, rs1)
  elif inst == 'fsw.s'    : FSW_S    (s, rs2, imm_s, rs1)
  elif inst == 'fmadd.s'  : FMADD_S  (s, rd,  rs1,   rs2, rs3)
  elif inst == 'fmsub.s'  : FMSUB_S  (s, rd,  rs1,   rs2, rs3)
  elif inst == 'fnmsub.s' : FNMSUB_S (s, rd,  rs1,   rs2, rs3)
  elif inst == 'fnmadd.s' : FNMADD_S (s, rd,  rs1,   rs2, rs3)
  elif inst == 'fadd.s'   : FADD_S   (s, rd,  rs1,   rs2)
  elif inst == 'fsub.s'   : FSUB_S   (s, rd,  rs1,   rs2)
  elif inst == 'fmul.s'   : FMUL_S   (s, rd,  rs1,   rs2)
  elif inst == 'fdiv.s'   : FDIV_S   (s, rd,  rs1,   rs2)
  elif inst == 'fsgnj.s'  : FSGNJ_S  (s, rd,  rs1,   rs2)
  elif inst == 'fsgnjn.s' : FSGNJN_S (s, rd,  rs1,   rs2)
  elif inst == 'fsgnjx.s' : FSGNJX_S (s, rd,  rs1,   rs2)
  elif inst == 'fmin.s'   : FMIN_S   (s, rd,  rs1,   rs2)
  elif inst == 'fmax.s'   : FMAX_S   (s, rd,  rs1,   rs2)
  elif inst == 'feq.s'    : FEQ_S    (s, rd,  rs1,   rs2)
  elif inst == 'flt.s'    : FLT_S    (s, rd,  rs1,   rs2)
  elif inst == 'fle.s'    : FLE_S    (s, rd,  rs1,   rs2)
  elif inst == 'fsqrt.s'  : FSQRT_S  (s, rd,  rs1)
  elif inst == 'fcvt.w.s' : FCVT_W_S (s, rd,  rs1)
  elif inst == 'fcvt.wu.s': FCVT_WU_S(s, rd,  rs1)
  elif inst == 'fmv.x.w'  : FMV_X_W  (s, rd,  rs1)
  elif inst == 'fclass.s' : FCLASS_S (s, rd,  rs1)
  elif inst == 'fcvt.s.w' : FCVT_S_W (s, rd,  rs1)
  elif inst == 'fcvt.s.wu': FCVT_S_WU(s, rd,  rs1)
  elif inst == 'fmv.w.x'  : FMV_W_X  (s, rd,  rs1)

#-------------------------------------------------------------------------------
# encode function (aka assembler, it's the inverse of the dec() function)

# generate encoder dictionary by inverting the decoder dictionary
# so that key = 'instruction' and value = ['opcode-bits', 'format-type']
enc_dict = {dec_dict[k][1]: [k, dec_dict[k][0]] for k in dec_dict}

def write_i32(s, x, addr):
  """write 32-bit int to memory (takes 4 byte-addresses)"""
  for i in range(0, 4): s.mem[addr + i] = (x >> (8*i)) & 0xff

def read_i32(s, addr):
  """"read 32-bit int from memory"""
  ret = _i8(s.mem[addr + 3]) << 3*8
  for i in range(0, 3): ret += s.mem[addr + i] << i*8
  return ret

def enc(s, inst, arg1, arg2, arg3=0, arg4=0):
  """encode instruction and write into mem[]"""
  [opcode_bits, typ] = enc_dict[inst]
  f7     = opcode_bits[0:7]
  f2     = opcode_bits[5:7]
  rs2c   = opcode_bits[8:13]  # rs2-code
  f3     = opcode_bits[14:17]
  opcode = opcode_bits[18:25]

  # swap arg2 and arg3 if typ == IL (this is needed for all load instructions)
  if typ == 'IL':
    arg2, arg3 = arg3, arg2
    typ = 'I'

  rd    = np.binary_repr(arg1, 5)
  rs1   = np.binary_repr(arg2, 5)
  rs2   = np.binary_repr(arg3, 5)
  rs3   = np.binary_repr(arg4, 5)
  imm_i = np.binary_repr(arg3, 12)
  imm_u = np.binary_repr(arg2, 20)
  imm_s = np.binary_repr(arg2, 12)
  imm_j = np.binary_repr(arg2, 21)
  imm_b = np.binary_repr(arg3, 13)

  # below table is copied from the spec with the following addition:
  # R2-type is same as R-type but with only 2 arguments (rd and rs1)
  if   typ == 'R' : bits = f7       + rs2  + rs1 + f3 + rd + opcode
  elif typ == 'R2': bits = f7       + rs2c + rs1 + f3 + rd + opcode
  elif typ == 'R4': bits = rs3 + f2 + rs2  + rs1 + f3 + rd + opcode
  elif typ == 'I' : bits = imm_i           + rs1 + f3 + rd + opcode
  elif typ == 'U' : bits = imm_u                      + rd + opcode
  elif typ == 'S' : bits = field(imm_s,11,5) + rd + rs2 + f3 + \
                           field(imm_s,4,0) + opcode
  elif typ == 'J' : bits = field(imm_j,20,20) + field(imm_j,10,1) + \
                           field(imm_j,11,11) + field(imm_j,19,12) + rd + opcode
  elif typ == 'B' : bits = field(imm_b,12,12) + field(imm_b,10,5) + rs1 + rd + \
                           f3 + field(imm_b,4,1) + field(imm_b,11,11) + opcode

  # write instruction into memory at address 's.pc'
  write_i32(s, bits2u(bits), s.pc)
  _pc(s)

#-------------------------------------------------------------------------------
# execute code from address 'start', stop execution after n instructions
def exe(s, start, instructions):
  s.pc = start
  for i in range(0, instructions):
    inst = read_i32(s, s.pc)  # fetch instruction from memory
    dec(np.binary_repr(_u(inst), 32))

#-------------------------------------------------------------------------------
# assembler mnemonics
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
def LI(s, rd, imm):
  LUI (s, rd, imm >> 12)
  ADDI(s, rd, rd, imm & 0xfff)

# TODO: add more pseudoinstructions

#-------------------------------------------------------------------------------
# useful functions for accessing state and memory
def clear_cpu(s):
  s.x = np.zeros(32, dtype=np.int32)
  s.pc = 0

def clear_mem(s, start=0, end=mem_size):
  """clear memory from address 'start' to 'end' (excluding 'end')"""
  for i in range(start, end): s.mem[i] = 0

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
