import numpy as np
import fnmatch

# TinyFive has two parts:
#   - Part I defines all state and instructions of RISC-V (without the
#     instruction bit-encodings).
#   - Part II implements encoding and decoding functions around the instructions
#     defined in part I.
# Part I is sufficient for emulating RISC-V. Part II is only needed if you want
# to emulate the instruction encoding of RISC-V.

class machine:
  def __init__(s, mem_size):  # for brevity we use 's' instead of 'self'
    """create state of CPU: memory mem[], regfiles x[] and f[], program counter 'pc'"""
    s.mem = np.zeros(mem_size, dtype=np.uint8)  # memory 'mem[]' is unsigned int8
    s.x   = np.zeros(32, dtype=np.int32)        # regfile 'x[]' is signed int32
    s.f   = np.zeros(32, dtype=np.float32)      # regfile 'f[]' for F-extension
    s.pc  = np.zeros(1, dtype=np.uint32)        # program counter (PC) is uint32
    s.label_dict = {}  # label dictionary (for assembly-code labels)

    # performance counters: ops-counters, regfile-usage
    s.ops = {'total': 0, 'load': 0, 'store': 0, 'mul': 0, 'add': 0, 'madd': 0, 'branch': 0}
    s.x_usage = np.zeros(32, dtype=np.int8)  # track usage of x registers
    s.f_usage = np.zeros(32, dtype=np.int8)  # track usage of f registers

  #-------------------------------------------------------------------------------
  # Part I
  #-------------------------------------------------------------------------------
  def i8(s,x): return np.int8(x)    # convert to 8-bit signed
  def u (s,x): return np.uint32(x)  # convert to 32-bit unsigned

  def ipc(s, incr=4):
    """increment pc by 'incr' and make sure that x[0] is always 0"""
    s.x[0] = 0
    s.pc += incr

  #-------------------------------------------------------------------------------
  # Base instructions (RV32I)

  # arithmetic
  def ADD (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] + s.x[rs2]; s.ipc()
  def ADDI(s,rd,rs1,imm): s.x[rd] = s.x[rs1] + imm;      s.ipc()
  def SUB (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] - s.x[rs2]; s.ipc()

  # bitwise logical
  def XOR (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] ^ s.x[rs2]; s.ipc()
  def XORI(s,rd,rs1,imm): s.x[rd] = s.x[rs1] ^ imm;      s.ipc()
  def OR  (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] | s.x[rs2]; s.ipc()
  def ORI (s,rd,rs1,imm): s.x[rd] = s.x[rs1] | imm;      s.ipc()
  def AND (s,rd,rs1,rs2): s.x[rd] = s.x[rs1] & s.x[rs2]; s.ipc()
  def ANDI(s,rd,rs1,imm): s.x[rd] = s.x[rs1] & imm;      s.ipc()

  # shift (note: 0x1f ensures that only the 5 LSBs are used as shift-amount)
  # (For rv64, we would use the 6 LSBs, so 0x3f)
  def SLL (s,rd,rs1,rs2): s.x[rd] = s.x[rs1]      << (s.x[rs2] & 0x1f); s.ipc()
  def SRA (s,rd,rs1,rs2): s.x[rd] = s.x[rs1]      >> (s.x[rs2] & 0x1f); s.ipc()
  def SRL (s,rd,rs1,rs2): s.x[rd] = s.u(s.x[rs1]) >> (s.x[rs2] & 0x1f); s.ipc()
  def SLLI(s,rd,rs1,imm): s.x[rd] = s.x[rs1]      << imm;               s.ipc()
  def SRAI(s,rd,rs1,imm): s.x[rd] = s.x[rs1]      >> imm;               s.ipc()
  def SRLI(s,rd,rs1,imm): s.x[rd] = s.u(s.x[rs1]) >> imm;               s.ipc()

  # set to 1 if less than
  def SLT  (s,rd,rs1,rs2): s.x[rd] = 1 if s.x[rs1]      < s.x[rs2]      else 0; s.ipc()
  def SLTI (s,rd,rs1,imm): s.x[rd] = 1 if s.x[rs1]      < imm           else 0; s.ipc()
  def SLTU (s,rd,rs1,rs2): s.x[rd] = 1 if s.u(s.x[rs1]) < s.u(s.x[rs2]) else 0; s.ipc()
  def SLTIU(s,rd,rs1,imm): s.x[rd] = 1 if s.u(s.x[rs1]) < s.u(imm)      else 0; s.ipc()

  # branch
  def BEQ (s,rs1,rs2,imm): s.ipc(imm if s.x[rs1]      == s.x[rs2]      else 4)
  def BNE (s,rs1,rs2,imm): s.ipc(imm if s.x[rs1]      != s.x[rs2]      else 4)
  def BLT (s,rs1,rs2,imm): s.ipc(imm if s.x[rs1]      <  s.x[rs2]      else 4)
  def BGE (s,rs1,rs2,imm): s.ipc(imm if s.x[rs1]      >= s.x[rs2]      else 4)
  def BLTU(s,rs1,rs2,imm): s.ipc(imm if s.u(s.x[rs1]) <  s.u(s.x[rs2]) else 4)
  def BGEU(s,rs1,rs2,imm): s.ipc(imm if s.u(s.x[rs1]) >= s.u(s.x[rs2]) else 4)

  # jump
  def JAL (s,rd,imm): s.x[rd] = s.pc + 4; s.ipc(imm)
  def JALR(s,rd,rs1,imm): t = s.pc + 4; s.pc = (s.x[rs1] + imm) & ~1; s.x[rd] = t; s.x[0] = 0

  # load immediate
  def LUI  (s,rd,imm): s.x[rd] = imm << 12;          s.ipc()
  def AUIPC(s,rd,imm): s.x[rd] = s.pc + (imm << 12); s.ipc()

  # load, note the different argument order, example: 'lb rd, offset(rs1)'
  def LB (s,rd,imm,rs1): s.x[rd] =  s.i8(s.mem[s.x[rs1] + imm]);  s.ipc()
  def LBU(s,rd,imm,rs1): s.x[rd] =       s.mem[s.x[rs1] + imm];   s.ipc()
  def LH (s,rd,imm,rs1): s.x[rd] = (s.i8(s.mem[s.x[rs1] + imm+1]) << 8) + \
                                         s.mem[s.x[rs1] + imm];   s.ipc()
  def LHU(s,rd,imm,rs1): s.x[rd] =      (s.mem[s.x[rs1] + imm+1]  << 8) + \
                                         s.mem[s.x[rs1] + imm];   s.ipc()
  def LW (s,rd,imm,rs1): s.x[rd] = (s.i8(s.mem[s.x[rs1] + imm+3]) << 24) + \
                                        (s.mem[s.x[rs1] + imm+2]  << 16) + \
                                        (s.mem[s.x[rs1] + imm+1]  << 8)  + \
                                         s.mem[s.x[rs1] + imm];   s.ipc()

  # store, note the different argument order, example: 'sb rs2, offset(rs1)'
  def SB(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm]   =  s.x[rs2] & 0xff; s.ipc()
  def SH(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm]   =  s.x[rs2] & 0xff; s.ipc(); \
                         s.mem[s.x[rs1] + imm+1] = (s.x[rs2] >> 8)  & 0xff
  def SW(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm]   =  s.x[rs2] & 0xff; s.ipc(); \
                         s.mem[s.x[rs1] + imm+1] = (s.x[rs2] >> 8)  & 0xff; \
                         s.mem[s.x[rs1] + imm+2] = (s.x[rs2] >> 16) & 0xff; \
                         s.mem[s.x[rs1] + imm+3] = (s.x[rs2] >> 24) & 0xff

  # the 3 missing instructions FENCE, ECALL, EBREAK are not needed here

  # TODO: when using the above uppercase instructions, then there is no check if
  # an immediate exceeds the 12-bit limit. Therefore, consider replacing 'imm'
  # by 'ci(imm)', where ci() is a def that raises an error flag if the immediate
  # exceeds the 12-bit limit.

  #-------------------------------------------------------------------------------
  # M-extension (RV32M)
  def muls(s,a,b): return np.multiply(    a,      b,  dtype=np.int64)
  def mulu(s,a,b): return np.multiply(s.u(a), s.u(b), dtype=np.uint64)
  def MUL   (s,rd,rs1,rs2): s.x[rd] = s.muls(s.x[rs1],    s.x[rs2]);        s.ipc()
  def MULH  (s,rd,rs1,rs2): s.x[rd] = s.muls(s.x[rs1],    s.x[rs2])  >> 32; s.ipc()
  def MULHSU(s,rd,rs1,rs2): s.x[rd] = s.muls(s.x[rs1],s.u(s.x[rs2])) >> 32; s.ipc()
  def MULHU (s,rd,rs1,rs2): s.x[rd] = s.mulu(s.x[rs1],s.x[rs2]) >> s.u(32); s.ipc()
  # TODO: why is Python integer division '//' and remainder '%' not exactly the same
  # as RISC-V 'div' and 'rem'? # For efficient mapping of Python code to RISC-V,
  # these basic instructions should be exactly the same
  def div(s,a,b): return np.fix(a/b).astype(int)
  def rem(s,a,b): return a - b * s.div(a, b)
  def DIV (s,rd,rs1,rs2): s.x[rd] = s.div(    s.x[rs1],      s.x[rs2]);  s.ipc()
  def DIVU(s,rd,rs1,rs2): s.x[rd] = s.div(s.u(s.x[rs1]), s.u(s.x[rs2])); s.ipc()
  def REM (s,rd,rs1,rs2): s.x[rd] = s.rem(   s.x[rs1],       s.x[rs2]);  s.ipc()
  def REMU(s,rd,rs1,rs2): s.x[rd] = s.rem(s.u(s.x[rs1]), s.u(s.x[rs2])); s.ipc()

  #-------------------------------------------------------------------------------
  # F-extension (RV32F)
  def FADD_S (s,rd,rs1,rs2): s.f[rd] = s.f[rs1] + s.f[rs2];     s.ipc()
  def FSUB_S (s,rd,rs1,rs2): s.f[rd] = s.f[rs1] - s.f[rs2];     s.ipc()
  def FMUL_S (s,rd,rs1,rs2): s.f[rd] = s.f[rs1] * s.f[rs2];     s.ipc()
  def FDIV_S (s,rd,rs1,rs2): s.f[rd] = s.f[rs1] / s.f[rs2];     s.ipc()
  def FSQRT_S(s,rd,rs1)    : s.f[rd] = np.sqrt(s.f[rs1]);       s.ipc()
  def FMIN_S (s,rd,rs1,rs2): s.f[rd] = min(s.f[rs1], s.f[rs2]); s.ipc()
  def FMAX_S (s,rd,rs1,rs2): s.f[rd] = max(s.f[rs1], s.f[rs2]); s.ipc()

  def FMADD_S (s,rd,rs1,rs2,rs3): s.f[rd] =  s.f[rs1] * s.f[rs2] + s.f[rs3]; s.ipc()
  def FMSUB_S (s,rd,rs1,rs2,rs3): s.f[rd] =  s.f[rs1] * s.f[rs2] - s.f[rs3]; s.ipc()
  def FNMADD_S(s,rd,rs1,rs2,rs3): s.f[rd] = -s.f[rs1] * s.f[rs2] - s.f[rs3]; s.ipc()
  def FNMSUB_S(s,rd,rs1,rs2,rs3): s.f[rd] = -s.f[rs1] * s.f[rs2] + s.f[rs3]; s.ipc()

  def FEQ_S(s,rd,rs1,rs2): s.x[rd] = int(s.f[rs1] == s.f[rs2]); s.ipc()
  def FLT_S(s,rd,rs1,rs2): s.x[rd] = int(s.f[rs1] <  s.f[rs2]); s.ipc()
  def FLE_S(s,rd,rs1,rs2): s.x[rd] = int(s.f[rs1] <= s.f[rs2]); s.ipc()

  def f2b(s,x): return (s.f[x]).view(np.uint32)       # float-to-bits
  def b2f(s,x): return np.uint32(x).view(np.float32)  # bits-to-float
  def FLW_S(s,rd,imm,rs1): s.f[rd] = s.b2f((s.mem[s.x[rs1] + imm+3] << 24) + \
                                           (s.mem[s.x[rs1] + imm+2] << 16) + \
                                           (s.mem[s.x[rs1] + imm+1] << 8)  + \
                                            s.mem[s.x[rs1] + imm]);   s.ipc()

  def FSW_S(s,rs2,imm,rs1): s.mem[s.x[rs1] + imm] = s.f2b(rs2) & 0xff;  s.ipc(); \
                            s.mem[s.x[rs1] + imm+1] = (s.f2b(rs2) >> 8)  & 0xff; \
                            s.mem[s.x[rs1] + imm+2] = (s.f2b(rs2) >> 16) & 0xff; \
                            s.mem[s.x[rs1] + imm+3] = (s.f2b(rs2) >> 24) & 0xff

  def fsgn(s,rs1,msb): return s.b2f((msb & 0x80000000) | (s.f2b(rs1) & 0x7fffffff))
  def FSGNJ_S (s,rd,rs1,rs2): s.f[rd] = s.fsgn(rs1,  s.f2b(rs2));             s.ipc()
  def FSGNJN_S(s,rd,rs1,rs2): s.f[rd] = s.fsgn(rs1, ~s.f2b(rs2));             s.ipc()
  def FSGNJX_S(s,rd,rs1,rs2): s.f[rd] = s.fsgn(rs1, s.f2b(rs2) ^ s.f2b(rs1)); s.ipc()

  def FCVT_S_W (s,rd,rs1): s.f[rd] = np.float32(    s.x[rs1]);   s.ipc()
  def FCVT_S_WU(s,rd,rs1): s.f[rd] = np.float32(s.u(s.x[rs1]));  s.ipc()
  def FCVT_W_S (s,rd,rs1): s.x[rd] = np.int32  (    s.f[rs1]);   s.ipc()
  def FCVT_WU_S(s,rd,rs1): s.x[rd] = np.uint32 (    s.f[rs1]);   s.ipc()
  def FMV_W_X  (s,rd,rs1): s.f[rd] = s.b2f(s.x[rs1]);            s.ipc()
  def FMV_X_W  (s,rd,rs1): s.x[rd] = s.f2b(rs1);                 s.ipc()
  def FCLASS_S (s,rd,rs1):
    if   np.isneginf(s.f[rs1])      : s.x[rd] = 1       # negative infinity
    elif np.isposinf(s.f[rs1])      : s.x[rd] = 1 << 7  # positive infinity
    elif s.f2b(rs1) == 0x80000000   : s.x[rd] = 1 << 3  # -0
    elif s.f2b(rs1) == 0            : s.x[rd] = 1 << 4  # +0
    elif s.f2b(rs1) == 0x7f800001   : s.x[rd] = 1 << 8  # signaling NaN
    elif s.f2b(rs1) == 0x7fc00000   : s.x[rd] = 1 << 9  # quiet NaN
    elif (s.f2b(rs1) >> 23) == 0    : s.x[rd] = 1 << 5  # positive subnormal
    elif (s.f2b(rs1) >> 23) == 0x100: s.x[rd] = 1 << 2  # negative subnormal
    elif s.f[rs1] < 0.0             : s.x[rd] = 1 << 1  # negative normal
    else                            : s.x[rd] = 1 << 6  # positive normal
    s.ipc()

  # TODOs:
  #   - add rounding mode (rm) argument. Only rm = 0 is implemented right now.
  #     See misc/experimental_rounding.py for more details.
  #   - add floating point CSR register

  #-------------------------------------------------------------------------------
  # Xom-extension

  # TODO: add OpenMachine custom instructions here

  #-------------------------------------------------------------------------------
  # Part II
  #-------------------------------------------------------------------------------

  #-------------------------------------------------------------------------------
  # decode instruction

  def field(s, bits, hi, lo):
    """extract bitfields from a bit-array using Verilog bit-indexing order,
    so [0] is the right-most bit (which is opposite order than bitstring),
    and [1:0] are the 2 least significant bits, etc."""
    return bits[len(bits) - 1 - hi : len(bits) - lo]

  def bits2u(s, bits):
    """convert bit-string to unsigned int"""
    return int(bits, 2)

  def bits2i(s, bits):
    """convert bit-string to signed int (subtract constant if msb is 1)"""
    return s.bits2u(bits) - (1 << len(bits) if bits[0] == '1' else 0)

  def dec(s, bits):
    """decode instruction"""
    opcode = s.field(bits,  6,  0)
    f3     = s.field(bits, 14, 12)
    rs2c   = s.field(bits, 24, 20)  # rs2 code
    f2     = s.field(bits, 26, 25)
    f7     = s.field(bits, 31, 25)
    rd     = s.bits2u(s.field(bits, 11,  7))
    rs1    = s.bits2u(s.field(bits, 19, 15))
    rs2    = s.bits2u(s.field(bits, 24, 20))
    rs3    = s.bits2u(s.field(bits, 31, 27))
    imm_i  = s.bits2i(s.field(bits, 31, 20))                                # I-type
    imm_s  = s.bits2i(s.field(bits, 31, 25) + s.field(bits, 11,  7))        # S-type
    imm_b  = s.bits2i(s.field(bits, 31, 31) + s.field(bits,  7,  7) +
                      s.field(bits, 30, 25) + s.field(bits, 11,  8) + '0')  # B-type
    imm_u  = s.bits2i(s.field(bits, 31, 12))                                # U-type
    imm_j  = s.bits2i(s.field(bits, 31, 31) + s.field(bits, 19, 12) +
                      s.field(bits, 20, 20) + s.field(bits, 30, 21) + '0')  # J-type
    opcode_bits = f7 + '_' + rs2c + '_' + f3 + '_' + opcode

    # decode instruction (opcode_bits -> inst)
    inst = 0
    for k in dec_dict:
      if fnmatch.fnmatch(opcode_bits, k):
        inst = dec_dict[k][1]
        break  # to speed up run-time
    if inst == 0:
      print('ERROR: this instruction is not supported: ' + bits)

    if   inst == 'lui'      : s.LUI      (rd,  imm_u)
    elif inst == 'auipc'    : s.AUIPC    (rd,  imm_u)
    elif inst == 'jal'      : s.JAL      (rd,  imm_j)
    elif inst == 'jalr'     : s.JALR     (rd,  rs1,   imm_i)
    elif inst == 'beq'      : s.BEQ      (rs1, rs2,   imm_b)
    elif inst == 'bne'      : s.BNE      (rs1, rs2,   imm_b)
    elif inst == 'blt'      : s.BLT      (rs1, rs2,   imm_b)
    elif inst == 'bge'      : s.BGE      (rs1, rs2,   imm_b)
    elif inst == 'bltu'     : s.BLTU     (rs1, rs2,   imm_b)
    elif inst == 'bgeu'     : s.BGEU     (rs1, rs2,   imm_b)
    elif inst == 'lb'       : s.LB       (rd,  imm_i, rs1)
    elif inst == 'lh'       : s.LH       (rd,  imm_i, rs1)
    elif inst == 'lw'       : s.LW       (rd,  imm_i, rs1)
    elif inst == 'lbu'      : s.LBU      (rd,  imm_i, rs1)
    elif inst == 'lhu'      : s.LHU      (rd,  imm_i, rs1)
    elif inst == 'sb'       : s.SB       (rs2, imm_s, rs1)
    elif inst == 'sh'       : s.SH       (rs2, imm_s, rs1)
    elif inst == 'sw'       : s.SW       (rs2, imm_s, rs1)
    elif inst == 'addi'     : s.ADDI     (rd,  rs1,   imm_i)
    elif inst == 'slti'     : s.SLTI     (rd,  rs1,   imm_i)
    elif inst == 'sltiu'    : s.SLTIU    (rd,  rs1,   imm_i)
    elif inst == 'xori'     : s.XORI     (rd,  rs1,   imm_i)
    elif inst == 'ori'      : s.ORI      (rd,  rs1,   imm_i)
    elif inst == 'andi'     : s.ANDI     (rd,  rs1,   imm_i)
    elif inst == 'slli'     : s.SLLI     (rd,  rs1,   rs2)
    elif inst == 'srli'     : s.SRLI     (rd,  rs1,   rs2)
    elif inst == 'srai'     : s.SRAI     (rd,  rs1,   rs2)
    elif inst == 'add'      : s.ADD      (rd,  rs1,   rs2)
    elif inst == 'sub'      : s.SUB      (rd,  rs1,   rs2)
    elif inst == 'sll'      : s.SLL      (rd,  rs1,   rs2)
    elif inst == 'slt'      : s.SLT      (rd,  rs1,   rs2)
    elif inst == 'sltu'     : s.SLTU     (rd,  rs1,   rs2)
    elif inst == 'xor'      : s.XOR      (rd,  rs1,   rs2)
    elif inst == 'srl'      : s.SRL      (rd,  rs1,   rs2)
    elif inst == 'sra'      : s.SRA      (rd,  rs1,   rs2)
    elif inst == 'or'       : s.OR       (rd,  rs1,   rs2)
    elif inst == 'and'      : s.AND      (rd,  rs1,   rs2)
    # M-extension
    elif inst == 'mul'      : s.MUL      (rd,  rs1,   rs2)
    elif inst == 'mulh'     : s.MULH     (rd,  rs1,   rs2)
    elif inst == 'mulhsu'   : s.MULHSU   (rd,  rs1,   rs2)
    elif inst == 'mulhu'    : s.MULHU    (rd,  rs1,   rs2)
    elif inst == 'div'      : s.DIV      (rd,  rs1,   rs2)
    elif inst == 'divu'     : s.DIVU     (rd,  rs1,   rs2)
    elif inst == 'rem'      : s.REM      (rd,  rs1,   rs2)
    elif inst == 'remu'     : s.REMU     (rd,  rs1,   rs2)
    # F-extension
    elif inst == 'flw.s'    : s.FLW_S    (rd,  imm_i, rs1)
    elif inst == 'fsw.s'    : s.FSW_S    (rs2, imm_s, rs1)
    elif inst == 'fmadd.s'  : s.FMADD_S  (rd,  rs1,   rs2, rs3)
    elif inst == 'fmsub.s'  : s.FMSUB_S  (rd,  rs1,   rs2, rs3)
    elif inst == 'fnmsub.s' : s.FNMSUB_S (rd,  rs1,   rs2, rs3)
    elif inst == 'fnmadd.s' : s.FNMADD_S (rd,  rs1,   rs2, rs3)
    elif inst == 'fadd.s'   : s.FADD_S   (rd,  rs1,   rs2)
    elif inst == 'fsub.s'   : s.FSUB_S   (rd,  rs1,   rs2)
    elif inst == 'fmul.s'   : s.FMUL_S   (rd,  rs1,   rs2)
    elif inst == 'fdiv.s'   : s.FDIV_S   (rd,  rs1,   rs2)
    elif inst == 'fsgnj.s'  : s.FSGNJ_S  (rd,  rs1,   rs2)
    elif inst == 'fsgnjn.s' : s.FSGNJN_S (rd,  rs1,   rs2)
    elif inst == 'fsgnjx.s' : s.FSGNJX_S (rd,  rs1,   rs2)
    elif inst == 'fmin.s'   : s.FMIN_S   (rd,  rs1,   rs2)
    elif inst == 'fmax.s'   : s.FMAX_S   (rd,  rs1,   rs2)
    elif inst == 'feq.s'    : s.FEQ_S    (rd,  rs1,   rs2)
    elif inst == 'flt.s'    : s.FLT_S    (rd,  rs1,   rs2)
    elif inst == 'fle.s'    : s.FLE_S    (rd,  rs1,   rs2)
    elif inst == 'fsqrt.s'  : s.FSQRT_S  (rd,  rs1)
    elif inst == 'fcvt.w.s' : s.FCVT_W_S (rd,  rs1)
    elif inst == 'fcvt.wu.s': s.FCVT_WU_S(rd,  rs1)
    elif inst == 'fmv.x.w'  : s.FMV_X_W  (rd,  rs1)
    elif inst == 'fclass.s' : s.FCLASS_S (rd,  rs1)
    elif inst == 'fcvt.s.w' : s.FCVT_S_W (rd,  rs1)
    elif inst == 'fcvt.s.wu': s.FCVT_S_WU(rd,  rs1)
    elif inst == 'fmv.w.x'  : s.FMV_W_X  (rd,  rs1)

    # update ops counters
    s.ops['total'] += 1
    if inst in ['lb', 'lh', 'lw', 'lbu', 'lhu', 'flw.s']:
      s.ops['load'] += 1
    elif inst in ['sb', 'sh', 'sw', 'fsw.s']:
      s.ops['store'] += 1
    elif inst in ['mul', 'mulh', 'mulhsu', 'mulhu', 'fmul.s']:
      s.ops['mul'] += 1
    elif inst in ['add', 'addi', 'sub', 'fadd.s', 'fsub.s']:
      s.ops['add'] += 1
    elif inst in ['fmadd.s', 'fmsub.s', 'fnmadd.s', 'fnmsub.s']:
      s.ops['madd'] += 1
    elif inst in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu']:
      s.ops['branch'] += 1

    # update register-file usage bins based on 'rd' value
    # exclude instructions that don't have an 'rd' field
    if inst not in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'sb', 'sh', 'sw', 'fsw.s']:
      if inst in ['fadd.s', 'fsub.s', 'fmul.s', 'fdiv.s', 'fsqrt.s', 'fmin.s', 'fmax.s',
                  'fmadd.s', 'fmsub.s', 'fnmadd.s', 'fnmsub.s', 'flw.s', 'fsgnj.s',
                  'fsgnjn.s', 'fsgnjx.s', 'fcvt.s.w', 'fcvt.s.wu', 'fmv.w.x']:
        s.f_usage[rd] = 1
      else:
        s.x_usage[rd] = 1

  #-------------------------------------------------------------------------------
  # assembler function asm() (it's the inverse of the dec() function)

  def write_i32(s, x, addr):
    """write 32-bit int to memory (takes 4 byte-addresses)"""
    for i in range(4): s.mem[addr + i] = (x >> (8*i)) & 0xff

  def read_i32(s, addr):
    """"read 32-bit int from memory"""
    ret = s.i8(s.mem[addr + 3]) << 3*8
    for i in range(3): ret += s.mem[addr + i] << i*8
    return ret

  def lbl(s, name):
    """add new label 'name' to the label dictionary"""
    s.label_dict.update({name: s.pc})

  def look_up_label(s, arg):
    """look up label if argument is a string"""
    if isinstance(arg, str):
      arg = s.label_dict[arg]
    return arg

  def asm(s, inst, arg1, arg2, arg3=0, arg4=0):
    """encode instruction and write into mem[]"""
    [opcode_bits, typ] = asm_dict[inst]
    f7     = opcode_bits[0:7]
    f2     = opcode_bits[5:7]
    rs2c   = opcode_bits[8:13]  # rs2-code
    f3     = opcode_bits[14:17]
    opcode = opcode_bits[18:25]

    # swap arg2 and arg3 if typ == IL (this is needed for all load instructions)
    if typ == 'IL':
      arg2, arg3 = arg3, arg2
      typ = 'I'

    # for branch and jump instructions with relative immediates, look up label
    # and calculate the immediate value by subtracting the current PC value from arg
    # TODO: double-check and verify this, especially for AUIPC and JAL
    if typ == 'B':
      arg3 = s.look_up_label(arg3)
      arg3 -= s.pc
    if inst in ['auipc', 'jal']:
      arg2 = s.look_up_label(arg2)
      arg2 -= s.pc

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
    elif typ == 'S' : bits = s.field(imm_s,11,5) + rd + rs2 + f3 + \
                             s.field(imm_s,4,0) + opcode
    elif typ == 'J' : bits = s.field(imm_j,20,20) + s.field(imm_j,10,1) + \
                             s.field(imm_j,11,11) + s.field(imm_j,19,12) + rd + opcode
    elif typ == 'B' : bits = s.field(imm_b,12,12) + s.field(imm_b,10,5) + rs1 + rd + \
                        f3 + s.field(imm_b,4,1)   + s.field(imm_b,11,11) + opcode

    # write instruction into memory at address 's.pc'
    s.write_i32(s.bits2u(bits), s.pc)
    s.ipc()

  def exe(s, start, end=None, instructions=0):
    """execute code from address 'start', stop execution after n 'instructions', or
    stop when pc reaches 'end' address"""
    s.pc = s.look_up_label(start)
    if end is None:  # this is for the case where argument 'instructions' is used
      for i in range(instructions):
        inst = s.read_i32(s.pc)  # fetch instruction from memory
        s.dec(np.binary_repr(s.u(inst), 32))
    else:  # this is for the case where argument 'end' is used
      while s.pc != s.look_up_label(end):
        inst = s.read_i32(s.pc)  # fetch instruction from memory
        s.dec(np.binary_repr(s.u(inst), 32))

  #-------------------------------------------------------------------------------
  # pseudoinstructions

  def hi20(s, val): return (val + 0x800) >> 12  # higher 20 bits (+1 if val[11]==1)
  def lo12(s, val):
    return (val & 0x7ff) - (val & 0x800) # lower 12 bits and convert to signed int12
  # above functions are similar to LLVM
  # https://github.com/llvm/llvm-project/blob/main/lld/ELF/Arch/RISCV.cpp
  # note that ADDI interprets the immediate as a 12-bit signed. So if bit[11] is
  # set (i.e. value -2048 = -0x800), then we have to add +1 to the upper 20 bits
  # (i.e. +0x1000) so that (hi20 >> 12) + lo12 is the same as 'val'. The term
  # '(val + 0x800) >> 12' in hi20 def will only add +1 if bit val[11] is set.
  # Example: 0xdeadbeef -> hi20 = 0xdeadc; lo12 = -273

  def LI(s, rd, imm):
    s.LUI (rd,     s.hi20(imm))
    s.ADDI(rd, rd, s.lo12(imm))
  # TODO: add a corresponding lower-case instruction

  # TODO: add more pseudoinstructions

  #-------------------------------------------------------------------------------
  # useful functions for accessing state and memory

  def clear_cpu(s):
    s.x = np.zeros(32, dtype=np.int32)
    s.f = np.zeros(32, dtype=np.float32)
    s.pc = 0
    s.label_dict = {}
    s.ops = {'total': 0, 'load': 0, 'store': 0, 'mul': 0, 'add': 0, 'madd': 0, 'branch': 0}
    s.x_usage.fill(0)
    s.f_usage.fill(0)
    # TODO: add an option that initializes all registers with 'np.empty' or
    # random data so we can find bugs that are hidden by register initialization

  def clear_mem(s, start=0, end=None):
    """clear memory from address 'start' to 'end' (excluding 'end')"""
    if end is None:
      end = np.size(s.mem)  # use maximum memory address by default
    for i in range(start, end): s.mem[i] = 0

  # TODO: replace write_* and read_* defs by generic 'write_mem' and 'read_mem'
  # that work for any tensor.  For read, have a size-parameter that can be set
  # to e.g. (4, 4) for a 4x4 matrix
  def write_i32_vec(s, vec, start):
    """write i32-vector to memory address 'start'"""
    for i in range(np.size(vec)):
      s.write_i32(vec[i], start + 4*i)

  def read_i32_vec(s, start, size):
    """read i32-vector of size 'size' from memory address 'start'"""
    ret = np.empty(size, dtype=np.int32)
    for i in range(size):
      ret[i] = s.read_i32(start + 4*i)
    return ret

  def write_f32(s, f, addr):
    """write 32-bit float to memory (takes 4 byte-addresses)"""
    x = f.view(np.uint32)
    for i in range(4): s.mem[addr + i] = (x >> (8*i)) & 0xff

  def read_f32(s, addr):
    """"read 32-bit float from memory"""
    ret = s.i8(s.mem[addr + 3]) << 3*8
    for i in range(3): ret += s.mem[addr + i] << i*8
    return s.b2f(ret)

  def write_f32_vec(s, vec, start):
    """write i32-vector to memory address 'start'"""
    for i in range(np.size(vec)):
      s.write_f32(vec[i], start + 4*i)

  def read_f32_vec(s, start, size):
    """read i32-vector of size 'size' from memory address 'start'"""
    ret = np.empty(size, dtype=np.float32)
    for i in range(size):
      ret[i] = s.read_f32(start + 4*i)
    return ret

  def mem_dump(s, start, size):
    """for debug: dump memory from byteaddress 'start', dump out 'size' bytes"""
    for i in range(start, start + size, 4):
      print('%08x' % s.u(s.read_i32(i)))

  def dump_state(s):
    print('pc   : %4d' % s.pc)
    for i in range(0, 32, 4):
      print('x[%2d]: %4d, x[%2d]: %4d, x[%2d]: %4d, x[%2d]: %4d' %
            (i, s.x[i], i+1, s.x[i+1], i+2, s.x[i+2], i+3, s.x[i+3]))

  def print_perf(s, start='start', end='end'):
    """print performance numbers, 'start' and 'end' are for printing image size"""
    print('Ops counters: ' + str(s.ops))
    print('x[] regfile : ' + str(np.sum(s.x_usage[1:32])) + ' out of 31 x-registers are used')
    print('f[] regfile : ' + str(np.sum(s.f_usage))       + ' out of 32 f-registers are used')
    print('Image size  : ' + str(s.look_up_label(end) - s.look_up_label(start)) + ' Bytes')

  def print_rel_err(s, X, X_ref):
    """print maximum relative error of matrix X relative to X_ref"""
    max_err = np.max(np.abs((X - X_ref) / X_ref))
    print('maximum relative error = ' + str(max_err))
    # TODO: can we force numpy, torch, tf to not use any optimizations for matmul,
    # so that each inner-product is calculated in the same order as a dot-product?

#---------------------------------------------------------------------------------
# Only needed for Part II: dictionaries for decoder and assembler
#---------------------------------------------------------------------------------
dec_dict = {  # the decoder dictionary is copied from table 24.2 of the spec
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

# generate assembler dictionary by inverting the decoder dictionary
# so that key = 'instruction' and value = ['opcode-bits', 'format-type']
asm_dict = {dec_dict[k][1]: [k, dec_dict[k][0]] for k in dec_dict}

#---------------------------------------------------------------------------------
# assembler mnemonics
#---------------------------------------------------------------------------------
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
