import numpy as np
from bitstring import Bits, pack

# sources
# https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf
# RISC-V book
# https://inst.eecs.berkeley.edu/~cs61c/fa18/img/riscvcard.pdf

#-------------------------------------------------------------------------------
# create state of CPU: memory 'mem[]', register file 'x[]', program counter 'pc'
#-------------------------------------------------------------------------------
class State(object): pass
s = State()

mem_size = 1000  # size of memory in bytes
s.mem = np.zeros(mem_size, dtype=np.uint8) # memory 'mem[]' is unsigned int8
s.x   = np.zeros(32,       dtype=np.int32) # register file 'x[]' is signed int32
s.pc  = 0

#-------------------------------------------------------------------------------
# RV32 instructions
#-------------------------------------------------------------------------------
def i32(x): return np.int32(x)   # convert to 32-bit signed
def i8(x):  return np.int8(x)    # convert to 8-bit signed
def u(x):   return np.uint32(x)  # convert to 32-bit unsigned

# arithmetic
def ADD (s, rd, rs1, rs2): s.x[rd] = s.x[rs1] + s.x[rs2]; s.pc += 4
def ADDI(s, rd, rs1, imm): s.x[rd] = s.x[rs1] + imm;      s.pc += 4
def SUB (s, rd, rs1, rs2): s.x[rd] = s.x[rs1] - s.x[rs2]; s.pc += 4

# bitwise logical
def XOR (s, rd, rs1, rs2): s.x[rd] = s.x[rs1] ^ s.x[rs2]; s.pc += 4
def XORI(s, rd, rs1, imm): s.x[rd] = s.x[rs1] ^ imm;      s.pc += 4
def OR  (s, rd, rs1, rs2): s.x[rd] = s.x[rs1] | s.x[rs2]; s.pc += 4
def ORI (s, rd, rs1, imm): s.x[rd] = s.x[rs1] | imm;      s.pc += 4
def AND (s, rd, rs1, rs2): s.x[rd] = s.x[rs1] & s.x[rs2]; s.pc += 4
def ANDI(s, rd, rs1, imm): s.x[rd] = s.x[rs1] & imm;      s.pc += 4

# shift (note: 0x1f ensures that only the 5 LSBs are used as shift-amount)
# (For rv64, we will need to use the 6 LSBs so 0x3f)
def SLL (s, rd, rs1, rs2): s.x[rd] = s.x[rs1]    << (s.x[rs2] & 0x1f); s.pc += 4
def SRA (s, rd, rs1, rs2): s.x[rd] = s.x[rs1]    >> (s.x[rs2] & 0x1f); s.pc += 4
def SRL (s, rd, rs1, rs2): s.x[rd] = u(s.x[rs1]) >> (s.x[rs2] & 0x1f); s.pc += 4
def SLLI(s, rd, rs1, imm): s.x[rd] = s.x[rs1]    << imm;               s.pc += 4
def SRAI(s, rd, rs1, imm): s.x[rd] = s.x[rs1]    >> imm;               s.pc += 4
def SRLI(s, rd, rs1, imm): s.x[rd] = u(s.x[rs1]) >> imm;               s.pc += 4

# set to 1 if less than
def SLT  (s, rd, rs1, rs2): s.x[rd] = 1 if s.x[rs1]    < s.x[rs2]    else 0; s.pc += 4
def SLTI (s, rd, rs1, imm): s.x[rd] = 1 if s.x[rs1]    < imm         else 0; s.pc += 4
def SLTU (s, rd, rs1, rs2): s.x[rd] = 1 if u(s.x[rs1]) < u(s.x[rs2]) else 0; s.pc += 4
def SLTIU(s, rd, rs1, imm): s.x[rd] = 1 if u(s.x[rs1]) < u(imm)      else 0; s.pc += 4

# branch
def BEQ (s, rs1, rs2, imm): s.pc += imm if s.x[rs1] == s.x[rs2]       else 4
def BNE (s, rs1, rs2, imm): s.pc += imm if s.x[rs1] != s.x[rs2]       else 4
def BLT (s, rs1, rs2, imm): s.pc += imm if s.x[rs1] <  s.x[rs2]       else 4
def BGE (s, rs1, rs2, imm): s.pc += imm if s.x[rs1] >= s.x[rs2]       else 4
def BLTU(s, rs1, rs2, imm): s.pc += imm if u(s.x[rs1]) <  u(s.x[rs2]) else 4
def BGEU(s, rs1, rs2, imm): s.pc += imm if u(s.x[rs1]) >= u(s.x[rs2]) else 4

# jump
def JAL (s, rd, imm): s.x[rd] = s.pc + 4; s.pc += imm
def JALR(s, rd, rs1, imm): t = s.pc + 4; s.pc = (s.x[rs1] + imm) & ~1; s.x[rd] = t

# load immediate
def LUI  (s, rd, imm): s.x[rd] = imm << 12;          s.pc += 4
def AUIPC(s, rd, imm): s.x[rd] = s.pc + (imm << 12); s.pc += 4

# load, note the different argument order, example: 'lb rd, offset(rs1)'
def LB (s, rd, imm, rs1): s.x[rd] =  i8(s.mem[s.x[rs1] + imm]);  s.pc += 4
def LBU(s, rd, imm, rs1): s.x[rd] =     s.mem[s.x[rs1] + imm];   s.pc += 4
def LH (s, rd, imm, rs1): s.x[rd] = (i8(s.mem[s.x[rs1] + imm + 1]) << 8) + \
                                        s.mem[s.x[rs1] + imm];   s.pc += 4
def LHU(s, rd, imm, rs1): s.x[rd] =    (s.mem[s.x[rs1] + imm + 1]  << 8) + \
                                        s.mem[s.x[rs1] + imm];   s.pc += 4
def LW (s, rd, imm, rs1): s.x[rd] = (i8(s.mem[s.x[rs1] + imm + 3]) << 24) + \
                                       (s.mem[s.x[rs1] + imm + 2]  << 16) + \
                                       (s.mem[s.x[rs1] + imm + 1]  << 8)  + \
                                        s.mem[s.x[rs1] + imm];   s.pc += 4

# store, note the different argument order, example: 'sb rs2, offset(rs1)'
def SB(s, rs2, imm, rs1): s.mem[s.x[rs1] + imm] = s.x[rs2] & 0xff;  s.pc += 4
def SH(s, rs2, imm, rs1): s.mem[s.x[rs1] + imm] = s.x[rs2] & 0xff;  s.pc += 4; \
                          s.mem[s.x[rs1] + imm + 1] = (s.x[rs2] >> 8) & 0xff
def SW(s, rs2, imm, rs1): s.mem[s.x[rs1] + imm] = s.x[rs2] & 0xff;  s.pc += 4; \
                          s.mem[s.x[rs1] + imm + 1] = (s.x[rs2] >> 8)  & 0xff; \
                          s.mem[s.x[rs1] + imm + 2] = (s.x[rs2] >> 16) & 0xff; \
                          s.mem[s.x[rs1] + imm + 3] = (s.x[rs2] >> 24) & 0xff

# TODOs:
#  - implement missing instructions FENCE, ECALL, EBREAK
#  - add ISA extensions M and F
#  - add a check for writing to x[0], which is a known issue here

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
  ret = i8(s.mem[addr + 3]) << 3*8
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
    print('%08x' % u(read_i32(s, i)))

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
  else:
    print('ERROR: this instruction is not supported: ' + str(inst))

#-------------------------------------------------------------------------------
# encode instruction (this is the inverse function of the dec() function above)
#-------------------------------------------------------------------------------
def r_type(f7, f3, opcode, rd, rs1, rs2):
  return pack('bin:7, uint:5, uint:5, bin:3, uint:5, bin:7',
               f7, rs2, rs1, f3, rd, opcode)

def i_type(f3, opcode, rd, rs1, imm):
  return pack('int:12, uint:5, bin:3, uint:5, bin:7', imm, rs1, f3, rd, opcode)

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

def enc(s, inst, arg1, arg2, arg3=0):
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
  elif inst == 'sltiu': st = i_type('011', '0010011', arg1, arg2, arg3)
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
  else:
    print('ERROR: this instruction is not supported ' + inst)

  # write instruction into memory at address 's.pc'
  write_i32(s, st.int, s.pc)
  s.pc += 4

#-------------------------------------------------------------------------------
# execute code from address 'start', stop execution after n instructions
#-------------------------------------------------------------------------------
def exe(s, start, instructions):
  s.pc = start
  for i in range(0, instructions):
    inst = read_i32(s, s.pc)
    dec(Bits(uint=int(u(inst)), length=32))

#-------------------------------------------------------------------------------
# assembler mnemonics
#-------------------------------------------------------------------------------
x0  = 0;  x1  = 1;  x2  = 2;  x3  = 3;  x4  = 4;  x5  = 5;  x6  = 6;  x7  = 7
x8  = 8;  x9  = 9;  x10 = 10; x11 = 11; x12 = 12; x13 = 13; x14 = 14; x15 = 15
x16 = 16; x17 = 17; x18 = 18; x19 = 19; x20 = 20; x21 = 21; x22 = 22; x23 = 23
x24 = 24; x25 = 25; x26 = 26; x27 = 27; x28 = 28; x29 = 29; x30 = 30; x31 = 31

zero = 0; ra = x1;  sp = x2;  gp = x3;  tp = x4;  fp = x8
t0 = x5;  t1 = x6;  t2 = x7;  t3 = x28; t4 = x29; t5 = x30; t6 = x31
a0 = x10; a1 = x11; a2 = x12; a3 = x13; a4 = x14; a5 = x15; a6 = x16; a7 = x17
s0 = x8;  s1 = x9;  s2 = x18; s3 = x19; s4 = x20; s5 = x21; s6 = x22; s7 = x23
s8 = x24; s9 = x25; s10 = x26; s11 = x27

#-------------------------------------------------------------------------------
# pseudoinstructions
#-------------------------------------------------------------------------------
def LI(s, rd, imm):
  LUI (s, rd, imm >> 12)
  ADDI(s, rd, rd, imm & 0xfff)

# TODO: add more pseudoinstructions
