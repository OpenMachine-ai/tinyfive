# assembly code routines for some neural network layers

import numpy as np
from machine import machine
# alternatively, uncomment the line below to use the tinyfive package
#   from tinyfive.machine import machine

# abbreviations for shape dimensions:
#   C : input channels (and output channels if the same as input channels)
#   F : output channels (or filters), only used if F is not the same as C
#   R : input resolution (and output resolution if the same as input).
#   Q : output resolution, only used if Q is not the same as R

#-------------------------------------------------------------------------------
# Conv2D 1x1 with C in-channels, F out-channels, RxR image resolution
#-------------------------------------------------------------------------------
def conv_1x1_concept(m, C, F, R, S, w, a, ref):
  """Proof of concept for conv_1x1 assembly implementation for C in-channels,
  F out-channels, and R resolution, weight matrix 'w' and activation matrix 'a',
  and output reference matrix 'ref'. The proof-of-concept is as follows:
    - Split W into 4x4 submatrices and A into Sx4 submatrices, where S could be
      4 or 3 (this is to support cases where R*R is divisible by 4 or 3)
    - Then compute matmuls between these smaller submatrices to generate the
      big matmul between A and W."""
  a_split = np.empty((R*R//S, C//4, S, 4))
  w_split = np.empty((C//4, F//4, 4, 4))
  for i in range(C//4):
    for j in range(F//4):
      w_split[i, j] = w[i*4:i*4+4, j*4:j*4+4]
  for i in range(R*R//S):
    for j in range(C//4):
      a_split[i, j] = a[i*S:i*S+S, j*4:j*4+4]
  # compute the big matmul by smaller 4x4 matmuls
  y_con = np.zeros((R*R, F))
  for i in range(R*R//S):
    for j in range(F//4):
      for k in range(C//4):
        y_con[S*i:S*i+S, 4*j:4*j+4] += np.matmul(a_split[i, k], w_split[k, j])
  m.print_rel_err(y_con, ref)  # compare y_con against reference 'ref'

def conv_1x1(m, C, F, R, a_base, w_base, y_base, code_start, trans=False, S=4):
  """assembly code for conv2D 1x1, C in-channels, F out-channels, resolution R.
  C and F can be up to 128 (because immediates are limited to 12-bit). If trans
  is True, then the output is written into memory in transposed form. S should
  be set to 4 if R*R is divisible by 4. Otherwise, if R*R is divisible by 3,
  then set S to 3, other R values are currently not supported.
  Register map:
    x5:  constant for x15 .. x17 (only needed for trans)
    x6 : constant for incrementing x14 .. x17 (only needed for trans)
    x7 : constant for incrementing x12 (only needed for F >= 128)
    x8 : 1st base address for A
    x10: 2nd base address for A
    x12: base address for W
    x14: base address for results Y
    x15 .. x17: additional base regs for results Y (only needed for trans)
    f11: to store elements of A
    f12 .. f15: 4 registers to store an entire row of W
    f16 .. f31: the 16 outputs res[0, 0] ... res[4, 4]. Note, if S=3, then
                only 12 of these 16 registers are used."""

  # store assembly program starting at address 'code_start'
  m.pc = code_start
  m.lbl('start')

  # only needed if 4*4*F >= 2048 (i.e. for F = 128)
  m.asm('lui',  7,    m.hi20(4*4*F))
  m.asm('addi', 7, 7, m.lo12(4*4*F))

  if trans:  # only needed for trans and if R > 11 (i.e. if 4*4*R*R >= 2048)
    m.asm('lui',  6,    m.hi20(4*4*R*R))
    m.asm('addi', 6, 6, m.lo12(4*4*R*R))
    m.asm('lui',  5,    m.hi20(4*R*R))
    m.asm('addi', 5, 5, m.lo12(4*R*R))

  # matmul (R*R, C) x (C, F) -> (R*R, F)
  for i in range(R*R//S):  # S is 4 or 3
    m.asm('lui',  8,      m.hi20(a_base + 4*C*S*i))  # x8 =  ...
    m.asm('addi', 8, 8,   m.lo12(a_base + 4*C*S*i))
    m.asm('lui',  14,     m.hi20(y_base + (4*S*i if trans else 4*F*S*i)))
    m.asm('addi', 14, 14, m.lo12(y_base + (4*S*i if trans else 4*F*S*i)))
    m.asm('add', 15, 14, 5)
    m.asm('add', 16, 15, 5)
    m.asm('add', 17, 16, 5)

    # matmul (S, C) x (C, F) -> (S, F)
    for j in range(F//4):
      # set base address pointers
      m.asm('add', 10, 8, 0) # reset A pointer to x8
      m.asm('lui',  12,     m.hi20(w_base + 16*j))  # x12 = w_base + 16*j
      m.asm('addi', 12, 12, m.lo12(w_base + 16*j))

      # matmul (S, C) x (C, 4) -> (S, 4)
      for k in range(C//4):
        # compute one Sx4 matmul (by computing 4 outer products)
        for ii in range(4):
          # load row ii of W into registers f12 ... f15
          for col in range(4):
            m.asm('flw.s', 12+col, 4*(col+F*ii), 12)
          # compute outer-product in row-major order
          for row in range(S):
            m.asm('flw.s', 11, 4*(C*row+ii), 10)  # load f11 with A[row, ii]
            for col in range(4):
              if ii==0 and k==0:  # no accumulation for the very first products
                m.asm('fmul.s', 16+4*row+col, 11, 12+col)  # f[] = f11 * f12
              else:
                m.asm('fmadd.s', 16+4*row+col, 11, 12+col, 16+4*row+col) # f[] += f11 * f12

        # increment base addresses for A and W
        m.asm('addi', 10, 10, 4*4)  # increment x10 by 16
        m.asm('add', 12, 12, 7)     # increment x12

      # store results in memory
      for row in range(S):
        for col in range(4):
          if trans:
            m.asm('fsw.s', 16+4*row+col, 4*row, 14+col)  # x14 for col=0, x15 for col=1, ..
          else:
            m.asm('fsw.s', 16+4*row+col, 4*(row*F+col), 14)
      if trans:
        for col in range(4):
          m.asm('add', 14+col, 14+col, 6)  # increment Y pointer by x7
      else:
        m.asm('addi', 14, 14, 4*4)  # increment Y pointer by 16
  m.lbl('end')

  # execute program from 'start' to 'end'
  m.exe(start='start', end='end')

  # TODOs:
  #  - replace the outer for-loop (i, j, k) by assembly code with branches to
  #    reduce the image size
  #  - clean up the indexing and base address pointers
  #  - use mnemonics x9 and f9 etc. instead of '9'
  #  - rewrite above using only upper-case instructions to speed up runtime
  # note on the 12-bit immediates: -2048 .. +2047. For weights, we could store
  # the W-matrix in transposed form, which gives us a bit more indexing room

#-------------------------------------------------------------------------------
# Same as conv_1x1 but with support for C,F > 128 or for large R when transposed
#-------------------------------------------------------------------------------
def conv_1x1_big(m, C, F, R, a_base, w_base, y_base, code_start, trans=False, S=4):
  """same as conv_1x1, but for C,F > 128 (up to 256 for now) if trans==False
  or for larger R if trans==True.
  Register map:
    x5: constant for x15 .. x17 (mainly needed for trans)
    x6: constant for incrementing x14 .. x17 (only needed for trans)
    x7: constant for incrementing x12 and x13 (only needed for F >= 128)
    x8,  x9 : 1st base address registers for A
    x10, x11: 2nd base address registers for A
    x12, x13: 2 base address registers for W (due to 12-bit limit)
    x14 .. x17: 4 base address registers for results Y
    f11: to store elements of A
    f12 .. f15: 4 registers to store an entire row of W
    f16 .. f31: the 16 outputs res[0, 0] ... res[4, 4]. Note, if S=3, then
                only 12 of these 16 registers are used."""

  # store assembly program starting at address 'code_start'
  m.pc = code_start
  m.lbl('start')

  m.asm('lui',  7,    m.hi20(4*4*F)) # only needed if 4*4*F >= 2048
  m.asm('addi', 7, 7, m.lo12(4*4*F))
  m.asm('lui',  5,    m.hi20(4*R*R if trans else 4*F))
  m.asm('addi', 5, 5, m.lo12(4*R*R if trans else 4*F))

  if trans:  # only needed for trans and if R > 11 (i.e. if 4*4*R*R >= 2048)
    m.asm('lui',  6,    m.hi20(4*4*R*R))
    m.asm('addi', 6, 6, m.lo12(4*4*R*R))

  # matmul (R*R, C) x (C, F) -> (R*R, F)
  for i in range(R*R//S):
    m.asm('lui',  8,      m.hi20(a_base + 4*C*S*i))  # x8 =  ...
    m.asm('addi', 8, 8,   m.lo12(a_base + 4*C*S*i))
    m.asm('lui',  9,      m.hi20(a_base + 4*C*S*i + 8*C))  # x9 = x8 + 4*C*2
    m.asm('addi', 9, 9,   m.lo12(a_base + 4*C*S*i + 8*C))
    m.asm('lui',  14,     m.hi20(y_base + (4*S*i if trans else 4*F*S*i)))  # x14 = ...
    m.asm('addi', 14, 14, m.lo12(y_base + (4*S*i if trans else 4*F*S*i)))
    m.asm('add', 15, 14, 5)
    m.asm('add', 16, 15, 5)
    m.asm('add', 17, 16, 5)

    # matmul (S, C) x (C, F) -> (S, F)
    for j in range(F//4):
      # set base address pointers
      m.asm('add', 10, 8, 0) # x10 = x8
      m.asm('add', 11, 9, 0) # x11 = x9
      m.asm('lui',  12,     m.hi20(w_base + 16*j))  # x12 = w_base + 16*j
      m.asm('addi', 12, 12, m.lo12(w_base + 16*j))
      m.asm('lui',  13,     m.hi20(w_base + 16*j + 8*F))  # x13 = x12 + 4*F*2
      m.asm('addi', 13, 13, m.lo12(w_base + 16*j + 8*F))

      # matmul (S, C) x (C, 4) -> (S, 4)
      for k in range(C//4):
        # compute one Sx4 matmul (by computing 4 outer products)
        for ii in range(4):
          # load row ii of W into registers f12 ... f15
          for col in range(4):
            m.asm('flw.s', 12+col, 4*(col+F*(ii%2)), 12+ii//2) # use x12 for ii=0,1, x13 for ii=2,3
          # compute outer-product in row-major order
          for row in range(S):
            m.asm('flw.s', 11, 4*(C*(row%2) + ii), 10+row//2)  # load f11 with A[row, ii]
            for col in range(4):
              if ii==0 and k==0:  # no accumulation for the very first products
                m.asm('fmul.s', 16+4*row+col, 11, 12+col)  # f[] = f11 * f12
              else:
                m.asm('fmadd.s', 16+4*row+col, 11, 12+col, 16+4*row+col) # f[] += f11 * f12

        # increment base addresses for A and W
        m.asm('addi', 10, 10, 4*4)  # increment by 16
        m.asm('addi', 11, 11, 4*4)  # increment by 16
        m.asm('add', 12, 12, 7)
        m.asm('add', 13, 13, 7)

      # store results in memory
      for row in range(S):
        for col in range(4):
          if trans:
            m.asm('fsw.s', 16+4*row+col, 4*row, 14+col)  # use x14 for col=0, x15 for col=1, ..
          else:
            m.asm('fsw.s', 16+4*row+col, 4*col, 14+row)  # use x14 for row=0, x15 for row=1, ..
      # increment Y pointers
      if trans:
        for col in range(4):
          m.asm('add', 14+col, 14+col, 6)  # increment Y pointer by x6
      else:
        for row in range(S):
          m.asm('addi', 14+row, 14+row, 4*4)  # increment Y pointer by 16
  m.lbl('end')

  # execute program from 'start' to 'end'
  m.exe(start='start', end='end')

# TODO: merge conv1x1_big() into conv1x1()

#-------------------------------------------------------------------------------
# Depthwise Conv2D 3x3 with C channels, RxR image, stride=1
#-------------------------------------------------------------------------------
def dw_conv_3x3_stride1(m, C, R, a_base, w_base, y_base, out_chan_first=True):
  """assembly code with upper-case instruction for depthwise conv2D 3x3 with
  C channels, R resolution, stride = 1. If out_chan_first==True, then the
  output shape is (channel, row, col); otherwise shape is (row, col, channel)
  Register map:
    x10 : base address for A[chan]
    x11 : base address for W[chan]
    x12 : base address for Y[chan]
    f0 .. f8: the 9 weights of a channel, stored in row-major order
    f9  : holds the latest activation loaded from memory
    f10 : accumulation register 'out0' for current output
    f11 : accumulation register 'out1' for next output
    f12 : accumulation register 'out2' for next-next output"""

  # init base addresses
  m.LI(10, a_base)
  m.LI(11, w_base)
  if out_chan_first:
    m.LI(12, y_base)

  for chan in range(C):
    if out_chan_first==False:
        m.LI(12, y_base)

    # load 3x3 weights for channel 'chan'
    for i in range(3):
      for j in range(3):
        m.FLW_S(3*i+j, (3*i + j)*4, 11)  # f[i, j] = W[chan, i, j]

    # compute all outputs (RxR) for channel 'chan'
    for row in range(R):
      for col in range(R):
        # load 3 activations, perform 9 muls, and store 1 output
        dot_start = 0 if row > 0 else 1    # first row is special
        dot_end   = 3 if row < R-1 else 2  # last row is special
        for dot in range(dot_start, dot_end):
          # load one activation from memory
          m.FLW_S(9, (R*(row-1+dot) + col)*4, 10)  # A[chan, row-1+dot, col]

          # compute 3 muls with weights W[dot, 0:3]
          if dot == dot_start:
            if col > 0:
              m.FMADD_S(10, 9, 3*dot+2, 11)  # f10 = f9 * W[dot, 2] + f11
              m.FMADD_S(11, 9, 3*dot+1, 12)  # f11 = f9 * W[dot, 1] + f12
            else:
              m.FMUL_S(11, 9, 3*dot+1)            # f11 = f9 * W[dot, 1]
            if col < R-1: m.FMUL_S(12, 9, 3*dot)  # f12 = f9 * W[dot, 0]
          else:
            m.FMADD_S(11, 9, 3*dot+1, 11)                # f11 += f9 * W[dot, 1]
            if col > 0:   m.FMADD_S(10, 9, 3*dot+2, 10)  # f10 += f9 * W[dot, 2]
            if col < R-1: m.FMADD_S(12, 9, 3*dot, 12)    # f12 += f9 * W[dot, 0]

        # store result
        if out_chan_first:
          if col > 0:    m.FSW_S(10, (R*row + col-1)*4, 12)  # Y[chan, row, col-1]
          if col == R-1: m.FSW_S(11, (R*row + col  )*4, 12)  # Y[chan, row, col]
        else:
          if col > 0:    m.FSW_S(10, (C*(col-1) + chan)*4, 12)  # Y[row, col-1, chan]
          if col == R-1: m.FSW_S(11, (C*col + chan)*4, 12)      # Y[row, col, chan]
      if out_chan_first==False:
        m.ADDI(12, 12, C*R*4)  # for Y(chan)

    # increment base addresses
    m.ADDI(11, 11, 9*4)    # for W(chan)
    m.ADDI(10, 10, R*R*4)  # for A(chan)
    if out_chan_first:
      m.ADDI(12, 12, R*R*4)  # for Y(chan)

    # TODOs:
    #  - reduce number of loads by computing several outputs in parallel (each output
    #    requires three registers for stride=1, so here we could compute 6 outputs in
    #    parallel; and when image-size is 6x6, process an entire column in parallel)

#-------------------------------------------------------------------------------
# Depthwise Conv2D 3x3 with C channels, RxR input image, stride=2
#-------------------------------------------------------------------------------
def dw_conv_3x3_stride2(m, C, R, a_base, w_base, y_base, out_chan_first=True):
  """assembly code with upper-case instruction for depthwise conv2D 3x3 with
  C channels, R input resolution, stride = 2 (so output resolution is Q = R/2).
  If out_chan_first==True, then the output shape is (channel, row, col);
  otherwise shape is (row, col, channel)
  Register map:
    x10 : base address for A[chan]
    x11 : base address for W[chan]
    x12 : base address for Y[chan]
    f0 .. f8: the 9 weights of a channel, stored in row-major order
    f9  : holds the latest activation loaded from memory
    f10 : accumulation register 'out0' for current output
    f11 : accumulation register 'out1' for next output"""

  # init base addresses
  Q = R//2  # output resolution
  m.LI(10, a_base)
  m.LI(11, w_base)
  if out_chan_first:
    m.LI(12, y_base)

  for chan in range(C):
    if out_chan_first==False:
        m.LI(12, y_base)

    # load 3x3 weights for channel 'chan'
    for i in range(3):
      for j in range(3):
        m.FLW_S(3*i+j, (3*i + j)*4, 11)  # f[i, j] = W[chan, i, j]

    # compute all outputs (QxQ) for channel 'chan'
    for row in range(1, R, 2):
      for col in range(R):
        # load 3 activations, perform 9 muls, and store 1 output
        for dot in range(0, 3 if row < R-1 else 2):  # last row is special
          # load one activation from memory
          m.FLW_S(9, (R*(row-1+dot) + col)*4, 10)  # A[chan, row-1+dot, col]

          # compute 3 muls with weights W[dot, 0:3]
          if (col % 2) == 0:  # even columns
            if col > 0:
              m.FMADD_S(10, 9, 3*dot+2, 10)  # f10 += f9 * W[dot, 2]
            if dot == 0:
              m.FMUL_S(11, 9, 3*dot)         # f11  = f9 * W[dot, 0]
            else:
              m.FMADD_S(11, 9, 3*dot, 11)  # f11 += f9 * W[dot, 0]
          else:  # odd columns
            if dot == 0:
              m.FMADD_S(10, 9, 3*dot+1, 11)    # f10  = f9 * W[dot, 1] + f11
            else:
               m.FMADD_S(10, 9, 3*dot+1, 10)  # f10 += f9 * W[dot, 1]
        # store result
        if out_chan_first:
          if col > 0 and (col % 2) == 0:
            m.FSW_S(10, (Q*(row-1)//2 + (col-2)//2)*4, 12)  # Y[chan, (row-1)/2, (col-2)/2]
          if (col == R-1):
            m.FSW_S(10, (Q*(row-1)//2 + (col-1)//2)*4, 12)  # Y[chan, (row-1)/2, (col-1)/2]
        else:
          if col > 0 and (col % 2) == 0:
            m.FSW_S(10, (C*(col-2)//2 + chan)*4, 12)  # Y[(row-1)/2, (col-2)/2, chan]
          if (col == R-1):
            m.FSW_S(10, (C*(col-1)//2 + chan)*4, 12)  # Y[(row-1)/2, (col-1)/2, chan]
      if out_chan_first==False:
        m.ADDI(12, 12, C*Q*4)  # for Y(chan)

    # increment base addresses
    m.ADDI(11, 11, 9*4)    # for W(chan)
    m.ADDI(10, 10, R*R*4)  # for A(chan)
    if out_chan_first:
      m.ADDI(12, 12, Q*Q*4)  # for Y(chan)

#-------------------------------------------------------------------------------
# Conv2D 3x3 with 3 in-channels, F out-channels, RxR image, stride=1
#-------------------------------------------------------------------------------
def conv_3x3x3_stride1(m, F, R, a_base, w_base, y_base):
  """assembly code with upper-case instruction for conv2D 3x3 with 3 in-channels,
  F out-channels, stride = 1, R input and output resolution.
  Register map:
    x10 : base address for A[chan]
    x11 : base address for W[chan]
    x12 : base address for Y[chan]
    f0 .. f26: the 27 weights (3, 3, 3) for one output channel
    f27 : holds the latest activation loaded from memory
    f28 : accumulation register 'out0' for current output
    f29 : accumulation register 'out1' for next output
    f30 : accumulation register 'out2' for next-next output"""

  # init base addresses
  m.LI(10, a_base)
  m.LI(11, w_base)
  m.LI(12, y_base)

  for chan in range(F): # 'chan' refers to 'output-channel'
    # load 3x3x3 weights for output-channel 'chan'
    for i in range(3):
      for j in range(3):
        for k in range(3):  # 'k' is input-channel
          m.FLW_S(9*i+3*j+k, (9*i+3*j+k)*4, 11)  # f[i, j, k] = W[i, j, k, chan]

    # compute all outputs (RxR) for channel 'chan'
    for row in range(R):
      for col in range(R):
        # load 3*3 activations, perform 27 muls, and store 1 output
        dot_start = 0 if row > 0   else 1  # first row is special
        dot_end   = 3 if row < R-1 else 2  # last row is special
        for dot in range(dot_start, dot_end):
          for k in range(3):  # 'k' is input-channel
            # load one activation from memory
            m.FLW_S(27, (3*R*(row-1+dot) + 3*col + k)*4, 10)  # A[row-1+dot, col, k]

            # compute 3 muls with weights W[dot, 0:3]
            if dot == dot_start and k == 0:
              if col > 0:
                m.FMADD_S(28, 27, 9*dot+3*2, 29)  # f28 = f27 * W[dot, 2, 0] + f29
                m.FMADD_S(29, 27, 9*dot+3*1, 30)  # f29 = f27 * W[dot, 1, 0] + f30
              else:
                m.FMUL_S(29, 27, 9*dot+3*1)       # f29 = f27 * W[dot, 1, 0]
              if col < R-1:
                m.FMUL_S(30, 27, 9*dot)           # f30 = f27 * W[dot, 0, 0]
            else:
              m.FMADD_S(29, 27, 9*dot+3*1+k, 29)                # f29 += f27 * W[dot, 1, k]
              if col > 0:   m.FMADD_S(28, 27, 9*dot+3*2+k, 28)  # f28 += f27 * W[dot, 2, k]
              if col < R-1: m.FMADD_S(30, 27, 9*dot+k, 30)      # f30 += f27 * W[dot, 0, k]
        # store result
        if col > 0:
          m.FSW_S(28, (R*row + col-1)*4, 12)  # Y[chan, row, col-1]
        if col == R-1:
          m.FSW_S(29, (R*row + col)*4, 12)    # Y[chan, row, col]
    # increment base addresses
    m.ADDI(11, 11, 27*4)   # for W
    m.ADDI(12, 12, R*R*4)  # for Y

#-------------------------------------------------------------------------------
# Conv2D 3x3 with 3 in-channels, F out-channels, RxR image, stride=2
#-------------------------------------------------------------------------------
def conv_3x3x3_stride2(m, F, R, a_base, w_base, y_base):
  """assembly code with upper-case instruction for conv2D 3x3 with 3 in-channels,
  F out-channels, stride = 2, R input resolution, and R/2 output resolution.
  Note on stride=2: keras does the striding as follows: the first valid output
  equals the [1, 1] output of the non-strided version, etc.
  Register map:
    x10 : base address for A[chan]
    x11 : base address for W[chan]
    x12 : base address for Y[chan]
    f0 .. f26: the 27 weights (3, 3, 3) for one output channel
    f27 : holds the latest activation loaded from memory
    f28 : accumulation register 'out0' for current output
    f29 : accumulation register 'out1' for next output"""

  # init base addresses
  Q = R//2  # output resolution
  m.LI(10, a_base)
  m.LI(11, w_base)
  m.LI(12, y_base)

  for chan in range(F): # 'chan' refers to 'output-channel'
    # load 3x3x3 weights for output-channel 'chan'
    for i in range(3):
      for j in range(3):
        for k in range(3):  # 'k' is input-channel
          m.FLW_S(9*i+3*j+k, (9*i+3*j+k)*4, 11)  # f[i, j, k] = W[i, j, k, chan]
    # compute all outputs (QxQ) for channel 'chan'
    for row in range(1, R, 2):
      for col in range(R):
        # load 3*3 activations, perform 27 muls, and store 1 output
        for dot in range(0, 3 if row < R-1 else 2):  # last row is special
          for k in range(3):  # 'k' is input-channel
            init = (dot == 0) and (k == 0)  # shortcut for below if/else

            # load one activation from memory
            m.FLW_S(27, (3*R*(row-1+dot) + 3*col + k)*4, 10)  # A[row-1+dot, col, k]

            # compute 3 muls with weights W[dot, 0:3]
            if (col % 2) == 0:  # even columns
              if col > 0:
                m.FMADD_S(28, 27, 9*dot+3*2+k, 28)  # f28 += f27 * W[dot, 2, k]
              if init:
                m.FMUL_S(29, 27, 9*dot+3*0)         # f29  = f27 * W[dot, 0, 0]
              else:
                m.FMADD_S(29, 27, 9*dot+3*0+k, 29)  # f29 += f27 * W[dot, 0, k]
            else:  # odd columns
              if init:
                m.FMADD_S(28, 27, 9*dot+3*1, 29)    # f28  = f27 * W[dot, 1, 0] + f29
              else:
                m.FMADD_S(28, 27, 9*dot+3*1+k, 28)  # f28 += f27 * W[dot, 1, k]
        # store result
        if col > 0 and (col % 2) == 0:
          m.FSW_S(28, (Q*(row-1)//2 + (col-2)//2)*4, 12)  # Y[chan, (row-1)/2, (col-2)/2]
        if (col == R-1):
          m.FSW_S(28, (Q*(row-1)//2 + (col-1)//2)*4, 12)  # Y[chan, (row-1)/2, (col-1)/2]
    # increment base addresses
    m.ADDI(11, 11, 27*4)   # for W
    m.ADDI(12, 12, Q*Q*4)  # for Y
    # TODOs:
    #  - reduce number of loads by computing several outputs in parallel (each output
    #    requires two registers, so we could compute two outputs in parallel)
    #  - above code uses values for immediates that might excee 12-bit, e.g. the immediate
    #    for loading the activations (and perhaps also for storing the results). Therefore,
    #    increment the base addresses more frequently
