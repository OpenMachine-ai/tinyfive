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
def conv_1x1(m, C, F, R, a_base, w_base, y_base, code_start):
  """assembly code with for conv2D 1x1 with F in-channels, F out-channels,
  and resolution R.
  Register map:
    x[9]  : 1st base address for A
    x[10] : 2nd base address for A
    x[11] : base address for W
    x[12] : base address for results Y
    f[11] : to store elements of A
    f[12] .. f[15]: 4 registers to store an entire row of W
    f[16] .. f[31]: the 16 outputs res[0, 0] ... res[4, 4]"""

  # store assembly program starting at address 'code_start'
  m.pc = code_start
  m.lbl('start')

  # matmul (R*R, C) x (C, F) -> (R*R, F)
  for i in range(R*R//4):
    m.asm('lui',  9,      m.hi20(a_base + 4*C*4*i))  # m.x[9] =  ...
    m.asm('addi', 9, 9,   m.lo12(a_base + 4*C*4*i))
    m.asm('lui',  12,     m.hi20(y_base + 4*F*4*i))  # m.x[12] = ...
    m.asm('addi', 12, 12, m.lo12(y_base + 4*F*4*i))

    # matmul (4, C) x (C, F) -> (4, F)
    for j in range(F//4):
      # set base address pointers
      m.asm('add', 10, 9, 0) # reset A pointer to x[9]
      m.asm('lui',  11,     m.hi20(w_base + 16*j))  # m.x[11] = w_base + 16*j
      m.asm('addi', 11, 11, m.lo12(w_base + 16*j))

      # matmul (4, C) x (C, 4) -> (4, 4)
      for k in range(C//4):
        # compute one 4x4 matmul (by computing 4 outer products)
        for ii in range(4):
          # load row ii of W into registers f[12] ... f[15]
          for col in range(4):
            m.asm('flw.s', 12+col, 4*(col+F*ii), 11)
          # compute outer-product in row-major order
          for row in range(4):
            m.asm('flw.s', 11, 4*(C*row+ii), 10)  # load f[11] with A[row, ii]
            for col in range(4):
              if ii==0 and k==0:  # no accumulation for the very first products
                m.asm('fmul.s', 16+4*row+col, 11, 12+col)  # f[] = f[11] * f[12]
              else:
                m.asm('fmadd.s', 16+4*row+col, 11, 12+col, 16+4*row+col) # f[] += f[11] * f[12]

        # increment base addresses for A and W
        m.asm('addi', 10, 10, 4*4)  # increment by 16
        m.asm('addi', 11, 11, 4*4*F//2)  # increment by 4*4*F by two times 4*4*F/2
        m.asm('addi', 11, 11, 4*4*F//2)
        # note on the last two lines: 12-bit immediates: -2048 .. +2047. So we increment
        # by 1024 two times to achieve 2048 increment. Alternatively, we could decrement
        # the index (because -2048 is possible in one instruction). Or store the W-matrix
        # in transposed form

      # store results in memory
      for row in range(4):
        for col in range(4):
          m.asm('fsw.s', 16+4*row+col, 4*(row*F+col), 12)
      m.asm('addi', 12, 12, 4*4)  # increment Y pointer by 16
  m.lbl('end')

  # execute program from 'start' to 'end'
  m.exe(start='start', end='end')

  # TODOs:
  #  - replace the outer for-loop (i, j, k) by assembly code with branches to
  #    reduce the image size
  #  - clean up the indexing and base address pointers
  #  - use mnemonics x9 and f9 etc. instead of '9'
  #  - rewrite above using only upper-case instructions to speed up runtime

#-------------------------------------------------------------------------------
# Depthwise Conv2D 3x3 with C channels, RxR image, stride=1
#-------------------------------------------------------------------------------
def dw_conv_3x3_stride1(m, C, R, a_base, w_base, y_base, out_chan_first=True):
  """assembly code with upper-case instruction for depthwise conv2D 3x3 with
  C channels, R resolution, stride = 1. If out_chan_first==True, then the
  output shape is (channel, row, col); otherwise shape is (row, col, channel)
  Register map:
    x[10] : base address for A[chan]
    x[11] : base address for W[chan]
    x[12] : base address for Y[chan]
    f[0] .. f[8]: the 9 weights of a channel, stored in row-major order
    f[9]  : holds the latest activation loaded from memory
    f[10] : accumulation register 'out0' for current output
    f[11] : accumulation register 'out1' for next output
    f[12] : accumulation register 'out2' for next-next output"""

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
    x[10] : base address for A[chan]
    x[11] : base address for W[chan]
    x[12] : base address for Y[chan]
    f[0] .. f[8]: the 9 weights of a channel, stored in row-major order
    f[9]  : holds the latest activation loaded from memory
    f[10] : accumulation register 'out0' for current output
    f[11] : accumulation register 'out1' for next output"""

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
    x[10] : base address for A[chan]
    x[11] : base address for W[chan]
    x[12] : base address for Y[chan]
    f[0] .. f[26]: the 27 weights (3, 3, 3) for one output channel
    f[27] : holds the latest activation loaded from memory
    f[28] : accumulation register 'out0' for current output
    f[29] : accumulation register 'out1' for next output
    f[30] : accumulation register 'out2' for next-next output"""

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
    x[10] : base address for A[chan]
    x[11] : base address for W[chan]
    x[12] : base address for Y[chan]
    f[0] .. f[26]: the 27 weights (3, 3, 3) for one output channel
    f[27] : holds the latest activation loaded from memory
    f[28] : accumulation register 'out0' for current output
    f[29] : accumulation register 'out1' for next output"""

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
