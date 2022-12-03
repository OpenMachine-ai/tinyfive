import math
import numpy as np
import softfloat as sf

# IEEE-754 and RISC-V define the following 5 rounding modes ('rm' in RISC-V):
# (see https://en.wikipedia.org/wiki/IEEE_754#Rounding_rules)
#  (A) rounding to nearest:
#     A.1: round to nearest, ties to even             (RISC-V rm = 0)
#     A.2: round to nearest, ties away from zero      (RISC-V rm = 4)
#  (B) directed rounding:
#     B.1: truncation (round toward 0)                (RISC-V rm = 1)
#     B.2: floor (round toward negative infinity)     (RISC-V rm = 2)
#     B.3: ceiling (round toward positive infinity)   (RISC-V rm = 3)
# Option A.1 is the default one and most commonly used one (no need to
# implement any of the other modes, not sure why RISC-V requires it)

# We can implement the 5 rounding modes in Python 3 as follows:
#   A.1: built-in round()
#   A.2: use below function "round_away()"
#   B.1: built-in int()
#   B.2: math.floor()
#   B.3: math.ceil()
# Note on Python3 vs. Python 2: The round() function in Python 3
# implements A.1, while Python 2 implemnents A.2, see below:
#   - Python3: https://docs.python.org/3/library/functions.html#round
#   - Python2: https://docs.python.org/2/library/functions.html#round

# SoftFloat:
#   - it's used by Spike and other RISC-V ISSs as the golden reference
#   - it uses the same rounding mode definitions as RISC-V (rm = 0,1,2,3,4)
#   - however, the rounding mode is set as a global variable, which is set to
#     0 by default and can't be changed when using the Python package. The only
#     exception are the convert-to-integer functions such as f32_to_i32, which
#     have a rounding mode argument.
#   - http://www.jhauser.us/arithmetic/SoftFloat.html
#   - Python package https://gitlab.com/cerlane/SoftFloat-Python

#-------------------------------------------------------------------------------
# rounding to int
#-------------------------------------------------------------------------------
def round_away(x):
  if x >= 0.0: return math.floor(x + 0.5)
  else:        return math.ceil(x - 0.5)

def sf_round(x, rm):
  return sf.f32_to_i32(sf.convertDoubleToF32(x), rm, False)

# print table between -4 ... +4 with increment of 'incr'
def print_table(incr):
  print('\n   mode:  A.1  A.2  B.1  B.2  B.3')
  a = np.arange(-4, 4+incr, incr)
  for i in range(a.size):
    x = a[i]
    A1 = round(x)
    A2 = round_away(x)
    B1 = int(x)
    B2 = math.floor(x)
    B3 = math.ceil(x)
    print('%7.4f: %3d  %3d  %3d  %3d  %3d' % (x, A1, A2, B1, B2, B3))
    # check against softfloat
    assert A1 == sf_round(x, 0)
    assert A2 == sf_round(x, 4)
    assert B1 == sf_round(x, 1)
    assert B2 == sf_round(x, 2)
    assert B3 == sf_round(x, 3)
    # check against numpy
    assert A1 == np.round(x) == np.rint(x)
    assert B1 == np.trunc(x) == np.fix(x)
    assert B2 == np.floor(x)
    assert B3 == np.ceil(x)

# table with only one mantissa bit (so increment = 0.5)
print_table(0.5)

# table with 2 mantissa bits (so increment = 2^-2)
print_table(2**(-2))

# table with 4 mantissa bits (so increment = 2^-4)
#print_table(2**(-4))

#-------------------------------------------------------------------------------
# rounding to FP32
#-------------------------------------------------------------------------------

# below are four cases for rounding to fp32 with mode A.1:
#   0.5  -> round to 0 (because 0 is even)
#   0.75 -> round to 1
#   1.0  -> round to 1
#   1.5  -> round to 2

# here, 2^(-23) corresponds to the LSB after rounding, all bits beyond it
# will be rounded off. So 2^(-23) corresponds to 'b1.0' = 1.0; and 2^(-24)
# corresponds to 'b0.1' = 0.5; and 2^(-25) corresponds to 'b0.01' = 0.25.

# the last 2 LSBs and 2 bits beyond: "00.10" (case 0.5 -> round to 0)
print('\n%.16f' % np.float32(1 + 2**(-24)))

# the last 2 LSBs and 2 bits beyond: "00.11" (case 0.75 -> round to 1)
print('\n%.16f' % np.float32(1 + 2**(-24) + 2**(-25)))

# the last 2 LSBs and 2 bits beyond: "01.00" (case 1.0 -> round to 1)
print('\n%.16f' % np.float32(1 + 2**(-23)))

# the last 2 LSBs and 2 bits beyond: "01.10" (case 1.5 -> round to 2)
print('\n%.16f' % np.float32(1 + 2**(-23) + 2**(-24)))

#-------------------------------------------------------------------------------
# test FP add
#-------------------------------------------------------------------------------

# softfloat example
A = sf.convertDoubleToF32(1.5)
B = sf.convertDoubleToF32(2.75)
print(sf.f32_add(A, B))

def test_fp_add(A, B):
  print('\nFull precision: %.16f' % (A + B))
  print('np.float32    : %.16f' % np.float32(np.float32(A) + np.float32(B)))
  print('softfloat_f32 : ' + str(sf.f32_add(sf.convertDoubleToF32(A),
                                            sf.convertDoubleToF32(B))))

test_fp_add(1, 2**(-23))
test_fp_add(1, 2**(-24))

test_fp_add(1, 2**(-24) * 1.5)
test_fp_add(1, 2**(-24) * (1 + 2**(-23)))
test_fp_add(1, 2**(-24) * (1 + 2**(-24)))
