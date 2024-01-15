
# Copyright (c) 2019-2020 Idein Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np

def sfu_sin(x):
    result = np.sin(x * np.pi)
    result[x < -0.5] = -1
    result[x >  0.5] = 1
    return result

ops = {
    # sfu regs/ops
    'recip' : lambda x: 1 / x,
    'rsqrt' : lambda x: 1 / np.sqrt(x),
    'exp' : lambda x: 2 ** x,
    'log' : np.log2,
    'sin' : sfu_sin,
    'rsqrt2' : lambda x: 1 / np.sqrt(x),
}



# SFU IO registers
@qpu
def qpu_sfu_regs(asm, sfu_regs):

    eidx(rf0, sig = ldunif)
    mov(rf10, rf5, sig = ldunif) # in
    mov(rf3, 4)
    shl(rf3, rf3, 4).mov(rf11, rf5)

    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw).add(rf10, rf10, rf3)
    nop()
    nop()
    nop(sig = ldtmu(rf1))

    g = globals()
    for reg in sfu_regs:
        mov(g[reg], rf1)
        nop() # required ? enough ?
        mov(tmud, rf5)
        mov(tmua, rf11)
        tmuwt().add(rf11, rf11, rf3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def boilerplate_sfu_regs(sfu_regs, domain_limitter):

    with Driver() as drv:

        code = drv.program(lambda asm: qpu_sfu_regs(asm, sfu_regs))
        X = drv.alloc((16, ), dtype = 'float32')
        Y = drv.alloc((len(sfu_regs), 16), dtype = 'float32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = domain_limitter(np.random.randn(*X.shape).astype('float32'))
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, reg in enumerate(sfu_regs):
            msg = 'mov({}, None)'.format(reg)
            assert np.allclose(Y[ix], ops[reg](X), rtol=1e-4), msg

def test_sfu_regs():
    boilerplate_sfu_regs(['recip','exp','sin'], lambda x: x)
    boilerplate_sfu_regs(['rsqrt','log','rsqrt2'], lambda x: x ** 2 + 1e-6)


# SFU ops
@qpu
def qpu_sfu_ops(asm, sfu_ops):

    eidx(rf10, sig = ldunif)
    mov(rf0, rf15, sig = ldunif) # in
    mov(rf13, 4)
    shl(rf13, rf13, 4).mov(rf1, rf15)

    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, rf13)
    nop()
    nop()
    nop(sig = ldtmu(rf11))

    g = globals()
    for op in sfu_ops:
        g[op](rf2, rf11) # ATTENTION: SFU ops requires rfN ?
        mov(tmud, rf2)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, rf13)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def boilerplate_sfu_ops(sfu_ops, domain_limitter):

    with Driver() as drv:

        code = drv.program(lambda asm: qpu_sfu_ops(asm, sfu_ops))
        X = drv.alloc((16, ), dtype = 'float32')
        Y = drv.alloc((len(sfu_ops), 16), dtype = 'float32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = domain_limitter(np.random.randn(*X.shape).astype('float32'))
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, op in enumerate(sfu_ops):
            msg = '{}(None, None)'.format(op)
            assert np.allclose(Y[ix], ops[op](X), rtol=1e-4), msg

def test_sfu_ops():
    boilerplate_sfu_ops(['recip','exp','sin'], lambda x: x)
    boilerplate_sfu_ops(['rsqrt','log','rsqrt2'], lambda x: x ** 2 + 1e-6)