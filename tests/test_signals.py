
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


# ldtmu
@qpu
def qpu_signal_ldtmu(asm):

    eidx(rf0, sig = ldunifrf(rf5))
    mov(rf10, rf5, sig = ldunifrf(rf5))
    mov(rf3, 4)
    shl(rf3, rf3, 4).mov(rf11, rf5)

    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw).add(rf10, rf10, rf3)        # start load X
    mov(rf0, 1.0)                                         # r0 <- 1.0
    mov(rf1, 2.0)                                         # r1 <- 2.0
    fadd(rf0, rf0, rf0).fmul(rf1, rf1, rf1, sig = ldtmu(rf31)) # r0 <- 2 * r0, r1 <- r1 ^ 2, rf31 <- X
    mov(tmud, rf31)
    mov(tmua, rf11)
    tmuwt().add(rf11, rf11, rf3)
    mov(tmud, rf0)
    mov(tmua, rf11)
    tmuwt().add(rf11, rf11, rf3)
    mov(tmud, rf1)
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

def test_signal_ldtmu():

    with Driver() as drv:

        code = drv.program(qpu_signal_ldtmu)
        X = drv.alloc((16, ), dtype = 'float32')
        Y = drv.alloc((3, 16), dtype = 'float32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.random.randn(*X.shape).astype('float32')
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y[0] == X).all()
        assert (Y[1] == 2).all()
        assert (Y[2] == 4).all()

# rot signal with rN source performs as a full rotate
@qpu
def qpu_full_rotate(asm):

    eidx(rf0, sig = ldunifrf(rf5))
    mov(rf10, rf5, sig = ldunifrf(rf5))
    mov(rf3, 4)
    shl(rf3, rf3, 4).mov(rf11, rf5)

    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw).add(rf10, rf10, rf3)
    nop()
    nop()
    nop(sig = ldtmu(rf0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().add(rf1, rf0, rf0)#, sig = rot(i))
        ror(rf1, rf1, i)
        mov(tmud, rf1)
        mov(tmua, rf11)
        tmuwt().add(rf11, rf11, rf3)

    for i in range(-15, 16):
        mov(rf5, i)
        nop() # require
        nop().add(rf1, rf0, rf0)#, sig = rot(i))
        ror(rf1, rf1, i)
        mov(tmud, rf1)
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

def test_full_rotate():

    with Driver() as drv:

        code = drv.program(qpu_full_rotate)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((2, len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X,X]) * 2
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[:,ix] == expected[(-rot%16):(-rot%16)+16]).all()


# rotate alias
@qpu
def qpu_rotate_alias(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        rotate(r1, r0, i)       # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        nop().rotate(r1, r0, i) # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        rotate(r1, r0, r5)       # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        nop().rotate(r1, r0, r5) # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_rotate_alias():

    with Driver() as drv:

        code = drv.program(qpu_rotate_alias)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((4, len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X,X])
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[:,ix] == expected[(-rot%16):(-rot%16)+16]).all()


# rot signal with rfN source performs as a quad rotate
@qpu
def qpu_quad_rotate(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(rf32))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().add(r1, rf32, rf32, sig = rot(i))
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        nop().add(r1, rf32, rf32, sig = rot(r5))
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_quad_rotate():

    with Driver() as drv:

        code = drv.program(qpu_quad_rotate)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((2, len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X.reshape(4,4)]*2, axis=1)*2
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[:,ix] == expected[:,(-rot%4):(-rot%4)+4].ravel()).all()


# quad_rotate alias
@qpu
def qpu_quad_rotate_alias(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(rf32))
    nop() # required before rotate

    for i in range(-15, 16):
        quad_rotate(r1, rf32, i)       # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        nop().quad_rotate(r1, rf32, i) # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        quad_rotate(r1, rf32, r5)       # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        nop().quad_rotate(r1, rf32, r5) # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_quad_rotate_alias():

    with Driver() as drv:

        code = drv.program(qpu_quad_rotate_alias)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((4, len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X.reshape(4,4)]*2, axis=1)
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[:,ix] == expected[:,(-rot%4):(-rot%4)+4].ravel()).all()


# instruction with r5rep dst performs as a full broadcast
@qpu
def qpu_full_broadcast(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().mov(r5rep, r0, sig = [rot(ix) for ix in [i] if ix != 0] )
        mov(tmud, r5)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_full_broadcast():

    with Driver() as drv:

        code = drv.program(qpu_full_broadcast)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = X
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[ix] == expected[(-rot%16)].repeat(16)).all()


# broadcast alias
@qpu
def qpu_broadcast_alias(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().mov(broadcast, r0, sig = [rot(ix) for ix in [i] if ix != 0] )
        mov(tmud, r5)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_broadcast_alias():

    with Driver() as drv:

        code = drv.program(qpu_broadcast_alias)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = X
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[ix] == expected[(-rot%16)].repeat(16)).all()


# instruction with r5 dst performs as a quad broadcast
@qpu
def qpu_quad_broadcast(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().mov(r5, r0, sig = [rot(ix) for ix in [i] if ix != 0] )
        mov(tmud, r5)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_quad_broadcast():

    with Driver() as drv:

        code = drv.program(qpu_quad_broadcast)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X,X])
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[ix] == expected[(-rot%16):(-rot%16)+16:4].repeat(4)).all()


# instruction with r5 dst performs as a quad broadcast
@qpu
def qpu_quad_broadcast_alias(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().mov(quad_broadcast, r0, sig = [rot(ix) for ix in [i] if ix != 0] )
        mov(tmud, r5)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_quad_broadcast_alias():

    with Driver() as drv:

        code = drv.program(qpu_quad_broadcast_alias)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X,X])
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[ix] == expected[(-rot%16):(-rot%16)+16:4].repeat(4)).all()