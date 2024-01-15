
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

# branch (destination from relative imm)
@qpu
def qpu_branch_rel_imm(asm):

    eidx(rf0, sig = ldunifrf(rf10))
    nop(sig = ldunifrf(rf11))
    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf1))

    b(2*8, cond = 'always')
    nop()
    nop()
    nop()
    add(rf1, rf1, 1)
    add(rf1, rf1, 1)
    add(rf1, rf1, 1) # jump comes here
    add(rf1, rf1, 1)

    mov(tmud, rf1)
    mov(tmua, rf11)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_branch_rel_imm():

    with Driver() as drv:

        code = drv.program(qpu_branch_rel_imm)
        X = drv.alloc((16, ), dtype = 'uint32')
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == X + 2).all()


# branch (destination from absolute imm)
@qpu
def qpu_branch_abs_imm(asm, absimm):

    eidx(rf0, sig = ldunifrf(rf10))
    nop(sig = ldunifrf(rf11))
    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf1))

    b(absimm, absolute = True, cond = 'always')
    nop()
    nop()
    nop()
    add(rf1, rf1, 1)
    add(rf1, rf1, 1)
    add(rf1, rf1, 1) # jump comes here
    add(rf1, rf1, 1)

    mov(tmud, rf1)
    mov(tmua, rf11)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_branch_abs_imm():

    with Driver() as drv:

        @qpu
        def qpu_dummy(asm):
            nop()
        dummy = drv.program(qpu_dummy)
        code = drv.program(lambda asm: qpu_branch_abs_imm(asm, int(dummy.addresses()[0]+16*8)))
        X = drv.alloc((16, ), dtype = 'uint32')
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == X + 2).all()


# branch (destination from label)
@qpu
def qpu_branch_rel_label(asm):

    eidx(rf0, sig = ldunifrf(rf10))
    nop(sig = ldunifrf(rf11))
    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf1))

    b(R.foo, cond = 'always')
    nop()
    nop()
    nop()
    add(rf1, rf1, 1)
    L.foo
    add(rf1, rf1, 1) # jump comes here
    L.bar
    add(rf1, rf1, 1)
    L.baz
    add(rf1, rf1, 1)

    mov(tmud, rf1)
    mov(tmua, rf11)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_branch_rel_label():

    with Driver() as drv:

        code = drv.program(qpu_branch_rel_label)
        X = drv.alloc((16, ), dtype = 'uint32')
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == X + 3).all()


# branch (destination from regfile)
@qpu
def qpu_branch_abs_reg(asm):

    eidx(rf0, sig = ldunifrf(rf10))
    nop(sig = ldunifrf(rf11))
    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf12))

    mov(rf1, 0)
    b(rf12, cond = 'always')
    nop()
    nop()
    nop()
    L.label
    add(rf1, rf1, 1)
    add(rf1, rf1, 1)
    add(rf1, rf1, 1)
    add(rf1, rf1, 1) # jump comes here

    mov(tmud, rf1)
    mov(tmua, rf11)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_branch_abs_reg():

    with Driver() as drv:

        code = drv.program(qpu_branch_abs_reg)
        X = drv.alloc((16, ), dtype = 'uint32')
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = code.addresses()[0] + 17*8
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == 1).all()


# branch (destination from link_reg)
@qpu
def qpu_branch_link_reg(asm, set_subroutine_link, use_link_reg_direct):

    eidx(rf0, sig = ldunifrf(rf10))
    nop(sig = ldunifrf(rf11))
    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf2))

    mov(rf12, 0)
    mov(rf13, 0)
    b(R.init_link, cond = 'always', set_link = True)
    nop() # delay slot
    nop() # delay slot
    nop() # delay slot
    L.init_link

    # subroutine returns to here if set_subroutine_link is False.
    add(rf13, rf13, 1)

    # jump to subroutine once.
    mov(null, rf12, cond = 'pushz')
    b(R.subroutine, cond = 'alla', set_link = set_subroutine_link)
    mov(rf12, 1) # delay slot
    nop()       # delay slot
    nop()       # delay slot

    # subroutine returns to here if set_subroutine_link is True.
    mov(rf1, 4)
    shl(rf1, rf1, 4)
    mov(tmud, rf13) # rf3 will be 1 if set_subroutine_link, else 2.
    mov(tmua, rf11).add(rf11, rf11, rf1)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

    L.subroutine

    mov(rf1, 4)
    shl(rf1, rf1, 4)
    mov(tmud, rf2)
    mov(tmua, rf11).add(rf11, rf11, rf1)
    tmuwt()

    if use_link_reg_direct:
        b(link, cond = 'always')
    else:
        lr(rf32) # lr instruction reads link register
        b(rf32, cond = 'always')
    nop() # delay slot
    nop() # delay slot
    nop() # delay slot

def test_branch_link_reg():

    for set_subroutine_link, expected in [(False, 2), (True, 1)]:
        for use_link_reg_direct in [False, True]:
            with Driver() as drv:

                code = drv.program(lambda asm: qpu_branch_link_reg(asm, set_subroutine_link, use_link_reg_direct))
                X = drv.alloc(16, dtype = 'uint32')
                Y = drv.alloc((2, 16), dtype = 'uint32')
                unif = drv.alloc(2, dtype = 'uint32')

                X[:] = (np.random.randn(16) * 1024).astype('uint32')
                Y[:] = 0.0

                unif[0] = X.addresses()[0]
                unif[1] = Y.addresses()[0,0]

                start = time.time()
                drv.execute(code, unif.addresses()[0])
                end = time.time()

                assert (Y[0] == X).all()
                assert (Y[1] == expected).all()


# uniform branch (destination from uniform relative value)
@qpu
def qpu_uniform_branch_rel(asm):

    eidx(rf0, sig = ldunifrf(rf10))
    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)

    b(R.label, cond = 'always').unif_addr()
    nop()
    nop()
    nop()
    L.label
    nop(sig = ldunifrf(tmud))
    mov(tmua, rf10)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_uniform_branch_rel():

    with Driver() as drv:

        code = drv.program(qpu_uniform_branch_rel)
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(5, dtype = 'uint32')

        Y[:] = 0.0

        unif[0] = Y.addresses()[0]
        unif[1] = 8 # relative address for uniform branch
        unif[2] = 5
        unif[3] = 6
        unif[4] = 7 # uniform branch point here

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == 7).all()


# uniform branch (destination from uniform absolute value)
@qpu
def qpu_uniform_branch_abs(asm):

    eidx(rf0, sig = ldunifrf(rf10))
    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)

    b(R.label, cond = 'always').unif_addr(absolute = True)
    nop()
    nop()
    nop()
    L.label
    nop(sig = ldunifrf(tmud))
    mov(tmua, rf10)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_uniform_branch_abs():

    with Driver() as drv:

        code = drv.program(qpu_uniform_branch_abs)
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(5, dtype = 'uint32')

        Y[:] = 0.0

        unif[0] = Y.addresses()[0]
        unif[1] = unif.addresses()[3] # absolute address for uniform branch
        unif[2] = 5
        unif[3] = 6 # uniform branch point here
        unif[4] = 7

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == 6).all()


# uniform branch (destination from register)
@qpu
def qpu_uniform_branch_reg(asm):


    eidx(rf0, sig = ldunifrf(rf10))
    nop(sig = ldunifrf(rf11))
    shl(rf0, rf0, 2)
    add(rf10, rf10, rf0)
    add(rf11, rf11, rf0)

    mov(tmua, rf10, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf12))

    b(R.label, cond = 'always').unif_addr(rf12)
    nop()
    nop()
    nop()
    L.label
    nop(sig = ldunifrf(rf13))
    mov(tmud, rf13)
    mov(tmua, rf11)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_uniform_branch_reg():

    with Driver() as drv:

        code = drv.program(qpu_uniform_branch_reg)
        X = drv.alloc((16, ), dtype = 'uint32')
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(6, dtype = 'uint32')

        X[1] = unif.addresses()[4] # absolute address for uniform branch
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0]
        unif[2] = 3
        unif[3] = 4
        unif[4] = 5 # uniform branch point here
        unif[5] = 6

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == 5).all()
