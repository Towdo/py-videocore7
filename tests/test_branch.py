
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np

# branch (destination from relative imm)
@qpu
def qpu_branch_rel_imm(asm):

    eidx(r0, sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    b(2*8, cond = 'always')
    nop()
    nop()
    nop()
    add(r1, r1, 1)
    add(r1, r1, 1)
    add(r1, r1, 1) # jump comes here
    add(r1, r1, 1)

    mov(tmud, r1)
    mov(tmua, rf1)
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

    eidx(r0, sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    b(absimm, absolute = True, cond = 'always')
    nop()
    nop()
    nop()
    add(r1, r1, 1)
    add(r1, r1, 1)
    add(r1, r1, 1) # jump comes here
    add(r1, r1, 1)

    mov(tmud, r1)
    mov(tmua, rf1)
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

    eidx(r0, sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    b(R.foo, cond = 'always')
    nop()
    nop()
    nop()
    add(r1, r1, 1)
    L.foo
    add(r1, r1, 1) # jump comes here
    L.bar
    add(r1, r1, 1)
    L.baz
    add(r1, r1, 1)

    mov(tmud, r1)
    mov(tmua, rf1)
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

    eidx(r0, sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf2))

    mov(r1, 0)
    b(rf2, cond = 'always')
    nop()
    nop()
    nop()
    L.label
    add(r1, r1, 1)
    add(r1, r1, 1)
    add(r1, r1, 1)
    add(r1, r1, 1) # jump comes here

    mov(tmud, r1)
    mov(tmua, rf1)
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


# uniform branch (destination from uniform relative value)
@qpu
def qpu_uniform_branch_rel(asm):

    eidx(r0, sig = ldunifrf(rf0))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)

    b(R.label, cond = 'always').unif_addr()
    nop()
    nop()
    nop()
    L.label
    nop(sig = ldunifrf(tmud))
    mov(tmua, rf0)
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

    eidx(r0, sig = ldunifrf(rf0))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)

    b(R.label, cond = 'always').unif_addr(absolute = True)
    nop()
    nop()
    nop()
    L.label
    nop(sig = ldunifrf(tmud))
    mov(tmua, rf0)
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


    eidx(r0, sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf2))

    b(R.label, cond = 'always').unif_addr(rf2)
    nop()
    nop()
    nop()
    L.label
    nop(sig = ldunifrf(rf3))
    mov(tmud, rf3)
    mov(tmua, rf1)
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
