
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

# FIXME: Fix those tests (require horizontal vector rotation)


@qpu
def cost(asm):
    mov(rf0, 8)
    shl(rf0, rf0, 8)
    shl(rf0, rf0, 8)
    with loop as l:
        sub(rf0, rf0, 1, cond = 'pushn')
        l.b(cond = 'anyna')
        nop()
        nop()
        nop()

@qpu
def qpu_serial(asm):

    nop(sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    nop(sig = ldunifrf(rf2))
    nop(sig = ldunifrf(rf3))

    eidx(rf10)
    shl(rf10, rf10, 2)
    add(rf2, rf2, rf10)
    add(rf3, rf3, rf10)
    mov(rf13, 4)
    shl(rf13, rf13, 4)

    for i in range(16):
        mov(tmua, rf2, sig = thrsw).add(rf2, rf2, rf13)
        nop()
        nop()
        nop(sig = ldtmu(rf10))
        mov(tmud, rf10)
        mov(tmua, rf3, sig = thrsw).add(rf3, rf3, rf13)
        tmuwt()

    cost(asm)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

# This code requires 16 thread execution.
# If # of thread < 16, thread id (= (tidx & 0b111110) >> 1) could be discontiguous.
# If # of thread > 16, thread id (= (tidx & 0b111110) >> 1) could be duplicated.
@qpu
def qpu_parallel_16(asm):

    tidx(rf10, sig = ldunifrf(rf0))
    shr(rf10, rf10, 1)
    mov(rf11, 1)
    shl(rf11, rf11, 5)
    sub(rf11, rf11, 1)
    band(rf31, rf10, rf11) # rf31 = (qpu_id * 2) + (thread_id >> 1)

    # rf31 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    nop(sig = ldunifrf(rf1))      # rf1 = unif[0,1]
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf31)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)
    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf10))                          # unif[th,2:18]
    mov(rf15rep, rf10)
    mov(rf2, rf15).rotate(rf15rep, rf10, -1)            # rf2 = unif[th,2]
    mov(rf3, rf15)                                  # rf3 = unif[th,3]

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf2, rf12, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf32))

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf32)
    add(tmua, rf3, rf12)
    tmuwt()

    cost(asm)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_parallel_16():

    with Driver() as drv:

        thread = 16

        serial_code = drv.program(qpu_serial)
        parallel_code = drv.program(qpu_parallel_16)
        X = drv.alloc((thread, 16), dtype = 'float32')
        Ys = drv.alloc((thread, 16), dtype = 'float32')
        Yp = drv.alloc((thread, 16), dtype = 'float32')
        unif = drv.alloc((thread, 4), dtype = 'uint32')

        X[:] = np.random.randn(*X.shape)
        Ys[:] = -1
        Yp[:] = -1

        unif[:,0] = unif.addresses()[:,0]
        unif[:,1] = unif.shape[1]
        unif[:,2] = X.addresses()[:,0]
        unif[:,3] = Ys.addresses()[:,0]

        start = time.time()
        drv.execute(serial_code, unif.addresses()[0,0])
        end = time.time()
        serial_cost = end - start

        unif[:,3] = Yp.addresses()[:,0]

        start = time.time()
        drv.execute(parallel_code, unif.addresses()[0,0], thread=thread)
        end = time.time()
        parallel_cost = end - start

        np.set_printoptions(threshold=np.inf)

        assert (X == Ys).all()
        assert (X == Yp).all()
        assert parallel_cost < serial_cost * 2

# If remove `barrierid` in this code, `test_barrier` will fail.
@qpu
def qpu_barrier(asm):

    tidx(rf10, sig = ldunifrf(rf0)) # rf0 = unif[0,0]
    shr(rf12, rf10, 2)
    band(rf11, rf10, 0b11)            # thread_id
    band(rf12, rf12, 0b1111)          # qpu_id
    shr(rf11, rf11, 1)
    shl(rf12, rf12, 1)
    add(rf31, rf11, rf12)             # rf31 = (qpu_id * 2) + (thread_id >> 1)

    nop(sig = ldunifrf(rf1))      # rf1 = unif[0,1]

    # rf31 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf31)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)
    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf10))                          # unif[th,2:18]
    mov(rf15rep, rf10)
    mov(rf2, rf15).rotate(rf15rep, rf10, -1)            # rf2 = unif[th,2]
    mov(rf3, rf15)                                  # rf3 = unif[th,3]

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf2, rf12, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf10))

    mov(rf11, rf31)
    shl(rf11, rf11, 8)
    L.loop
    sub(rf11, rf11, 1, cond = 'pushn')
    b(R.loop, cond = 'anyna')
    nop()
    nop()
    nop()

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf10)
    add(tmua, rf3, rf12)
    tmuwt()

    barrierid(syncb, sig = thrsw)

    add(rf32, rf31, 1)
    band(rf32, rf32, 0b1111) # rf32 = (rf31 + 1) mod 16

    # rf32 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf32)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)
    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf10))                          # unif[(th+1)%16,2:18]
    mov(rf15rep, rf10)
    mov(rf4, rf15).rotate(rf15rep, rf10, -1)            # rf4 = unif[(th+1)%16,2]
    mov(rf5, rf15)                                  # rf5 = unif[(th+1)%16,3]

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf5, rf12, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf10))

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf10)
    add(tmua, rf3, rf12)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_barrier():

    with Driver() as drv:

        thread = 16

        code = drv.program(qpu_barrier)
        X = drv.alloc((thread, 16), dtype = 'float32')
        Y = drv.alloc((thread, 16), dtype = 'float32')
        unif = drv.alloc((thread, 4), dtype = 'uint32')

        X[:] = np.random.randn(*X.shape)
        Y[:] = -1

        unif[:,0] = unif.addresses()[:,0]
        unif[:,1] = unif.shape[1]
        unif[:,2] = X.addresses()[:,0]
        unif[:,3] = Y.addresses()[:,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0,0], thread=thread)
        end = time.time()

        np.set_printoptions(threshold=np.inf)

        assert (Y == np.concatenate([X[1:],X[:1]])).all()
