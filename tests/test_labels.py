
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

@qpu
def qpu_label_with_namespace(asm):

    mov(rf0, 0)

    with namespace('ns1'):
        b(R.test, cond = 'always')
        nop()
        nop()
        nop()
        add(rf0, rf0, 10)
        L.test
        add(rf0, rf0, 1)

        with namespace('nested'):
            b(R.test, cond = 'always')
            nop()
            nop()
            nop()
            add(rf0, rf0, 10)
            L.test
            add(rf0, rf0, 1)

    with namespace('ns2'):
        b(R.test, cond = 'always')
        nop()
        nop()
        nop()
        add(rf0, rf0, 10)
        L.test
        add(rf0, rf0, 1)

    b(R.test, cond = 'always')
    nop()
    nop()
    nop()
    add(rf0, rf0, 10)
    L.test
    add(rf0, rf0, 1)

    with namespace('ns3'):
        b(R.test, cond = 'always')
        nop()
        nop()
        nop()
        add(rf0, rf0, 10)
        L.test
        add(rf0, rf0, 1)

    eidx(rf1, sig = ldunifrf(rf12))
    shl(rf1, rf1, 2)

    mov(tmud, rf0)
    add(tmua, rf12, rf1)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_label_with_namespace():

    with Driver() as drv:

        code = drv.program(qpu_label_with_namespace)
        data = drv.alloc(16, dtype = 'uint32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 1234

        unif[0] = data.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (data == 5).all()