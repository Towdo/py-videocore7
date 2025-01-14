
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


# Disclaimer:
# Horizontal vector rotation does not seem to be a thing anymore as of VC7.
# This algorithm is optimized to profit from vector rotation.
# This version is thus far from optimal.


from time import clock_gettime, CLOCK_MONOTONIC
import numpy as np
from videocore6 import pack_unpack
from videocore6.driver import Driver
from videocore6.assembler import qpu


def getsec():
    return clock_gettime(CLOCK_MONOTONIC)


@qpu
def load_params(asm, thread, regs):

    g = globals()

    for i in range(5):
        g['r'+str(i)] = g['rf' + str(i+52)]

    if thread == 1:
        bxor(r0, r0, r0, sig = ldunifrf(rf0))
    elif thread == 8:
        #  8 threads (1 threads / qpu)
        tidx(r0, sig = ldunifrf(rf0))
        shr(r0, r0, 2)
        mov(r1, 0b1111)
    elif thread == 16:
        # 16 threads (2 threads / qpu)
        tidx(r0, sig = ldunifrf(rf0))
        shr(r0, r0, 1)
        mov(r1, 1)
        shl(r1, r1, 5)
        sub(r1, r1, 1)
    else:
        assert thread in [1,8,16]

    band(r3, r0, r1, sig = ldunifrf(rf1))
    shl(r0, rf1, 2)
    umul24(r0, r0, r3)
    eidx(r1).add(r0, r0, rf0)
    shl(r1, r1, 2)
    mov(r3, 4)
    shl(r3, r3, 4).add(r0, r0, r1)
    n = len(regs)
    for i in range(n):
        mov(tmua, r0, sig = thrsw)
        nop()
        nop()
        nop(sig = ldtmu(r1))
        mov(r5rep, r1)
        mov(regs[i], rf0)
        add(r0, r0, 4)
    nop()
    nop()
    nop()

@qpu
def qpu_sgemm_rnn_naive(asm, thread):

    params = [
        'P',
        'Q',
        'R',
        'A_base',
        'A_stride',
        'B_base',
        'B_stride',
        'C_base',
        'C_stride',
        'alpha',
        'beta',
    ]

    values = [
        'A_cur',
        'B_cur',
        'C_cur',
        'i', 'j', 'k',
    ]

    g = globals()
    for i, reg in enumerate(params + values):
        g['reg_' + reg] = g['rf' + str(i+34)]

    for i in range(6):
        g['r'+str(i)] = g['rf' + str(i+52)]

    load_params(asm, thread, [g['reg_' + reg] for reg in params])

    add(r0, reg_P, 15)
    shr(r0, r0, 4)
    shl(r0, r0, 4)
    add(r1, reg_R, 15)
    shr(r1, r1, 4)
    shl(r1, r1, 6)
    umul24(r3, r0, reg_A_stride)
    add(reg_A_base, reg_A_base, r3)
    add(reg_B_base, reg_B_base, r1)
    umul24(r3, r0, reg_C_stride)
    add(reg_C_base, reg_C_base, r3)
    add(reg_C_base, reg_C_base, r1)

    for i in range(0, 33): #FIXME: NOT OPTIMAL
        mov(rf[i], 0.0)

    # i=(p+15)/16.
    add(r0, reg_P, 15)
    shr(reg_i, r0, 4)
    with loop as li:

        # j=(r+15)/16
        add(r0, reg_R, 15)
        shr(reg_j, r0, 4)
        with loop as lj:

            shl(r0, reg_i, 4)
            umul24(r3, r0, reg_C_stride)
            shl(r1, reg_j, 6)
            sub(reg_C_cur, reg_C_base, r3)
            sub(reg_C_cur, reg_C_cur, r1)
            umul24(r3, r0, reg_A_stride)
            sub(reg_A_cur, reg_A_base, r3)
            sub(reg_B_cur, reg_B_base, r1)

            mov(reg_k, reg_Q)
            with loop as lk:

                eidx(r0)
                mov(r1, reg_A_cur).add(reg_A_cur, reg_A_cur, 4)
                mov(tmua, r1, sig = thrsw)      # r1 is address in A
                shl(r2, r0, 2)
                add(r2, r2, reg_B_cur).add(reg_B_cur, reg_B_cur, reg_B_stride)
                mov(tmua, r2, sig = thrsw)      # r1 is a address in B

                nop(sig = ldtmu(r0))        # Loads alpha?
                mov(r5rep, r0)
                nop(sig = ldtmu(r4))        #
                nop().fmul(r3, rf0, r4)

                for i in range(2, 17):
                    add(r1, r1, reg_A_stride)
                    mov(tmua, r1, sig = thrsw)
                    nop()
                    nop(sig = ldtmu(r0))
                    mov(r5rep, r0)

                    fadd(rf[i-1], rf[i-1], r3).fmul(r3, rf0, r4)

                fadd(rf16, rf16, r3)

                sub(reg_k, reg_k, 1, cond = 'pushz')
                lk.b(cond = 'anyna')
                nop() # delay slot
                nop() # delay slot
                nop() # delay slot

            eidx(r0)
            shl(r0, r0, 2)
            add(r1, reg_C_cur, r0)
            mov(tmua, r1, sig = thrsw).add(r1, r1, reg_C_stride)
            fmul(rf[1], rf[1], reg_alpha)
            for i in range(2, 17):
                mov(tmua, r1, sig = thrsw).add(r1, r1, reg_C_stride)
                fmul(rf[i], rf[i], reg_alpha, sig = ldtmu(rf[i+15]))
            mov(r0, reg_beta).fmul(r3, rf[17], reg_beta, sig = ldtmu(rf[32]))
            for i in range(1, 17):
                fadd(rf[i], rf[i], r3).fmul(r3, rf[i+17], r0)

            eidx(r0)
            shl(r0, r0, 2)
            add(r1, reg_C_cur, r0)
            for i in range(1, 17):
                mov(tmud, rf[i])
                mov(tmua, r1).add(r1, r1, reg_C_stride)
                mov(rf[i], 0.0)
                mov(rf[i+16], 0.0)
                tmuwt()

            sub(reg_j, reg_j, 1, cond = 'pushz')
            lj.b(cond = 'anyna')
            nop() # delay slot
            nop() # delay slot
            nop() # delay slot

        sub(reg_i, reg_i, 1, cond = 'pushz')
        li.b(cond = 'anyna')
        nop()
        nop()
        nop()

    # DEBUGGING: Prints values into rows of C
    # Jump to R.debug and replace rf1 with the stuff you want to print
    # This is hopefully 1212480 lol
    #
    # L.debug
    #
    # nop()
    # nop()
    # nop()
    # mov(rf60, 1)
    # shl(rf60, rf60, 3)
    # add(rf60, rf60, 1)
    # shl(rf60, rf60, 2)
    # add(rf60, rf60, 1)
    # shl(rf60, rf60, 9)
    # shl(rf60, rf60, 6)

    # eidx(rf61)
    # shl(rf61, rf61, 2)
    # add(rf60, rf60, rf61)

    # tidx(rf61)
    # shl(rf61, rf61, 6)
    # add(rf60, rf60, rf61)

    # mov(tmud, rf1)
    # mov(tmua, rf60)

    # DEBUGGING END

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def sgemm_rnn_naive():

    thread = 8  #FIXME: ONLY 8 Works currently

    assert thread == 8, "Algorithm currenlty only works for 8 threads."

    P = 512
    Q = 512
    R = 512

    assert P % (16 * 2) == 0
    assert R % (16 * 4) == 0

    with Driver() as drv:

        code = drv.program(lambda asm: qpu_sgemm_rnn_naive(asm, thread))

        A = drv.alloc((P, Q), dtype = 'float32')
        B = drv.alloc((Q, R), dtype = 'float32')
        C = drv.alloc((P, R), dtype = 'float32')

        np.random.seed(0)
        alpha = np.random.randn()
        beta = np.random.randn()
        A_ref = np.random.randn(*A.shape).astype(A.dtype)
        B_ref = np.random.randn(*B.shape).astype(B.dtype)
        C_ref = np.random.randn(*C.shape).astype(C.dtype)

        A[:] = A_ref
        B[:] = B_ref
        C[:] = C_ref

        start = getsec()
        C_ref[:] = alpha * A_ref.dot(B_ref) + beta * C_ref
        time_ref = getsec() - start

        def block_2x4_params(i, j):
            tile_P = P // 2
            tile_R = R // 4
            return [
                tile_P, Q, tile_R,
                A.addresses()[tile_P*i, 0       ],
                A.strides[0],
                B.addresses()[0       , tile_R*j],
                B.strides[0],
                C.addresses()[tile_P*i, tile_R*j],
                C.strides[0],
                *pack_unpack('f', 'I', [alpha, beta]),
            ]

        unif_params = drv.alloc((thread, len(block_2x4_params(0,0))), dtype = 'uint32')
        for th in range(thread):
            unif_params[th] = block_2x4_params(th // 4, th % 4)

        unif = drv.alloc(2, dtype = 'uint32')
        unif[0] = unif_params.addresses()[0,0]
        unif[1] = unif_params.shape[1]

        start = getsec()
        drv.execute(code, unif.addresses()[0], thread = thread)
        time_gpu = getsec() - start

        np.set_printoptions(threshold=np.inf)
        

        def Gflops(sec):
            return (2 * P * Q * R + 3 * P * R) / sec * 1e-9

        print(f'==== sgemm example ({P}x{Q} times {Q}x{R}) ====')
        print(f'numpy: {time_ref:.4} sec, {Gflops(time_ref):.4} Gflop/s')
        print(f'QPU:   {time_gpu:.4} sec, {Gflops(time_gpu):.4} Gflop/s')
        print(f'Minimum absolute error: {np.min(np.abs(C - C_ref))}')
        print(f'Maximum absolute error: {np.max(np.abs(C - C_ref))}')
        print(f'Minimum relative error: {np.min(np.abs((C - C_ref) / C_ref))}')
        print(f'Maximum relative error: {np.max(np.abs((C - C_ref) / C_ref))}')


def main():

    sgemm_rnn_naive()


if __name__ == '__main__':
    main()
