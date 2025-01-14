# Adapting py-videocore to work with the new Raspberry Pi 5

Disclaimer: I'm currently not actively working on this anymore since there doesn't seem to be any performance to be gained over the CPU.
In fact, it seems to be challenging enough to just beat a single core of the CPU using the whole GPU in any real world task.
Feel free to use this project as a reference on how the GPU works. But think twice before building a project on top of this!


I mostly have no idea what I'm doing so please feel free to help me!

Current status
- [X] Basic ALU Operations (see tests/test_alu.py)
- [X] Conditionals  (should work)
- [ ] Updated all tests to work without accumulators
  - [X] alu
  - [X] branch
  - [X] condition_codes
  - [X] driver
  - [X] drm
  - [X] labels
  - [ ] parallel
  - [ ] sfu
  - [ ] signals
  - [ ] tmu
  - [ ] unifa
  - [X] v3d
- [ ] Updated all examples to work without accumulators and horizontal vector rotations
  - [X] sgemm.py
  - [ ] Rest
- [X] Fixed benchmarks

Current bugs:
- [ ] Unpacking and Packing in one operation causes problems? (tests/test_alu.py:fmul)


# py-videocore7

A Python library for GPGPU programming on Raspberry Pi 5, which realizes
assembling and running QPU programs.

For Raspberry Pi Zero/1/2/3, use
[nineties/py-videocore](https://github.com/nineties/py-videocore) instead.

For Raspberry Pi 4, use
[Idein/py-videocore6](https://github.com/Idein/py-videocore6) instead.


## About VideoCore VII QPU

Raspberry Pi 5 (BCM2712) has a GPU named VideoCore VII QPU in its SoC.
The basic instruction set (add/mul ALU dual issue, three delay slots et al.)
remains the same as VideoCore VI QPU of Raspberry Pi 4, and some units
now perform differently.
For instance, horizontal vector rotation does not seem to be possible anymore.

Theoretical peak performance of QPUs are as follows.

- VideoCore IV QPU @ 250MHz: 250 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 24 [Gflop/s]
- VideoCore IV QPU @ 300MHz: 300 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 28.8 [Gflop/s]
- VideoCore VI QPU @ 500MHz: 500 [MHz] x 2 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 32 [Gflop/s]
- VideoCore VII QPU @ 800MHz: 800 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 76.8 [Gflop/s]

(In my current tests, the GPU clock frequency seems to be 950 MHz?.. This would result in 91.2 [Gflop/s])


## Requirements

`py-videocore7` communicates with the V3D hardware through `/dev/dri/card0`,
which is exposed by the DRM V3D driver.
To access the device, you need to belong to `video` group or be `root` user.
If you choose the former, run `sudo usermod --append --groups video $USER`
(re-login to take effect).


## Installation (THIS IS NOT UPDATED YET)

You can install `py-videocore6` directly using `pip`:

```console
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python3-pip python3-numpy
$ pip3 install --user --upgrade pip setuptools wheel
$ pip3 install --user git+https://github.com/Idein/py-videocore6.git
```

If you are willing to run tests and examples, install `py-videocore6` after
cloning it:

```console
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python3-pip python3-numpy libatlas3-base
$ python3 -m pip install --user --upgrade pip setuptools wheel
$ git clone https://github.com/Idein/py-videocore6.git
$ cd py-videocore6/
$ python3 -m pip install --target sandbox/ --upgrade . nose
```


## Running tests and examples

In the `py-videocore6` directory cloned above:

```console
$ python3 setup.py build_ext --inplace
$ PYTHONPATH=sandbox/ python3 -m nose -v -s
```

```console
$ PYTHONPATH=sandbox/ python3 examples/sgemm.py
==== sgemm example (1024x1024 times 1024x1024) ====
numpy: 0.6986 sec, 3.078 Gflop/s
QPU:   0.5546 sec, 3.878 Gflop/s
Minimum absolute error: 0.0
Maximum absolute error: 0.0003814697265625
Minimum relative error: 0.0
Maximum relative error: 0.13375753164291382
```

```console
$ PYTHONPATH=sandbox/ python3 examples/summation.py
==== summaton example (32.0 Mi elements) ====
Preparing for buffers...
Executing on QPU...
0.01853448400004254 sec, 7241.514141947083 MB/s
```

```console
$ PYTHONPATH=sandbox/ python3 examples/memset.py
==== memset example (64.0 MiB) ====
Preparing for buffers...
Executing on QPU...
0.01788834699993913 sec, 3751.5408215319367 MB/s
```

```console
$ PYTHONPATH=sandbox/ python3 examples/scopy.py
==== scopy example (16.0 Mi elements) ====
Preparing for buffers...
Executing on QPU...
0.02768789600000332 sec, 2423.761776625857 MB/s
```

```console
$ sudo PYTHONPATH=sandbox/ python3 examples/pctr_gpu_clock.py
==== QPU clock measurement with performance counters ====
500.529835 MHz
```

You may see lower performance without `force_turbo=1` in `/boot/config.txt`.


## References

- DRM V3D driver which controls QPU via hardware V3D registers: [linux/drivers/gpu/drm/v3d](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/gpu/drm/v3d)
- Mesa library which partially includes the QPU instruction set: [mesa/src/broadcom/qpu](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/broadcom/qpu)
- Mesa also includes QPU program disassembler, which can be tested with: [Terminus-IMRC/vc6qpudisas](https://github.com/Terminus-IMRC/vc6qpudisas)
