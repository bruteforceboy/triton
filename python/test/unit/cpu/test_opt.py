import os
import platform
import re
import pytest
import torch

import triton
import triton.language as tl
from triton._C.libtriton import cpu
"""
This test is only for the CPU backend.
"""


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


def is_x86():
    return is_cpu() and \
        triton.runtime.driver.active.get_current_target().arch == "x86_64"


def test_scalar_pointer_arith(device):

    @triton.jit
    def kernel(src, dst, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs)
        tl.store(dst + offs, x)

    src = torch.rand((128, ), dtype=torch.float32, device=device)
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, BLOCK_SIZE=128)
    assert (src == res).all()

    # Check TTCIR doesn't have pointer extraction from a tensor.
    ttcir = meta.asm["ttcir"]
    assert ttcir.count("extract") == 0


@pytest.mark.parametrize("size", [32, 128, 135])
@pytest.mark.parametrize("tile_size", [16])
def test_optimize_tile_mask(size, tile_size, device):
    pytest.xfail("masked vector ops no longer eliminated by upstream canonicalizers")

    @triton.jit
    def kernel(src, dst, size, TILE_SIZE: tl.constexpr):
        for i in range(0, tl.cdiv(size, TILE_SIZE)):
            offs = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
            mask = offs < size
            x = tl.load(src + offs, mask=mask, other=0)
            tl.store(dst + offs, x, mask=mask)

    src = torch.rand((size, ), dtype=torch.float32, device='cpu')
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, size, TILE_SIZE=tile_size)
    assert (src == res).all()

    # Check number of masked loads and stores.
    tttcir = meta.asm["tttcir"]
    masked_loads = tttcir.count("maskedload")
    masked_stores = tttcir.count("maskedstore")
    if size % tile_size == 0:
        assert masked_loads == 0
        assert masked_stores == 0
    else:
        assert masked_loads == 1
        assert masked_stores == 1


# Regression test for compilation failure in masks optimization
def test_vec_cdiv(device):

    @triton.jit
    def kernel(in_ptr, out_ptr):
        offs = tl.arange(0, 16)
        x = tl.load(in_ptr + offs)
        res = (x + 15) // 16
        tl.store(out_ptr + offs, res)

    arg0 = torch.zeros((16, ), dtype=torch.int32)
    arg1 = torch.empty_like(arg0)
    kernel[(1, )](arg0, arg1)


@pytest.mark.cpu
def test_cpu_triple_matches_compiled_target(device):
    if device != "cpu":
        pytest.skip("CPU target triple test requires --device cpu.")

    @triton.jit
    def kernel(dst):
        tl.store(dst, 1)

    output = torch.empty((), dtype=torch.int32, device=device)
    compiled = kernel[(1, )](output)
    assert output.item() == 1

    triple = cpu.llvm.get_cpu_triple()
    target = re.search(r'target triple = "([^"]+)"', compiled.asm["llir"])
    assert target is not None
    assert target.group(1) == triple
    machine = platform.machine().lower()
    # On macOS, both Python and LLVM can report arm64. Apply aliases to both
    # values to avoid comparing aarch64 with arm64 and failing incorrectly.
    arch_aliases = {"arm64": "aarch64", "amd64": "x86_64"}
    expected_arch = arch_aliases.get(machine, machine)
    reported_arch = triple.split("-")[0]
    assert arch_aliases.get(reported_arch, reported_arch) == expected_arch


@pytest.mark.cpu
@pytest.mark.parametrize("grid", [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 5, 7), (130, 1, 1), (1, 70, 2)])
@pytest.mark.parametrize("num_cpu_threads", [0, 1, 3, 200])
def test_cpu_launcher_program_ids(grid, num_cpu_threads, device):
    if device != "cpu":
        pytest.skip("CPU launcher test requires --device cpu.")

    @triton.jit
    def kernel(out_ptr):
        x = tl.program_id(0)
        y = tl.program_id(1)
        z = tl.program_id(2)
        nx = tl.num_programs(0)
        ny = tl.num_programs(1)
        base = ((z * ny + y) * nx + x) * 6
        tl.store(out_ptr + base + 0, x)
        tl.store(out_ptr + base + 1, y)
        tl.store(out_ptr + base + 2, z)
        tl.store(out_ptr + base + 3, nx)
        tl.store(out_ptr + base + 4, ny)
        tl.store(out_ptr + base + 5, tl.num_programs(2))

    # Grids smaller than the thread count, explicit thread counts, and empty
    # grids must still run every program exactly once with the right ids.
    gx, gy, gz = grid
    out = torch.full((max(gx * gy * gz, 1) * 6, ), -1, dtype=torch.int32, device=device)
    kernel[grid](out, num_cpu_threads=num_cpu_threads)
    expected = [[x, y, z, gx, gy, gz] for z in range(gz) for y in range(gy) for x in range(gx)]
    expected = torch.tensor(expected or [[-1] * 6], dtype=torch.int32, device=device).flatten()
    assert torch.equal(out, expected)
