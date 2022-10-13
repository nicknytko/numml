import torch
import numml.sparse as sp
import time
import sys


def time_op(f, it):
    t_start = time.time()
    for i in range(it):
        f()
    torch.cuda.synchronize()
    t_end = time.time()
    return t_end - t_start

def print_bar(thick=False):
    if thick:
        print(f'|{"":=<32}|{"":=<14}|{"":=<14}|{"":=<14}|{"":=<14}|{"":=<6}|')
    else:
        print(f'|{"":-<32}|{"":-<14}|{"":-<14}|{"":-<14}|{"":-<14}|{"":-<6}|')

def print_header():
    print_bar(True)
    print(f'|{"Test Name":32}|{"CPU Time (s)":>14}|{"GPU Time (s)":>14}|{"GPU Speedup":>14}|{"Iterations":>14}|{"N":>6}|')
    print_bar(False)

def print_results(test_name, cpu_t, gpu_t, it, N):
    print(f'|{test_name:32.32}|{cpu_t:14.4f}|{gpu_t:14.4f}|{cpu_t/gpu_t:14.4f}|{it:14d}|{N:6d}|')


if not torch.cuda.is_available():
    print('Timing tests must run with CUDA support enabled.')
    sys.exit(1)

gpu = torch.device('cuda:0')
tests = {}

### Timing tests

def test_spmv():
    N = 1024
    A = sp.eye(N)*2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    x = torch.rand(N)

    A_c = A.to(gpu)
    x_c = x.to(gpu)

    it = 1000
    fwd_cpu_t = time_op(lambda: A@x, it)
    fwd_gpu_t = time_op(lambda: A_c@x_c, it)
    print_results('SPMV Forward', fwd_cpu_t, fwd_gpu_t, it, N)

    A.requires_grad = True
    x.requires_grad = True
    A_c.requires_grad = True
    x_c.requires_grad = True

    it = 1000
    bwd_cpu_t = time_op(lambda: (A@x).sum().backward(), it)
    bwd_gpu_t = time_op(lambda: (A_c@x_c).sum().backward(), it)
    print_results('SPMV Backward', bwd_cpu_t, bwd_gpu_t, it, N)
tests['spmv'] = test_spmv

def test_spspmm():
    N = 1024
    A = sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    B = (-A).copy()

    A_c = A.to(gpu)
    B_c = B.to(gpu)

    it = 500
    fwd_cpu_t = time_op(lambda: A@B, it)
    fwd_gpu_t = time_op(lambda: A_c@B_c, it)
    print_results('SPSPMM Forward', fwd_cpu_t, fwd_gpu_t, it, N)

    A.requires_grad = True
    B.requires_grad = True
    A_c.requires_grad = True
    B_c.requires_grad = True

    it = 500
    bwd_cpu_t = time_op(lambda: (A@B).sum().backward(), it)
    bwd_gpu_t = time_op(lambda: (A_c@B_c).sum().backward(), it)
    print_results('SPSPMM Backward', bwd_cpu_t, bwd_gpu_t, it, N)
tests['spspmm'] = test_spspmm

def test_spadd():
    N = 1024
    A = sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    B = (-A).copy()

    A_c = A.to(gpu)
    B_c = B.to(gpu)

    it = 500
    fwd_cpu_t = time_op(lambda: A+B, it)
    fwd_gpu_t = time_op(lambda: A_c+B_c, it)
    print_results('SP + SP Forward', fwd_cpu_t, fwd_gpu_t, it, N)

    A.requires_grad = True
    B.requires_grad = True
    A_c.requires_grad = True
    B_c.requires_grad = True

    it = 500
    bwd_cpu_t = time_op(lambda: (A+B).sum().backward(), it)
    bwd_gpu_t = time_op(lambda: (A_c+B_c).sum().backward(), it)
    print_results('SP + SP Backward', bwd_cpu_t, bwd_gpu_t, it, N)
tests['spadd'] = test_spadd

def test_spdmm():
    N = 128
    A = sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    B = (-A).copy().to_dense()

    A_c = A.to(gpu)
    B_c = B.to(gpu)

    it = 20
    fwd_cpu_t = time_op(lambda: A@B, it)
    fwd_gpu_t = time_op(lambda: A_c@B_c, it)
    print_results('SPDMM Forward', fwd_cpu_t, fwd_gpu_t, it, N)

    A.requires_grad = True
    B.requires_grad = True
    A_c.requires_grad = True
    B_c.requires_grad = True

    it = 10
    bwd_cpu_t = time_op(lambda: (A@B).sum().backward(), it)
    bwd_gpu_t = time_op(lambda: (A_c@B_c).sum().backward(), it)
    print_results('SPDMM Backward', bwd_cpu_t, bwd_gpu_t, it, N)
tests['spdmm'] = test_spdmm

def test_splu():
    N = 128
    A = sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    A_c = A.to(gpu)

    it = 20
    fwd_cpu_t = time_op(lambda: sp.splu(A), it)
    fwd_gpu_t = time_op(lambda: sp.splu(A_c), it)
    print_results('SPLU Forward', fwd_cpu_t, fwd_gpu_t, it, N)

    A.requires_grad = True
    A_c.requires_grad = True

    it = 10
    bwd_cpu_t = time_op(lambda: sp.splu(A).sum().backward(), it)
    bwd_gpu_t = time_op(lambda: sp.splu(A_c).sum().backward(), it)
    print_results('SPLU Backward', bwd_cpu_t, bwd_gpu_t, it, N)
tests['splu'] = test_splu

def test_trisolve():
    N = 512
    A = sp.eye(N) * 2 - sp.eye(N, k=-1)
    A_c = A.to(gpu)
    x = torch.ones(N)
    x_c = x.to(gpu)

    it = 20
    fwd_cpu_t = time_op(lambda: A.solve_triangular(upper=False, unit=False, b=x), it)
    fwd_gpu_t = time_op(lambda: A_c.solve_triangular(upper=False, unit=False, b=x_c), it)
    print_results('SP Tri-Solve Forward', fwd_cpu_t, fwd_gpu_t, it, N)

    A.requires_grad = True
    A_c.requires_grad = True

    it = 10
    bwd_cpu_t = time_op(lambda: A.solve_triangular(upper=False, unit=False, b=x).sum().backward(), it)
    bwd_gpu_t = time_op(lambda: A_c.solve_triangular(upper=False, unit=False, b=x_c).sum().backward(), it)
    print_results('SP Tri-Solve Backward', bwd_cpu_t, bwd_gpu_t, it, N)
tests['trisolve'] = test_trisolve

### Arg checks

args = sys.argv[1:]
if len(args) == 0:
    print_header()
    for i, v in enumerate(tests.values()):
        v()
        print_bar(i == len(tests) - 1)
else:
    for arg in args:
        if not arg in tests:
            print(f'Unknown test "{arg}".  Valid choices are: {list(tests.keys())}.')
            sys.exit(1)
    print_header()
    for i, v in enumerate(args):
        tests[v]()
        print_bar(i == len(args) - 1)
