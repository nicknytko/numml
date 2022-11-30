import torch
import torch.linalg
import numml.sparse as sp
import time
import sys
import scipy.linalg
import scipy.sparse.linalg
import cupy
import cupyx.scipy.linalg
import cupyx.scipy.sparse.linalg
import pandas as pd
import pickle

# Use one thread for CPU core for fair comparison
torch.set_num_threads(1)

def cuda_sync():
    torch.cuda.synchronize()
    cupy.cuda.runtime.deviceSynchronize()

def time_op(f, it):
    cuda_sync()
    t_start = time.time()
    for i in range(it):
        f()
    cuda_sync()
    t_end = time.time()
    return t_end - t_start

if not torch.cuda.is_available():
    print('Timing tests must run with CUDA support enabled.')
    sys.exit(1)

gpu = torch.device('cuda:0')
tests = {}

### Timing tests

class TestResult:
    def __init__(self, **kwargs):
        self.data = dict(kwargs)

class TestResultsMidrule:
    def __init__(self):
        pass

class TestResults:
    def __init__(self, column_names):
        self.names = column_names
        self.calculated_columns = {}
        self.results = []
        self.are_columns_calculated = True

    def set_calculated_col(self, name, f):
        self.calculated_columns[name] = f

    def _get_column_datatypes(self):
        types = {}
        for row in self.results:
            if not isinstance(row, TestResult):
                continue

            for k, v in row.data.items():
                if k not in types:
                    types[k] = type(v)
                else:
                    ex_type = types[k]
                    cr_type = type(v)

                    if cr_type == str:
                        types[k] = str
                    elif ex_type == int and cr_type == float:
                        types[k] = float
                    elif ex_type == bool and (cr_type == int or cr_type == float):
                        types[k] = cr_type
        return types

    def _calculate_columns(self):
        if self.are_columns_calculated:
            return

        calc_set = set(self.calculated_columns.keys())
        for row in self.results:
            if not isinstance(row, TestResult):
                continue

            for to_calc in calc_set:
                if to_calc not in row.data:
                    row.data[to_calc] = self.calculated_columns[to_calc](row.data)

        self.are_columns_calculated = True

    def add(self, d):
        self.results.append(d)
        if len(self.calculated_columns) > 0 and isinstance(d, TestResult):
            self.are_columns_calculated = False

    def to_pandas(self):
        self._calculate_columns()

        list_of_results = filter(lambda x: isinstance(x, TestResult), self.results)
        list_of_dict = list(map(lambda x: x.data, list_of_results))

        df = pd.DataFrame.from_dict(list_of_dict)
        df.rename(columns=self.names, inplace=True)

        return df

    def to_latex(self):
        self._calculate_columns()
        types = self._get_column_datatypes()

        # Starting tabular
        out = r'\begin{tabular}{'
        for T in types.values():
            if T == str:
                out += 'l'
            else:
                out += 'r'
        out += '}\n\\toprule\n'

        # Header
        out += ' & '.join(self.names.values()) + ' \\\\ \n'
        out += '\\midrule\n'

        # Each row
        for i, row in enumerate(self.results):
            if isinstance(row, TestResult):
                data = row.data
                for j, key in enumerate(self.names):
                    if key in data:
                        if types[key] == str:
                            out += data[key]
                        elif types[key] == int:
                            out += f'{data[key]:,}'
                        elif types[key] == float:
                            out += f'{data[key]:.4f}'
                        elif types[key] == bool:
                            out += ("True" if data[key] else "False")
                    if j != len(self.names) - 1:
                        out += " & "
                out += ' \\\\ \n'
            elif isinstance(row, TestResultsMidrule) and i != len(self.results) - 1:
                out += '\\midrule\\n'

        # ending tabular
        out += '\\bottomrule\n\\end{tabular}\n'
        return out

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump((self.names, self.results), f)

    def load(self_or_fname, fname_opt):
        if fname_opt is None:
            # Calling as static
            fname = self_or_fname
            with open(fname, 'rb') as f:
                names, results = pickle.load(f)
            res = TestResults(names)
            res.results = results
            return res
        else:
            # Calling like results.load(fname)
            self = self_or_fname
            fname = fname_opt

            with open(fname, 'rb') as f:
                names, results = pickle.load(f)

            self.names = names
            self.results = results
            self.calculated_columns = {}
            self.are_columns_calculated = True


def test_spmv(results):
    N = 32_768
    A = sp.eye(N)*2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    x = torch.rand(N)

    A_c = A.to(gpu)
    x_c = x.to(gpu)

    A_sci = A.to_scipy_csr()
    x_sci = x.numpy()
    A_cp = A_c.to_cupy_csr()
    x_cp = cupy.asarray(x_c)

    A_cp@x_cp

    it = 1000
    results.add(TestResult(
        name='SpMV Forward',
        cpu_numml=time_op(lambda: A@x, it),
        cpu_sci=time_op(lambda: A_sci@x_sci, it),
        gpu_numml=time_op(lambda: A_c@x_c, it),
        gpu_cupy=time_op(lambda: A_cp @ x_cp, it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    x.requires_grad = True
    A_c.requires_grad = True
    x_c.requires_grad = True

    it = 1000
    results.add(TestResult(
        name='SpMV Backward',
        cpu_numml=time_op(lambda: (A@x).sum().backward(), it),
        gpu_numml=time_op(lambda: (A_c@x_c).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['spmv'] = test_spmv


def test_dmv(results):
    N = 32_768
    A = (sp.eye(N)*2 - sp.eye(N, k=-1) - sp.eye(N, k=1)).to_dense()
    x = torch.rand(N)

    A_c = A.to(gpu)
    x_c = x.to(gpu)

    A_np = A.numpy()
    x_np = x.numpy()

    A_cp = cupy.asarray(A_c)
    x_cp = cupy.asarray(x_c)

    A_cp@x_cp

    it = 1000
    results.add(TestResult(
        name='DMV Forward',
        cpu_numml=time_op(lambda: A@x, it),
        cpu_sci=time_op(lambda: A_np@x_np, it),
        gpu_numml=time_op(lambda: A_c@x_c, it),
        gpu_cupy=time_op(lambda: A_cp @ x_cp, it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    x.requires_grad = True
    A_c.requires_grad = True
    x_c.requires_grad = True

    it = 1000
    results.add(TestResult(
        name='DMV Backward',
        cpu_numml=time_op(lambda: (A@x).sum().backward(), it),
        gpu_numml=time_op(lambda: (A_c@x_c).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['dmv'] = test_dmv


def test_spspmm(results):
    N = 16_384
    A = sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    B = (-A).copy()

    A_c = A.to(gpu)
    B_c = B.to(gpu)

    A_sci = A.to_scipy_csr()
    B_sci = B.to_scipy_csr()
    A_cp = A_c.to_cupy_csr()
    B_cp = B_c.to_cupy_csr()

    A_c@B_c
    A_cp@B_cp

    it = 2
    results.add(TestResult(
        name='SpSpMM Forward',
        cpu_numml=time_op(lambda: A@B, it),
        cpu_sci=time_op(lambda: A_sci@B_sci, it),
        gpu_numml=time_op(lambda: A_c@B_c, it),
        gpu_cupy=time_op(lambda: A_cp@B_cp, it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    B.requires_grad = True
    A_c.requires_grad = True
    B_c.requires_grad = True

    it = 2
    results.add(TestResult(
        name='SpSpMM Backward',
        cpu_numml=time_op(lambda: (A@B).sum().backward(), it),
        gpu_numml=time_op(lambda: (A_c@B_c).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['spspmm'] = test_spspmm


def test_spdmm(results):
    N = 16_384
    A = sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    B = (-A).copy().to_dense()

    A_c = A.to(gpu)
    B_c = B.to(gpu)

    A_sci = A.to_scipy_csr()
    B_np = B.numpy()
    A_cp = A_c.to_cupy_csr()
    B_cp = cupy.asarray(B_c)

    A_cp @ B_cp

    it = 2
    results.add(TestResult(
        name='SpDMM Forward',
        cpu_numml=time_op(lambda: A@B, it),
        cpu_sci=time_op(lambda: A_sci@B_np, it),
        gpu_numml=time_op(lambda: A_c@B_c, it),
        gpu_cupy=time_op(lambda: A_cp@B_cp, it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    B.requires_grad = True
    A_c.requires_grad = True
    B_c.requires_grad = True

    it = 2
    results.add(TestResult(
        name='SpSpMM Backward',
        cpu_numml=time_op(lambda: (A@B).sum().backward(), it),
        gpu_numml=time_op(lambda: (A_c@B_c).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['spdmm'] = test_spdmm


def test_ddmm(results):
    N = 16_384
    A = (sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)).to_dense()
    B = (-A).clone()

    A_c = A.to(gpu)
    B_c = B.to(gpu)

    A_np = A.numpy()
    B_np = B.numpy()
    A_cp = cupy.asarray(A_c)
    B_cp = cupy.asarray(B_c)

    A_cp @ B_cp

    it = 2
    results.add(TestResult(
        name='DDMM Forward',
        cpu_numml=time_op(lambda: A@B, it),
        cpu_sci=time_op(lambda: A_np@B_np, it),
        gpu_numml=time_op(lambda: A_c@B_c, it),
        gpu_cupy=time_op(lambda: A_cp@B_cp, it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    B.requires_grad = True
    A_c.requires_grad = True
    B_c.requires_grad = True

    it = 2
    results.add(TestResult(
        name='DDMM Backward',
        cpu_numml=time_op(lambda: (A@B).sum().backward(), it),
        gpu_numml=time_op(lambda: (A_c@B_c).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['ddmm'] = test_ddmm


def test_spadd(results):
    N = 32_768
    A = sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
    B = (-A).copy()

    A_c = A.to(gpu)
    B_c = B.to(gpu)

    A_sci = A.to_scipy_csr()
    B_sci = B.to_scipy_csr()
    A_cp = A_c.to_cupy_csr()
    B_cp = B_c.to_cupy_csr()

    A_cp + B_cp

    it = 10
    results.add(TestResult(
        name='Sp + Sp Forward',
        cpu_numml=time_op(lambda: A+B, it),
        cpu_sci=time_op(lambda: A_sci+B_sci, it),
        gpu_numml=time_op(lambda: A_c+B_c, it),
        gpu_cupy=time_op(lambda: A_cp+B_cp, it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    B.requires_grad = True
    A_c.requires_grad = True
    B_c.requires_grad = True

    it = 10
    results.add(TestResult(
        name='Sp + Sp Backward',
        cpu_numml=time_op(lambda: (A+B).sum().backward(), it),
        gpu_numml=time_op(lambda: (A_c+B_c).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['spadd'] = test_spadd


def test_dadd(results):
    N = 32_768
    A = (sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)).to_dense()
    B = (-A).clone()

    A_c = A.to(gpu)
    B_c = B.to(gpu)

    A_np = A.numpy()
    B_np = B.numpy()
    A_cp = cupy.asarray(A_c)
    B_cp = cupy.asarray(B_c)

    A_cp + B_cp

    it = 10
    results.add(TestResult(
        name='D + D Forward',
        cpu_numml=time_op(lambda: A+B, it),
        cpu_sci=time_op(lambda: A_np+B_np, it),
        gpu_numml=time_op(lambda: A_c+B_c, it),
        gpu_cupy=time_op(lambda: A_cp+B_cp, it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    B.requires_grad = True
    A_c.requires_grad = True
    B_c.requires_grad = True

    it = 10
    results.add(TestResult(
        name='D + D Backward',
        cpu_numml=time_op(lambda: (A+B).sum().backward(), it),
        gpu_numml=time_op(lambda: (A_c+B_c).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['dadd'] = test_dadd


# def test_splu():
#     N = 128
#     A = sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)
#     A_c = A.to(gpu)

#     it = 20
#     fwd_cpu_t = time_op(lambda: sp.splu(A), it)
#     fwd_gpu_t = time_op(lambda: sp.splu(A_c), it)
#     print_results('SPLU Forward', fwd_cpu_t, fwd_gpu_t, it, N)

#     A.requires_grad = True
#     A_c.requires_grad = True

#     it = 10
#     bwd_cpu_t = time_op(lambda: sp.splu(A).sum().backward(), it)
#     bwd_gpu_t = time_op(lambda: sp.splu(A_c).sum().backward(), it)
#     print_results('SPLU Backward', bwd_cpu_t, bwd_gpu_t, it, N)
# tests['splu'] = test_splu


# def test_dlu():
#     N = 128
#     A = (sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)).to_dense()
#     A_c = A.to(gpu)

#     it = 20
#     fwd_cpu_t = time_op(lambda: sp.splu(A), it)
#     fwd_gpu_t = time_op(lambda: sp.splu(A_c), it)
#     print_results('DLU Forward', fwd_cpu_t, fwd_gpu_t, it, N)

#     A.requires_grad = True
#     A_c.requires_grad = True

#     it = 10
#     bwd_cpu_t = time_op(lambda: sp.splu(A).sum().backward(), it)
#     bwd_gpu_t = time_op(lambda: sp.splu(A_c).sum().backward(), it)
#     print_results('DLU Backward', bwd_cpu_t, bwd_gpu_t, it, N)
# tests['dlu'] = test_dlu


def test_sptrsv(results):
    N = 32_768
    A = sp.eye(N) * 2 - sp.eye(N, k=-1)
    x = torch.ones(N)

    A_c = A.to(gpu)
    x_c = x.to(gpu)

    A_sci = A.to_scipy_csr()
    x_np = x.numpy()
    A_cp = A_c.to_cupy_csr()
    x_cp = cupy.asarray(x_c)

    cupyx.scipy.sparse.linalg.spsolve_triangular(A_cp, x_cp, lower=True)

    it = 20
    results.add(TestResult(
        name='SpTRSV Forward',
        cpu_numml=time_op(lambda: A.solve_triangular(upper=False, unit=False, b=x), it),
        cpu_sci=time_op(lambda: scipy.sparse.linalg.spsolve_triangular(A_sci, x_np, lower=True), it),
        gpu_numml=time_op(lambda: A_c.solve_triangular(upper=False, unit=False, b=x_c), it),
        gpu_cupy=time_op(lambda: cupyx.scipy.sparse.linalg.spsolve_triangular(A_cp, x_cp, lower=True), it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    A_c.requires_grad = True

    it = 10
    results.add(TestResult(
        name='SpTRSV Backward',
        cpu_numml=time_op(lambda: A.solve_triangular(upper=False, unit=False, b=x).sum().backward(), it),
        gpu_numml=time_op(lambda: A_c.solve_triangular(upper=False, unit=False, b=x_c).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['sptrsv'] = test_sptrsv


def test_dtrsv(results):
    N = 32_768
    A = (sp.eye(N) * 2 - sp.eye(N, k=-1)).to_dense()
    x = torch.ones(N)

    A_c = A.to(gpu)
    x_c = x.to(gpu)

    A_np = A.numpy()
    x_np = x.numpy()
    A_cp = cupy.asarray(A_c)
    x_cp = cupy.asarray(x_c)

    cupyx.scipy.linalg.solve_triangular(A_cp, x_cp, lower=True)

    it = 20
    results.add(TestResult(
        name='DTRSV Forward',
        cpu_numml=time_op(lambda: torch.linalg.solve_triangular(A, x.reshape((-1, 1)), upper=False, unitriangular=False), it),
        cpu_sci=time_op(lambda: scipy.linalg.solve_triangular(A_np, x_np, lower=True), it),
        gpu_numml=time_op(lambda: torch.linalg.solve_triangular(A_c, x_c.reshape((-1, 1)), upper=False, unitriangular=False), it),
        gpu_cupy=time_op(lambda: cupyx.scipy.linalg.solve_triangular(A_cp, x_cp, lower=True), it),
        iterations=it,
        n=N
    ))

    A.requires_grad = True
    A_c.requires_grad = True

    it = 10
    results.add(TestResult(
        name='DTRSV Backward',
        cpu_numml=time_op(lambda: torch.linalg.solve_triangular(A, x.reshape((-1, 1)), upper=False, unitriangular=False).sum().backward(), it),
        gpu_numml=time_op(lambda: torch.linalg.solve_triangular(A_c, x_c.reshape((-1, 1)), upper=False, unitriangular=False).sum().backward(), it),
        iterations=it,
        n=N
    ))
    results.add(TestResultsMidrule())
tests['dtrsv'] = test_dtrsv

### Arg checks

results = TestResults({
    'name': 'Test Name',
    'cpu_numml': 'CPU (ours/Torch, s)',
    'cpu_sci': 'CPU (NumPy/SciPy, s)',
    'gpu_numml': 'GPU (ours/Torch, s)',
    'gpu_cupy': 'GPU (CuPy, s)',
    'gpu_spdup': 'GPU Speedup (ours)',
    'iterations': 'Iterations',
    'n': 'N'
    })
results.set_calculated_col('gpu_spdup', lambda row: row['cpu_numml'] / row['gpu_numml'])

args = sys.argv[1:]
if len(args) == 0:
    for i, v in enumerate(tests.values()):
        print(f'Running {v.__name__}...', end='', flush=True)
        v(results)
        print(f' ... done', flush=True)
else:
    for arg in args:
        if not arg in tests:
            print(f'Unknown test "{arg}".  Valid choices are: {list(tests.keys())}.')
            sys.exit(1)
    for i, v in enumerate(args):
        print(f'Running {v}...', end='', flush=True)
        tests[v](results)
        print(f' ... done', flush=True)

results._calculate_columns()
print(results.to_pandas())
print(results.to_latex())

results.save('timing_results.pkl')
