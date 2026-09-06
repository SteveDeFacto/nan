"""Exercise the patched copy against the pinned original, under ASan/UBSan.

Run after building the CPU backend in the patched llama.cpp checkout. Tests
copy only synthetic data and need neither model weights nor a GPU.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--library-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--baseline-ref', default='HEAD')
p.add_argument('--baseline-repo', type=Path)
a = p.parse_args()
repo = Path(__file__).resolve().parents[1]
a.source = a.source.resolve()
a.library_dir = a.library_dir.resolve()
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=True)

path = 'ggml/src/ggml-cpu/ops.cpp'
baseline = subprocess.check_output(['git', 'show', a.baseline_ref + ':' + path],
                                   cwd=a.baseline_repo or a.source, text=True)
patched = (a.source / path).read_text()
def function(text, name):
    start = text.index('static ', text.index(name) - 20)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end] + '\n'

(a.output / 'baseline-copy.h').write_text(
    function(baseline, 'ggml_compute_forward_dup_same_cont(') +
    function(baseline, 'ggml_compute_forward_dup_bytes('))
(a.output / 'candidate-copy.h').write_text(
    function(patched, 'ggml_compute_forward_dup_bytes_parallel('))
for header in ['ggml/include/ggml.h', 'ggml/include/gguf.h',
               'ggml/src/ggml-impl.h', 'ggml/src/ggml-cpu/ggml-cpu-impl.h']:
    shutil.copyfile(a.source / header, a.output / Path(header).name)
shutil.copyfile(repo / 'test/native/ggml-parallel-copy.cpp', a.output / 'check.cpp')
binary = a.output / 'check'
subprocess.run(['c++', '-std=c++17', '-O2', '-g', '-fopenmp',
                '-DGGML_MAX_NAME=128', '-fsanitize=address,undefined',
                '-fno-omit-frame-pointer', '-I' + str(a.output),
                str(a.output / 'check.cpp'), '-L' + str(a.library_dir),
                '-lggml-base', '-Wl,-rpath,' + str(a.library_dir), '-o', str(binary)],
               check=True)
subprocess.run([str(binary)], check=True)
