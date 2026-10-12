import subprocess
import sys

MODULES = [
    "collection",
    "env",
    "io",
    "labels",
    "paths",
    "random_forest",
    "relations",
    "resources",
    "results",
    "selection",
    "threads",
]

BLOCK = 'sys.modules.update({"pyarrow": None, "sklearn": None, "torch": None, "tensorflow": None})'


def run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_common_modules_do_not_import_heavy_dependencies():
    imports = "; ".join(f"import automl_and_llm_benchmark.common.{name}" for name in MODULES)
    proc = run(f"import sys; {BLOCK}; import automl_and_llm_benchmark.common; {imports}")
    assert proc.returncode == 0, proc.stderr


def test_heavy_modules_not_imported_eagerly():
    imports = "; ".join(f"import automl_and_llm_benchmark.common.{name}" for name in MODULES)
    # pandas itself may import pyarrow when it is installed, so compare with a baseline.
    code = (
        "import sys; import numpy, pandas; "
        "heavy = ('pyarrow', 'sklearn', 'torch', 'tensorflow', 'autogluon', 'flaml'); "
        "before = {m for m in heavy if m in sys.modules}; "
        f"import automl_and_llm_benchmark.common; {imports}; "
        "added = [m for m in heavy if m in sys.modules and m not in before]; "
        "assert not added, added"
    )
    proc = run(code)
    assert proc.returncode == 0, proc.stderr


def test_light_modules_do_not_import_numpy_or_pandas():
    code = (
        "import sys; "
        "import automl_and_llm_benchmark.common.env, "
        "automl_and_llm_benchmark.common.threads, "
        "automl_and_llm_benchmark.common.paths, "
        "automl_and_llm_benchmark.common.resources, "
        "automl_and_llm_benchmark.common.collection; "
        "assert 'numpy' not in sys.modules, 'numpy imported'; "
        "assert 'pandas' not in sys.modules, 'pandas imported'"
    )
    proc = run(code)
    assert proc.returncode == 0, proc.stderr
