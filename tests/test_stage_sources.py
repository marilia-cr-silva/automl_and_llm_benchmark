"""Source scans of the stage scripts (they are notebook-style and cannot be imported)."""

import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
DA_SCRIPTS = [
    "04-rf/domain_adaptation/random_forest_baseline_100_concat.py",
    "05-ag/domain_adaptation/autogluon_benchmark.py",
    "06-flaml/domain_adaptation/flaml_benchmark_baseline_100_concat.py",
]
LLM_SCRIPTS = ["07-llama/final_checkpoint_llama_3_2.py", "08-mistral/mistral.py"]
STAGE_SCRIPTS = sorted(p.relative_to(SRC).as_posix() for p in SRC.glob("0[1-8]-*/**/*.py"))


def text(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def test_stage_scripts_were_found():
    assert len(STAGE_SCRIPTS) == 13


@pytest.mark.parametrize("script", DA_SCRIPTS)
def test_domain_adaptation_scripts_require_canonical(script):
    assert re.search(r"^require_canonical\(", text(script), re.M)


@pytest.mark.parametrize("script", LLM_SCRIPTS)
def test_llm_scripts_pass_the_collection(script):
    source = text(script)
    assert "current_collection()" in source
    assert "collection=COLLECTION_NAME" in source
    assert "static_parts(COLLECTION_NAME," in source


@pytest.mark.parametrize(
    ("script", "loader"),
    [
        (LLM_SCRIPTS[0], "generator = build_llama()"),
        (LLM_SCRIPTS[1], "= build_mistral()"),
    ],
)
def test_llm_scripts_fail_fast_before_model_loading(script, loader):
    source = text(script)
    guard = source.index("if COLLECTION_NAME is None:")
    assert guard < source.index(loader)
