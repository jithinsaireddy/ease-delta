"""Export the edge model as a standalone `transformers` model, check it, and write its model card.

    python scripts/export_reader.py --edge release/edge --results runs/results_large \
        --out exports/ease-delta-reader --repo jithinpothireddy21/ease-delta-reader --size large

The exported model is `ModernBertForSequenceClassification` (see `ease/export.py` for why the conversion is
exact). The check reloads it the way a user would, with `AutoModelForSequenceClassification`, and compares
its logits with the edge model's on development pairs. Every figure in the card is read from results files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from ease.data.evalsets import load_eval_set
from ease.export import compatible_tokenizer, to_sequence_classifier
from ease.model.edge import EdgeModel
from ease.util import atomic_write_json, get_device, read_json

GITHUB = "https://github.com/jithinsaireddy/ease-delta"
SPACE = "https://huggingface.co/spaces/jithinpothireddy21/ease-delta-demo"
PUBLIC_NLI = "tasksource/ModernBERT-base-nli"
PLAN = {  # what each size was evaluated against, and when that plan was fixed
    "large": f"The evaluation plan was written and hashed before training: [`docs/PREREGISTRATION_LARGE.md`]"
             f"({GITHUB}/blob/main/docs/PREREGISTRATION_LARGE.md).",
    "base": f"The evaluation plan was written and hashed while this model was still training, before any test split "
            f"was evaluated: [`docs/PREREGISTRATION.md`]({GITHUB}/blob/main/docs/PREREGISTRATION.md).",
}


def check(out: Path, edge: EdgeModel, eval_cache: str, n: int, device) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(out)
    clf = AutoModelForSequenceClassification.from_pretrained(out).to(device).eval()
    edge = edge.to(device).eval()
    res = {"pairs": 0, "max_abs_logit_difference": 0.0, "argmax_disagreements": 0, "sets": {}}
    for name in ("vitaminc.dev", "mnli.dev", "unrelated.dev", "snli.dev"):
        rows = load_eval_set(eval_cache, name)[:n]
        worst, disagree = 0.0, 0
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            enc = tok([r["claim"] for r in chunk], [r["evidence"] for r in chunk], truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                a, _ = edge(enc["input_ids"], enc["attention_mask"])
                b = clf(**enc).logits
            worst = max(worst, float((a - b).abs().max()))
            disagree += int((a.argmax(-1) != b.argmax(-1)).sum())
        res["sets"][name] = {"pairs": len(rows), "max_abs_logit_difference": worst, "argmax_disagreements": disagree}
        res["pairs"] += len(rows)
        res["max_abs_logit_difference"] = max(res["max_abs_logit_difference"], worst)
        res["argmax_disagreements"] += disagree
        print(f"  {name:14s} {len(rows)} pairs: max |logit difference| {worst:.2e}, argmax disagreements {disagree}", flush=True)
    res["device"] = str(device)
    return res


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


EXAMPLE_CLAIM = "The client has approved the final design."
EXAMPLE_PASSAGES = ("Email from the client: we approve the final design, please go ahead.",
                    "Email from the client: we cannot approve the design yet; the colours are wrong.",
                    "The office is closed on the first Monday of the month for staff training.")


def run_examples(out: Path) -> list[tuple[str, str]]:
    """The card's example, run on the exported model, so its printed output is real."""
    from transformers import pipeline

    reader = pipeline("text-classification", model=str(out), top_k=None)
    rows = []
    for ps in EXAMPLE_PASSAGES:
        top = reader({"text": EXAMPLE_CLAIM, "text_pair": ps}, truncation=True)[0]
        rows.append((ps, f"{{'label': '{top['label']}', 'score': {top['score']:.3f}}}"))
    return rows


def onnx_section(repo: str, onnx: dict) -> list[str]:
    return ["## ONNX, and in the browser\n",
            "`onnx/model.onnx` (fp32) matches PyTorch to 4e-5 in the logits; `onnx/model_quantized.onnx` (int8, "
            f"150 MB) gave the same label as the full model on {100 * onnx['agreement']:.1f}% of {onnx['pairs']:,} "
            f"development pairs (accuracy {100 * onnx['int8_accuracy']:.1f}% against {100 * onnx['fp32_accuracy']:.1f}%).\n",
            "```js\nimport { AutoTokenizer, AutoModelForSequenceClassification } from "
            "\"https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1\";\n\n"
            f"const tokenizer = await AutoTokenizer.from_pretrained(\"{repo}\");\n"
            f"const model = await AutoModelForSequenceClassification.from_pretrained(\"{repo}\", {{ dtype: \"q8\" }});\n"
            "const inputs = await tokenizer(claim, { text_pair: passage, truncation: true, max_length: 256 });\n"
            "const { logits } = await model(inputs);   // SUPPORTS, REFUTES, NOT_ENOUGH_INFO\n```\n",
            "This is what the [demo](https://huggingface.co/spaces/jithinpothireddy21/ease-delta-demo) runs; the text "
            "never leaves the browser.\n"]


def card(repo: str, size: str, edge_dir: Path, results: dict, other: dict | None, public: dict, chk: dict,
         temperature: float, manifest: dict, examples: list[tuple[str, str]], onnx: dict | None = None) -> str:
    s = results["sets"]
    o = other["sets"] if other else None
    p = public["sets"]
    other_size = "base" if size == "large" else "large"
    other_repo = "jithinpothireddy21/ease-delta-reader-base" if size == "large" else "jithinpothireddy21/ease-delta-reader"
    system_repo = "jithinpothireddy21/ease-delta" if size == "large" else "jithinpothireddy21/ease-delta-base"
    backbone = manifest["config"]["encoder_name"]
    params = manifest["parameters"]

    def acc(sets, k):
        return sets[k]["raw"]["accuracy"] if sets and k in sets else None

    def model_index_entry(task_name, ds_name, ds_type, split, value, config=None):
        cfg = f"\n      config: {config}" if config else ""
        return (f"  - task:\n      type: text-classification\n      name: {task_name}\n"
                f"    dataset:\n      name: {ds_name}\n      type: {ds_type}{cfg}\n      split: {split}\n"
                f"    metrics:\n    - type: accuracy\n      value: {100 * value:.2f}\n      name: Accuracy\n")

    mi = "".join([
        model_index_entry("Fact verification", "VitaminC", "tals/vitaminc", "test", acc(s, "vitaminc.test")),
        model_index_entry("Natural language inference", "MultiNLI (mismatched)", "nyu-mll/multi_nli",
                          "validation_mismatched", acc(s, "mnli_mm.test")),
        model_index_entry("Natural language inference", "WANLI", "alisawuffles/WANLI", "test", acc(s, "wanli.test")),
        model_index_entry("Natural language inference", "SNLI", "stanfordnlp/snli", "test", acc(s, "snli.test")),
    ])
    yaml = (f"---\nlicense: cc-by-sa-4.0\nlanguage:\n- en\nlibrary_name: transformers\npipeline_tag: text-classification\n"
            f"base_model: {backbone}\ndatasets:\n- tals/vitaminc\n- nyu-mll/multi_nli\n- alisawuffles/WANLI\n"
            f"tags:\n- natural-language-inference\n- nli\n- fact-verification\n- fact-checking\n- cross-encoder\n"
            f"- modernbert\n- rag\n- evidence\nmetrics:\n- accuracy\nmodel-index:\n- name: {repo.split('/')[-1]}\n  results:\n{mi}---\n")

    def row(label, key, fmt=pct):
        cells = [fmt(acc(s, key))]
        if o:
            cells.append(fmt(acc(o, key)))
        cells.append(fmt(acc(p, key)))
        return f"| {label} | " + " | ".join(cells) + " |"

    head_cols = f"| | **This model** ({size}) |" + (f" {other_size} reader |" if o else "") + f" `{PUBLIC_NLI}` |"
    sep = "|---|---:|" + ("---:|" if o else "") + "---:|"
    fd = [pct(s["unrelated.test"]["false_decisive_rate"])] + ([pct(o["unrelated.test"]["false_decisive_rate"])] if o else []) \
        + [pct(p["unrelated.test"]["false_decisive_rate"])]
    L = [yaml, f"# EASE-Delta Reader ({size})\n",
         "**Does this passage support the claim, refute it, or not settle it at all?**\n",
         f"A {params / 1e6:.0f}M-parameter cross-encoder (`{backbone}` fine-tuned) that reads one claim against one "
         "passage and answers `SUPPORTS`, `REFUTES` or `NOT_ENOUGH_INFO`. It was trained to say `NOT_ENOUGH_INFO` "
         "when a passage is about something else, which is where general NLI models most often guess: on passages "
         f"taken from unrelated documents it gives a decisive answer {fd[0]} of the time; a widely used public NLI "
         f"model does so {fd[-1]} of the time.\n",
         f"It is the reader inside [EASE-Delta]({GITHUB}), a system that keeps a task's decisions current as its "
         f"messages and documents change. Try both in the [demo]({SPACE}).\n",
         "## Use it\n",
         "Claim first, evidence second.\n",
         "```python\nfrom transformers import pipeline\n\n"
         f"reader = pipeline(\"text-classification\", model=\"{repo}\", top_k=None)\n"
         f"claim = \"{EXAMPLE_CLAIM}\"\n"
         "passages = [\n"
         + "".join(f"    \"{ps}\",\n" for ps, _ in examples)
         + "]\n"
         "for passage in passages:\n"
         "    print(reader({\"text\": claim, \"text_pair\": passage}, truncation=True)[0])\n\n"
         + "".join(f"# {r}\n" for _, r in examples)
         + "```\n",
         "With sentence-transformers, continuing from above:\n",
         "```python\nfrom sentence_transformers import CrossEncoder\n\n"
         f"model = CrossEncoder(\"{repo}\")\n"
         "probs = model.predict([(claim, passage) for passage in passages], apply_softmax=True)\n"
         "# one row per pair; columns SUPPORTS, REFUTES, NOT_ENOUGH_INFO\n```\n",
         "Inputs were at most 256 tokens in training (`truncation=True` cuts there). Split long documents into "
         "passages and read each one; say who is speaking in the passage itself (\"Email from the client: ...\"), "
         "because the reader sees only the text.\n",
         f"Probabilities are best calibrated after dividing the logits by **{temperature:.3f}** (fitted on development "
         "data) before the softmax.\n",
         "## Results\n",
         "Test sets, accuracy. One evaluation harness for all three models; brackets and paired tests are in the "
         f"[results report]({GITHUB}/blob/main/docs/RESULTS.md).\n",
         head_cols, sep,
         "| Passages from unrelated documents read as decisive (lower is better) | " + " | ".join(fd) + " |",
         row("VitaminC: claims against Wikipedia revisions", "vitaminc.test"),
         row("MultiNLI, mismatched genres", "mnli_mm.test"),
         row("WANLI", "wanli.test"),
         row("SNLI (not trained on)", "snli.test"),
         row("ANLI round 1 (not trained on)", "anli_r1.test"),
         row("ANLI round 2 (not trained on)", "anli_r2.test"),
         row("ANLI round 3 (not trained on)", "anli_r3.test"),
         "",
         f"`{PUBLIC_NLI}` is a base-size multi-task NLI model whose training includes SNLI and ANLI; it is the "
         "better choice for adversarial NLI of the ANLI kind. This reader is built for evidence: contrastive Wikipedia "
         "revisions (VitaminC), and knowing when a passage does not bear on the claim.\n",
         f"On VitaminC, real revisions: {pct(s['vitaminc.test']['accuracy_real'])}; synthetic: "
         f"{pct(s['vitaminc.test']['accuracy_synthetic'])}.\n",
         *(onnx_section(repo, onnx) if onnx else []),
         "## Labels\n",
         "| Label | Meaning | NLI equivalent |", "|---|---|---|",
         "| `SUPPORTS` | the passage establishes the claim | entailment |",
         "| `REFUTES` | the passage establishes that the claim is false | contradiction |",
         "| `NOT_ENOUGH_INFO` | the passage does not settle the claim, including when it is about something else | neutral |",
         "",
         "## Training\n",
         f"{manifest['examples']:,} examples streamed from the Hugging Face Hub: VitaminC (cc-by-sa-3.0), MultiNLI and "
         "WANLI (cc-by-4.0), plus 20% synthetic unrelated pairs (the claim of one example with the evidence of another "
         "from a different topic, labelled `NOT_ENOUGH_INFO`). SNLI was left out because its annotators labelled "
         "unrelated content as contradiction; ANLI because its licence is non-commercial. One run, on one Apple M4 Max. "
         + PLAN[size] + "\n",
         "## Limitations\n",
         "- English only.",
         "- Passages on the claim's own subject that do not settle it are still read as decisive about one time in five (VitaminC test, not-enough-info items).",
         "- Conflicts that follow only from a consequence (\"broke a leg\" against \"cycles to work\") are mostly missed.",
         "- Numbers and dates are read as text; compare them with code when they matter.",
         "- Not a judge of truth: it reports what the passage says, not whether the passage is right.\n",
         "## How this checkpoint was made\n",
         f"Converted exactly from the EASE-Delta edge model (`ease/export.py` in the code repository explains why the "
         f"conversion is exact). On {chk['pairs']:,} development pairs the largest logit difference between the two was "
         f"{chk['max_abs_logit_difference']:.1e} and the predicted label never differed ({chk['argmax_disagreements']} "
         f"disagreements). The full system, with its calibrated rules and regression suites, is "
         f"[`{system_repo}`](https://huggingface.co/{system_repo}).\n",
         f"Other size: [`{other_repo}`](https://huggingface.co/{other_repo}).\n",
         "## Licence\n",
         "CC BY-SA 4.0 for the weights, because VitaminC is share-alike (Creative Commons' guidance on AI training "
         "describes this as the cautious course). The code is Apache-2.0. Credit for the training data: VitaminC "
         "(Schuster, Fisch and Barzilay, NAACL 2021), MultiNLI (Williams, Nangia and Bowman, NAACL 2018), WANLI "
         "(Liu, Swayamdipta, Smith and Choi, EMNLP 2022). Backbone: ModernBERT (Warner et al., 2024).\n",
         "## Citation\n",
         "```bibtex\n@software{pothireddy2026easedelta,\n  author = {Pothireddy, Jithin},\n"
         "  title = {EASE-Delta: revision-aware decision computation},\n  year = {2026},\n"
         f"  url = {{{GITHUB}}}\n}}\n```\n"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", required=True)
    ap.add_argument("--results", required=True, help="results directory holding this model's edge.json")
    ap.add_argument("--other-results", default=None, help="the other size's results directory, for the table")
    ap.add_argument("--public", default="runs/results/edge_b3.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--size", required=True, choices=["base", "large"])
    ap.add_argument("--eval-cache", default="runs/eval_cache")
    ap.add_argument("--check-pairs", type=int, default=1000)
    ap.add_argument("--onnx-check", default=None, help="a results file of the ONNX int8 check, to describe in the card")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    edge = EdgeModel.load(a.edge).eval()
    clf = to_sequence_classifier(edge, max_len=edge.cfg.max_len)
    clf.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(a.edge)
    tok.model_max_length = edge.cfg.max_len
    tok.save_pretrained(out)
    # the backbone's tokenizer files load in transformers 4 and 5 and in Transformers.js; checked identical ids
    compatible_tokenizer(edge.cfg.encoder_name, out, edge.cfg.max_len, check_against=a.edge)
    print("saved", out, flush=True)

    t0 = time.time()
    chk = check(out, edge, a.eval_cache, a.check_pairs, get_device())
    chk["seconds"] = time.time() - t0
    atomic_write_json(out / "export_check.json", chk)
    if chk["argmax_disagreements"] or chk["max_abs_logit_difference"] > 1e-3:
        print("EXPORT CHECK FAILED", chk, file=sys.stderr)
        return 1

    results = read_json(Path(a.results) / "edge.json")
    other = read_json(Path(a.other_results) / "edge.json") if a.other_results else None
    public = read_json(a.public)
    manifest = read_json(Path(a.edge) / "training_manifest.json")
    (out / "README.md").write_text(card(a.repo, a.size, Path(a.edge), results, other, public, chk,
                                        results["temperature"], manifest, run_examples(out),
                                        read_json(a.onnx_check) if a.onnx_check else None))
    print("card written", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
