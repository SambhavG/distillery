import os, sys, json, math, time, random, itertools, threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.layout import Layout
from rich.table import Table
from rich import box
import questionary

console = Console()

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
except Exception as e:
    console.print(f"[red]Missing ML deps[/red]: {e}")
    sys.exit(1)

import modal

app = modal.App("distillery")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    params_b: float
    size_gb: float
    reasoning: bool


@dataclass
class Paths:
    system_prompt: Path
    problems: Path
    out_dir: Path


@dataclass
class DistillConfig:
    batch_size: int = 8
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    compute: str = "local"
    use_4bit: bool = False
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    epochs: int = 1
    lr: float = 5e-5
    train_bs: int = 2
    gradient_accum: int = 4
    seed: int = 7


MODELS: List[ModelSpec] = [
    ModelSpec(
        "Llama 3.1 70B Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        70,
        140,
        False,
    ),
    ModelSpec("Qwen2.5 72B Instruct", "Qwen/Qwen2.5-72B-Instruct", 72, 145, False),
    ModelSpec(
        "DeepSeek-R1 70B (reasoning)",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        70,
        140,
        True,
    ),
    ModelSpec(
        "Llama 3.1 8B Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", 8, 16, False
    ),
    ModelSpec(
        "Mistral 7B Instruct v0.3", "mistralai/Mistral-7B-Instruct-v0.3", 7, 14, False
    ),
    ModelSpec("Qwen2.5 7B Instruct", "Qwen/Qwen2.5-7B-Instruct", 7, 14, False),
    ModelSpec(
        "Phi-3 Mini Instruct", "microsoft/Phi-3-mini-4k-instruct", 3.8, 7.5, False
    ),
]

POTION_FRAMES = [
    r"""
          ___
         (___)
         <   >
          ) (
        _(   )_
       (_______)
    """,
    r"""
          ___
         (___)
         < . >
          ) (
        _(   )_
       (_______)
    """,
    r"""
          ___
         (___)
         <   >
          ) (
        _( . )_
       (_______)
    """,
]


def potion_art(frame_idx: int, width: int = 60) -> Panel:
    shades_g = ["#98ff98", "#00ff7f", "#66ff99"]
    shades_p = ["#cf9bff", "#b366ff", "#a64dff"]
    g = shades_g[frame_idx % len(shades_g)]
    p = shades_p[(frame_idx // 2) % len(shades_p)]
    left = "[bold " + g + "]" + POTION_FRAMES[frame_idx % 3] + "[/]"
    right = "[bold " + p + "]" + POTION_FRAMES[(frame_idx + 1) % 3] + "[/]"
    grid = Table.grid(padding=3)
    grid.add_row(Align.center(left), Align.center(right))
    return Panel(
        Align.center(grid),
        title="[bright_white]distillery[/] ",
        border_style="bright_black",
    )


def dropdown(label: str, options: List[str]) -> str:
    return questionary.select(label, choices=options).unsafe_ask()


def ask_path(label: str) -> Path:
    p = questionary.path(label).unsafe_ask()
    path = Path(p).expanduser()
    assert path.exists(), f"{label} missing"
    return path


def ascii_header() -> Panel:
    title = (
        "[bold bright_white]distillery[/] [bright_black]· agentic model distillation[/]"
    )
    sub = "[green]brewing[/] ➜ [magenta]distilling[/] ➜ [white]sampling[/]"
    return Panel(Align.center(f"{title}\n{sub}"), border_style="bright_black")


def format_model_option(m: ModelSpec) -> str:
    tag = "reasoning" if m.reasoning else "general"
    return f"{m.name} • {m.params_b}B • {m.size_gb}GB • {tag}"


def select_models() -> Tuple[ModelSpec, ModelSpec]:
    names = [format_model_option(m) for m in MODELS]
    big_label = dropdown("Choose big model", names)
    small_label = dropdown("Choose small model", names)
    big = MODELS[names.index(big_label)]
    small = MODELS[names.index(small_label)]
    return big, small


def load_text(p: Path) -> str:
    t = p.read_text(encoding="utf-8")
    assert isinstance(t, str)
    return t


def load_problems(p: Path) -> List[str]:
    lines = [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8").splitlines()]
    out = [ln.replace("\\n", "\n") for ln in lines if ln.strip() != ""]
    assert len(out) > 0
    return out


def batched(seq: List[Any], n: int) -> List[List[Any]]:
    assert n >= 1
    it = iter(seq)
    while True:
        b = list(itertools.islice(it, n))
        if not b:
            break
        yield b


def dataset_from_triples(system: str, qs: List[str], as_: List[str]) -> Dataset:
    records = []
    for q, a in zip(qs, as_):
        records.append({"system": system, "user": q, "assistant": a})
    return Dataset.from_list(records)


class RichCallback:
    def __init__(self, progress: Progress, task_id: Any):
        self.progress = progress
        self.task_id = task_id

    def on_step_end(self, args, state, control, **kwargs):
        self.progress.advance(self.task_id, 1)


modal_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "datasets", "peft", "accelerate"
)
modal_vol = modal.NetworkFileSystem.from_name("distillery-nfs", create_if_missing=True)


@app.function(gpu="A10G", timeout=60 * 60 * 12)
def modal_generate_batch(
    repo: str, system: str, batch: List[str], cfg_dict: Dict[str, Any]
) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        repo, device_map="auto", torch_dtype=torch.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    outs = []
    dev = next(model.parameters()).device
    for q in batch:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(dev)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                max_new_tokens=int(cfg_dict["max_new_tokens"]),
                do_sample=True,
                temperature=float(cfg_dict["temperature"]),
                top_p=float(cfg_dict["top_p"]),
                eos_token_id=eos,
                pad_token_id=pad,
            )
        outs.append(tok.decode(gen[0], skip_special_tokens=True))
    return outs


@app.function(
    gpu="A10G", network_file_systems={"/mnt/out": modal_vol}, timeout=60 * 60 * 24
)
def modal_finetune(
    repo: str,
    records: List[Dict[str, str]],
    out_name: str,
    cfg_dict: Dict[str, Any],
) -> str:
    import os, math, torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        repo, device_map="auto", torch_dtype=torch.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    data = Dataset.from_list(records)

    def tok_map(batch):
        msgs = [
            {"role": "system", "content": batch["system"]},
            {"role": "user", "content": batch["user"]},
            {"role": "assistant", "content": batch["assistant"]},
        ]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=False, return_tensors=None
        )
        out = tok(ids, truncation=True, max_length=2048)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = data.map(tok_map, remove_columns=data.column_names)
    dc = DataCollatorForLanguageModeling(tok, mlm=False)
    if bool(cfg_dict.get("use_lora", True)):
        lcfg = LoraConfig(
            r=int(cfg_dict["lora_r"]),
            lora_alpha=int(cfg_dict["lora_alpha"]),
            lora_dropout=float(cfg_dict["lora_dropout"]),
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lcfg)
    out_dir = f"/mnt/out/{out_name}"
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg_dict["train_bs"]),
        gradient_accumulation_steps=int(cfg_dict["gradient_accum"]),
        learning_rate=float(cfg_dict["lr"]),
        num_train_epochs=int(cfg_dict["epochs"]),
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        tokenizer=tok,
        args=args,
        data_collator=dc,
        train_dataset=tokenized,
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def brew_animation(stop_evt: threading.Event, subtitle: str):
    layout = Layout()
    layout.split_column(Layout(name="art", size=11), Layout(name="body"))
    frame = 0
    with Live(layout, refresh_per_second=12, transient=True):
        while not stop_evt.is_set():
            layout["art"].update(potion_art(frame))
            layout["body"].update(
                Panel(Align.center(subtitle), border_style="bright_black")
            )
            frame = (frame + 1) % 6
            time.sleep(0.08)


def run_brewing_ui(run_fn, subtitle: str):
    stop_evt = threading.Event()
    t = threading.Thread(target=brew_animation, args=(stop_evt, subtitle), daemon=True)
    t.start()
    try:
        return run_fn()
    finally:
        stop_evt.set()
        t.join()


def select_preview(examples: List[Tuple[str, str, str]]):
    table = Table(title="samples", box=box.SIMPLE_HEAVY)
    table.add_column("user")
    table.add_column("assistant")
    for u, a in random.sample(examples, k=min(5, len(examples))):
        table.add_row(
            u[:48] + "..." if len(u) > 48 else u, (a[:64] + "...") if len(a) > 64 else a
        )
    console.print(Panel(table, border_style="bright_black"))


@app.local_entrypoint()
def main():
    console.clear()
    console.print(ascii_header())
    big, small = select_models()
    sp = ask_path("Path to system prompt file")
    pp = ask_path("Path to problems file")
    out_dir = Path.cwd() / f"distillery_out/{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DistillConfig()
    system = load_text(sp)
    problems = load_problems(pp)
    selection_table = Table.grid(padding=1)
    selection_table.add_row(
        f"[bold]big[/] [white]{big.name}[/]",
        "[bold]small[/] [white]" + small.name + "[/]",
    )
    console.print(
        Panel(selection_table, title="selection", border_style="bright_black")
    )

    def do_brew():
        outs = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[green]brewing[/] via Modal"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as p:
            task = p.add_task("brew", total=len(problems))
            for batch in batched(problems, cfg.batch_size):
                res = modal_generate_batch.remote(big.repo, system, batch, cfg.__dict__)
                assert isinstance(res, list)
                outs.extend(res)
                p.advance(task, len(batch))
        return outs

    completions = run_brewing_ui(
        do_brew, "[green]brewing knowledge from the big model[/]"
    )
    triples = []
    for q, full in zip(problems, completions):
        if full.strip() == "":
            full = ""
        triples.append((q, full))
    ds = dataset_from_triples(system, [q for q, _ in triples], [a for _, a in triples])
    (out_dir / "brew.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"system": system, "user": q, "assistant": a})
                for q, a in triples
            ]
        ),
        encoding="utf-8",
    )
    console.print(Panel(f"[white]brewed {len(ds)} samples[/]", border_style="green"))

    def do_distill():
        recs = [
            {"system": r["system"], "user": r["user"], "assistant": r["assistant"]}
            for r in ds
        ]
        name = f"ft-{int(time.time())}"
        remote_path = modal_finetune.remote(small.repo, recs, name, cfg.__dict__)
        return f"modal://distillery-nfs/{name}"

    save_path = run_brewing_ui(
        do_distill, "[magenta]distilling into a smaller model[/]"
    )
    console.print(Panel(f"[white]output[/]: {save_path}", border_style="bright_black"))
    preview = dropdown("Preview generations with the finetuned model?", ["Yes", "No"])
    if preview == "Yes":
        sample_inputs = random.sample(problems, k=min(3, len(problems)))

        def run_preview():
            outs = []
            res = modal_generate_batch.remote(
                small.repo,
                system,
                sample_inputs,
                {"max_new_tokens": 1024, "temperature": 0.7, "top_p": 0.95},
            )
            outs.extend(res)
            return outs

        previews = run_brewing_ui(
            run_preview, "[white]sampling from the small model[/]"
        )
        pairs = list(zip(sample_inputs, previews))
        select_preview([(u, a, "") for u, a in pairs])
    console.print(
        Panel("[bold green]done[/] • happy distilling", border_style="bright_black")
    )


if __name__ == "__main__":
    main()
