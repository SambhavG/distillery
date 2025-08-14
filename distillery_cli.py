#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Distillery (pretty CLI)
# - Same functionality as your original script
# - Nicer visuals using Rich: spinners, progress bars, tables, panels
#
# Requirements:
#   pip install aiohttp questionary rich
#
# Tip: run with a reasonably wide terminal for best results.

import os
import json
import asyncio
import argparse
import aiohttp
import questionary
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from textwrap import shorten

# --- Rich UI imports ---
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as rich_traceback_install
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    DownloadColumn,
    TransferSpeedColumn,
)

# Better tracebacks
rich_traceback_install(show_locals=False)
console = Console()

# --------------------------------------------------------------------------------------
# Data types and model lists
# --------------------------------------------------------------------------------------


@dataclass
class ModelCard:
    name: str
    params_b: float
    size_gb: float
    reasoning: bool


MODELS_BIG = [
    ModelCard("Qwen/Qwen2.5-14B-Instruct", 14, 28.0, False),
    ModelCard("mistralai/Mixtral-8x7B-Instruct-v0.1", 46, 90.0, True),
    ModelCard("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 32, 64.0, True),
    ModelCard("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 8, 16.0, True),
]
MODELS_SMALL = [
    ModelCard("Qwen/Qwen2.5-0.5B-Instruct", 0.5, 1.0, False),
    ModelCard("meta-llama/Meta-Llama-3.2-3B-Instruct", 3, 6.2, False),
    ModelCard("microsoft/Phi-3.5-mini-instruct", 3, 5.5, False),
]

# --------------------------------------------------------------------------------------
# Prompts (questionary)
# --------------------------------------------------------------------------------------


async def choose_model(cards: List[ModelCard], title: str) -> ModelCard:
    choices = [
        questionary.Choice(
            f"{c.name} — {c.params_b}B — {c.size_gb}GB — {'reasoning' if c.reasoning else 'vanilla'}",
            value=c,
        )
        for c in cards
    ]
    ans = await questionary.select(title, choices=choices).ask_async()
    if not isinstance(ans, ModelCard):
        raise RuntimeError("Selection cancelled.")
    return ans


async def ask_path(prompt: str) -> str:
    p = await questionary.path(prompt).ask_async()
    if not isinstance(p, str):
        raise RuntimeError("Path entry cancelled.")
    return p


async def ask_confirm(prompt: str) -> bool:
    v = await questionary.confirm(prompt).ask_async()
    if not isinstance(v, bool):
        raise RuntimeError("Confirmation cancelled.")
    return v


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

SSE_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)


async def sse_post(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any]):
    """Yield JSON events from a text/event-stream endpoint."""
    async with session.post(
        url,
        json=payload,
        headers={"Accept": "text/event-stream"},
        timeout=SSE_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        buffer = b""
        async for chunk in resp.content.iter_chunked(1024):
            buffer += chunk
            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)
                for line in block.split(b"\n"):
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        yield json.loads(data.decode("utf-8"))
                    except json.JSONDecodeError:
                        # Ignore malformed lines; continue gracefully
                        continue


def read_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    if not isinstance(s, str):
        raise ValueError("File content is not a string.")
    return s.replace("\\n", "\n")


def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    lines = [x for x in lines if x.strip() != ""]
    if not all(isinstance(x, str) for x in lines):
        raise ValueError("Some lines are not strings.")
    return [x.replace("\\n", "\n") for x in lines]


def show_banner() -> None:
    console.print(
        Panel.fit(
            "[b cyan]distillery[/b cyan] [dim]— CLI[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.rule(style="dim")


def show_model_summary(big: ModelCard, small: ModelCard) -> None:
    tbl = Table(box=box.SIMPLE_HEAVY, show_header=True)
    tbl.add_column("Role", style="bold")
    tbl.add_column("Model")
    tbl.add_column("Params")
    tbl.add_column("Size")
    tbl.add_column("Type")
    tbl.add_row(
        "Big (brew)",
        big.name,
        f"{big.params_b}B",
        f"{big.size_gb} GB",
        "reasoning" if big.reasoning else "vanilla",
    )
    tbl.add_row(
        "Small (distill)",
        small.name,
        f"{small.params_b}B",
        f"{small.size_gb} GB",
        "reasoning" if small.reasoning else "vanilla",
    )
    console.print(tbl)
    console.rule(style="dim")


def show_samples(rows: List[Dict[str, Any]], k: int = 6) -> None:
    take = [r for r in rows if r][:k]
    if not take:
        console.print(Panel.fit("[dim](no samples to show)[/dim]", border_style="dim"))
        return

    tbl = Table(
        title="Samples",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        header_style="bold",
        expand=True,
    )
    tbl.add_column("Index", justify="right", style="cyan", no_wrap=True)
    tbl.add_column("Question", style="white")
    tbl.add_column("Answer (first 200 chars)", style="green")

    for r in take:
        idx = str(r.get("index", "?"))
        prob = shorten(
            (r.get("problem") or "").replace("\n", " "), width=80, placeholder="…"
        )
        comp = shorten(
            (r.get("completion") or "").replace("\n", " "), width=200, placeholder="…"
        )
        tbl.add_row(idx, prob, comp)

    console.print(tbl)
    console.rule(style="dim")


# --------------------------------------------------------------------------------------
# Core phases with pretty progress bars
# --------------------------------------------------------------------------------------


async def brew_phase(
    base_url: str, model: str, system: str, problems: List[str]
) -> List[Dict[str, Any]]:
    total = len(problems)
    results: List[Optional[Dict[str, Any]]] = [None] * total

    console.print(
        Panel.fit(
            f"[bold]Brewing[/] with [cyan]{model}[/] on [magenta]{total}[/] problems",
            border_style="cyan",
        )
    )

    # Fancy progress bar
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]Brewing[/]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("brew", total=total)
        async with aiohttp.ClientSession(
            base_url=base_url, timeout=SSE_TIMEOUT
        ) as session:
            async for evt in sse_post(
                session,
                "/v1/brew",
                {"model": model, "system": system, "problems": problems},
            ):
                t = evt.get("type")
                if t == "sample":
                    i = evt.get("index")
                    if isinstance(i, int) and 0 <= i < total:
                        results[i] = evt
                        progress.update(
                            task,
                            advance=1,
                            description=f"[bold cyan]Brewing[/] (sample {i + 1}/{total})",
                        )
                elif t == "error":
                    msg = evt.get("message", "Unknown error")
                    raise RuntimeError(f"Brew error: {msg}")

    if not all(r is not None for r in results):
        raise RuntimeError("Missing some brew results")
    console.print("[green]✓ Brewing complete.[/green]")
    console.rule(style="dim")
    return results  # type: ignore


async def finetune_phase(
    base_url: str, model: str, triples: List[Dict[str, str]]
) -> str:
    console.print(
        Panel.fit(
            f"[bold]Distilling[/] into [cyan]{model}[/] from [magenta]{len(triples)}[/] examples",
            border_style="cyan",
        )
    )
    route: Optional[str] = None
    steps = 0

    # Indeterminate spinner with live step count
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Distilling...", total=None)
        async with aiohttp.ClientSession(
            base_url=base_url, timeout=SSE_TIMEOUT
        ) as session:
            async for evt in sse_post(
                session, "/v1/finetune", {"model": model, "examples": triples}
            ):
                t = evt.get("type")
                if t == "log":
                    steps += 1
                    if steps % 10 == 0:
                        progress.update(
                            task, description=f"Distilling... [bold]{steps}[/] steps"
                        )
                elif t == "done":
                    route = evt.get("download")
                    break
                elif t == "error":
                    msg = evt.get("message", "Unknown error")
                    raise RuntimeError(f"Finetune error: {msg}")

    if not isinstance(route, str) or not route:
        raise RuntimeError("Finetune did not return a download route")
    console.print(
        f"[green]✓ Distillation complete.[/green] Download route: [cyan]{route}[/cyan]"
    )
    console.rule(style="dim")
    return route


async def download_artifact(base_url: str, route: str, out_path: str):
    console.print(
        Panel.fit(
            f"[bold]Downloading[/] artifact → [cyan]{out_path}[/cyan]",
            border_style="cyan",
        )
    )

    async with aiohttp.ClientSession(base_url=base_url, timeout=SSE_TIMEOUT) as session:
        async with session.get(route, timeout=SSE_TIMEOUT) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Download failed: HTTP {resp.status}")

            total_bytes: Optional[int] = None
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                total_bytes = int(cl)

            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            with Progress(
                TextColumn("[bold green]Downloading[/]"),
                BarColumn(bar_width=None),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("download", total=total_bytes)
                with open(out_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1 << 20):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

    console.print(f"[green]✓ Downloaded to[/green] [cyan]{out_path}[/cyan]")
    console.rule(style="dim")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


async def main_async():
    parser = argparse.ArgumentParser(prog="distillery")
    parser.add_argument(
        "--backend", required=True, help="Base URL, e.g. http://localhost:8000"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.getcwd(), "distilled_model.tar.gz"),
        help="Output file path for the distilled model archive",
    )
    args = parser.parse_args()

    show_banner()

    # Model selection
    big = await choose_model(MODELS_BIG, "Choose big model")
    small = await choose_model(MODELS_SMALL, "Choose small model")
    show_model_summary(big, small)

    # Paths and data
    sys_path = await ask_path("Path to system prompt")
    prob_path = await ask_path("Path to problems list")

    with console.status("[bold]Reading input files...[/bold]", spinner="dots"):
        system = read_file(sys_path)
        problems = read_lines(prob_path)

    console.print(f"[bold]System prompt:[/bold] {len(system)} chars")
    console.print(f"[bold]Problems:[/bold] {len(problems)} items")
    console.rule(style="dim")

    # Brew
    rows = await brew_phase(args.backend, big.name, system, problems)
    show_samples(rows, k=min(8, len(rows)))

    # Distill
    triples = [
        {"system": system, "question": r["problem"], "answer": r["completion"]}
        for r in rows
    ]
    route = await finetune_phase(args.backend, small.name, triples)

    # Download
    console.print(
        Panel.fit(
            f"[bold]Preparing download[/] [dim]{args.backend}{route}[/dim] → [cyan]{args.out}[/cyan]",
            border_style="cyan",
        )
    )
    await download_artifact(args.backend, route, args.out)

    # Unpack
    console.print(Panel.fit("[bold]Unpacking artifact[/]", border_style="cyan"))
    import tarfile

    try:
        # We show a spinner while extracting (tarfile doesn't provide progress callbacks)
        with console.status("[bold]Extracting...[/bold]", spinner="earth"):
            with tarfile.open(args.out, "r:gz") as tar:
                tar.extractall(filter="data")
        console.print("[green]✓ Extraction complete.[/green]")
    finally:
        # Clean up the .tar.gz to match original behavior
        if os.path.exists(args.out):
            os.remove(args.out)

    console.rule(style="dim")
    console.print(
        Panel.fit("[bold green]All done![/bold green] ✨", border_style="green")
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
