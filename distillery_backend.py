import os, io, json, asyncio, uuid, tarfile, time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import modal

# ---- Image / deps -----------------------------------------------------------
# Added: peft + safetensors for LoRA export. Keep the rest as-is.
vimage = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "fastapi==0.112.2",
        "uvicorn==0.30.6",
        "transformers>=4.43.0",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "huggingface_hub[hf_transfer]>=0.24.5",
        "torch>=2.3.0",
        "peft>=0.11.1",
        "safetensors>=0.4.3",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # You can uncomment the next line if you want to suppress some PyTorch kernel warnings.
            # "PYTORCH_DISABLE_KERNEL_WARNINGS": "1",
        }
    )
)

app = modal.App("distillery-backend")
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("distillery-outputs", create_if_missing=True)


# ---- Generation params ------------------------------------------------------
@dataclass
class GenParams:
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@app.function(
    image=vimage,
    gpu="A100-80GB:1",  # same resources as before
    volumes={"/root/.cache/huggingface": hf_cache, "/outputs": out_vol},
    timeout=60 * 60,
)
@modal.asgi_app()
def serve():
    import torch
    import torch.nn as nn
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model

    app = FastAPI()
    MODELS: Dict[str, Tuple[Any, Any]] = {}  # cache for inference-only
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Helpers ------------------------------------------------------------
    def ensure_dir(p: str) -> None:
        os.makedirs(p, exist_ok=True)
        assert os.path.isdir(p), f"Expected directory to exist: {p}"

    def load_model_for_inference(name: str) -> Tuple[Any, Any]:
        """
        Loads and caches a model+tokenizer for inference (generation).
        Uses device_map='auto' to maximize GPU utilization during generation.
        """
        assert isinstance(name, str) and len(name) > 0
        if name in MODELS:
            return MODELS[name]

        tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        if tok.pad_token is None:
            # Avoid padding issues during batching
            tok.pad_token = tok.eos_token or tok.unk_token
        MODELS[name] = (mdl, tok)
        return MODELS[name]

    def load_model_for_training(name: str) -> Tuple[Any, Any]:
        """
        Loads a *fresh* model+tokenizer for training (no caching shared with inference).
        No device_map (Trainer/Accelerate will manage device placement).
        """
        tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=None,  # critical: let Trainer/Accelerate control placement
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        return mdl, tok

    def apply_chat(tok, system: str, user: str) -> str:
        assert isinstance(system, str) and isinstance(user, str)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        # Fallback generic chat prompt:
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    def guess_lora_targets(model: nn.Module) -> List[str]:
        """
        Heuristic to pick LoRA target module names across a variety of decoder-only LMs.
        We prefer attention + MLP projection layers when available. If none detected,
        we fall back to all linear leaf module names (safe but broader).
        """
        preferred = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # LLaMA-style attn
            "gate_proj",
            "up_proj",
            "down_proj",  # LLaMA MLP
            "c_attn",
            "c_proj",  # GPT-2 style
            "out_proj",
            "fc_in",
            "fc_out",  # Common patterns
            "Wqkv",
            "Wq",
            "Wk",
            "Wv",
            "Wo",
            "Wup",
            "Wdown",  # MPT/GPT-NeoX variants
        }

        leaf_linear_names = set()
        preferred_hits = set()

        linear_like = (nn.Linear,)
        # Some models wrap linear in quantized/bnb layers; match by class name string too
        linear_like_names = {"Linear8bitLt", "Linear4bit"}

        for name, module in model.named_modules():
            leaf = name.split(".")[-1]
            cls_name = module.__class__.__name__
            is_linear = isinstance(module, linear_like) or (
                cls_name in linear_like_names
            )
            if not is_linear:
                continue
            leaf_linear_names.add(leaf)
            if (leaf in preferred) or any(key in leaf for key in preferred):
                preferred_hits.add(leaf)

        if preferred_hits:
            return sorted(preferred_hits)
        # Fallback: all leaf linear names (safer than failing)
        return sorted(leaf_linear_names)

    # ---- Routes -------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok", "time": _now()}

    @app.post("/v1/brew")
    async def brew(request: Request):
        body = await request.json()
        assert isinstance(body, dict)
        model = body.get("model")
        system = body.get("system")
        problems = body.get("problems")
        params = body.get("params", {})
        assert (
            isinstance(model, str)
            and isinstance(system, str)
            and isinstance(problems, list)
        ), "model/system/problems are required"
        for p in problems:
            assert isinstance(p, str), "each problem must be a string"

        gp = GenParams(
            **{
                k: v
                for k, v in params.items()
                if k in {"max_new_tokens", "temperature", "top_p"}
            }
        )

        async def stream():
            # Flush headers + first chunk immediately so client doesn't time out
            yield f"data: {json.dumps({'type':'begin','total':len(problems),'time':_now()})}\n\n"

            # Heavy work happens after the first yield
            loop = asyncio.get_event_loop()
            mdl, tok = await loop.run_in_executor(None, load_model_for_inference, model)
            eos_id = tok.eos_token_id or tok.pad_token_id
            assert eos_id is not None, "Tokenizer must have an eos/pad token id"
            bs = min(8, max(1, int(body.get("batch_size", 4))))
            idx = 0
            while idx < len(problems):
                batch = problems[idx : idx + bs]
                prompts = [apply_chat(tok, system, q) for q in batch]
                enc = tok(prompts, return_tensors="pt", padding=True).to(mdl.device)
                mdl.eval()
                with torch.no_grad():
                    out = mdl.generate(
                        **enc,
                        do_sample=True,
                        temperature=gp.temperature,
                        top_p=gp.top_p,
                        max_new_tokens=gp.max_new_tokens,
                        eos_token_id=eos_id,
                        pad_token_id=tok.pad_token_id,
                    )
                for j, seq in enumerate(out):
                    in_len = enc["input_ids"][j].shape[-1]
                    gen = seq[in_len:]
                    text = tok.decode(gen, skip_special_tokens=True)
                    payload = {
                        "type": "sample",
                        "index": idx + j,
                        "problem": batch[j],
                        "completion": text,
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                idx += bs
            yield f"data: {json.dumps({'type':'done','time':_now()})}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/finetune")
    async def finetune(request: Request):
        """
        Train a LoRA (rank=64) adapter and export only the adapter files.
        Request body:
          {
            "model": "repo/name",
            "examples": [{"system": "...", "question": "...", "answer": "..."}, ...],
            "train_args": { optional overrides: "epochs", "batch_size", "grad_accum", "lr",
                            "warmup", "scheduler", "logging_steps", "max_length" }
            "job_id": "optional-string"
          }
        """
        body = await request.json()
        assert isinstance(body, dict)
        model = body.get("model")
        triples = body.get("examples")
        train_args = body.get("train_args", {})
        assert (
            isinstance(model, str) and isinstance(triples, list) and len(triples) > 0
        ), "'model' and non-empty 'examples' are required"
        for t in triples:
            assert (
                isinstance(t, dict)
                and "system" in t
                and "question" in t
                and "answer" in t
            ), "each example must include system/question/answer"
            assert (
                isinstance(t["system"], str)
                and isinstance(t["question"], str)
                and isinstance(t["answer"], str)
            ), "system/question/answer must be strings"

        job_id = body.get("job_id") or uuid.uuid4().hex
        out_dir = f"./outputs/{job_id}"
        ensure_dir("./outputs")
        ensure_dir(out_dir)

        # Simple debug breadcrumbs (kept but quieter)
        print("PWD:", os.getcwd())
        print("ROOT LS:", os.listdir("/"))
        print("OUT LS:", os.listdir("./outputs"))

        q: asyncio.Queue = asyncio.Queue()

        class Cb:
            def on_log(self, logs: Dict[str, Any]):
                logs["time"] = _now()
                # Non-blocking put into SSE queue
                asyncio.run(q.put({"type": "log", "logs": logs}))

            def on_train_end(self, ok: bool, error: Optional[str] = None):
                payload: Dict[str, Any] = {"type": "end", "time": _now(), "ok": ok}
                if error:
                    payload["error"] = error
                asyncio.run(q.put(payload))

        cb = Cb()

        def run_train():
            try:
                # ---- Load base model/tokenizer for TRAINING -----------------
                base_model, tok = load_model_for_training(model)

                # Optional but often helpful for LoRA
                if hasattr(base_model, "enable_input_require_grads"):
                    base_model.enable_input_require_grads()

                # ---- Build LoRA config --------------------------------------
                target_modules = guess_lora_targets(base_model)
                assert len(target_modules) > 0, "Failed to detect LoRA target modules"

                lora_cfg = LoraConfig(
                    r=64,  # rank=64 as requested
                    lora_alpha=128,  # a common, stable choice
                    lora_dropout=0.05,  # small dropout for stability
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=target_modules,
                )

                peft_model = get_peft_model(base_model, lora_cfg)
                # Print trainable params for sanity
                try:
                    peft_model.print_trainable_parameters()
                except Exception:
                    pass

                eos_id = tok.eos_token_id or tok.pad_token_id
                assert eos_id is not None, "Tokenizer must have an eos/pad token id"

                # ---- Prepare dataset ----------------------------------------
                def to_text(ex):
                    s = apply_chat(tok, ex["system"], ex["question"])
                    a = ex["answer"].rstrip() + (tok.eos_token or "")
                    return s + a

                texts = [to_text(x) for x in triples]
                ds = Dataset.from_dict({"text": texts})

                def tok_map(ex):
                    enc = tok(
                        ex["text"],
                        truncation=True,
                        padding="max_length",
                        max_length=int(train_args.get("max_length", 2048)),
                    )
                    # standard causal LM labels = input_ids
                    enc["labels"] = enc["input_ids"].copy()
                    return enc

                ds = ds.map(tok_map, remove_columns=["text"], batched=False)
                dc = DataCollatorForLanguageModeling(tok, mlm=False)

                # ---- Training args ------------------------------------------
                # IMPORTANT: fully disable checkpoint saving to avoid creating checkpoint-1
                args = TrainingArguments(
                    output_dir=out_dir,
                    num_train_epochs=float(train_args.get("epochs", 1.0)),
                    per_device_train_batch_size=int(train_args.get("batch_size", 2)),
                    gradient_accumulation_steps=int(train_args.get("grad_accum", 8)),
                    learning_rate=float(train_args.get("lr", 2e-5)),
                    warmup_ratio=float(train_args.get("warmup", 0.03)),
                    lr_scheduler_type=str(train_args.get("scheduler", "cosine")),
                    logging_steps=int(train_args.get("logging_steps", 10)),
                    save_strategy="no",  # <--- critical: no intermediate saves
                    report_to=[],
                    remove_unused_columns=False,  # safer for PEFT
                    bf16=torch.cuda.is_available(),  # keep fp16 False for stability with bfloat16
                    fp16=False,
                    save_total_limit=0,
                    save_safetensors=True,
                    dataloader_pin_memory=False,
                )

                class BridgeTrainer(Trainer):
                    # Fix FutureWarning by using 'processing_class' instead of 'tokenizer'
                    def __init__(self, *a, **k):
                        if "tokenizer" in k and "processing_class" not in k:
                            k["processing_class"] = k.pop("tokenizer")
                        super().__init__(*a, **k)

                    def log(self, logs):
                        super().log(logs)
                        cb.on_log(logs)

                trainer = BridgeTrainer(
                    model=peft_model,
                    args=args,
                    train_dataset=ds,
                    data_collator=dc,
                    processing_class=tok,  # avoids the tokenizer deprecation warning
                )

                trainer.train()

                # ---- Save ONLY the LoRA adapter + tokenizer -----------------
                # The PEFT save_pretrained writes adapter_config.json + adapter_model.(safetensors|bin)
                peft_model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)

                # Assertions to guarantee artifacts exist before we announce success
                adapter_cfg = os.path.join(out_dir, "adapter_config.json")
                adapter_pt_sft = os.path.join(out_dir, "adapter_model.safetensors")
                adapter_pt_bin = os.path.join(out_dir, "adapter_model.bin")

                assert os.path.exists(adapter_cfg), "Missing adapter_config.json"
                assert os.path.exists(adapter_pt_sft) or os.path.exists(
                    adapter_pt_bin
                ), "Missing adapter model weights"

                # Pack artifacts
                tar_path = f"./outputs/{job_id}.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(out_dir, arcname=os.path.basename(out_dir))

                # Final record for convenience
                with open(f"./outputs/{job_id}.json", "w") as f:
                    json.dump(
                        {
                            "job_id": job_id,
                            "artifact": tar_path,
                            "adapter_config": "adapter_config.json",
                            "adapter_weights": (
                                "adapter_model.safetensors"
                                if os.path.exists(adapter_pt_sft)
                                else "adapter_model.bin"
                            ),
                            "tokenizer_files_included": True,
                        },
                        f,
                    )

                cb.on_train_end(ok=True)

            except Exception as e:
                # Surface error via stream + keep process alive
                print("TRAINING ERROR:", repr(e))
                cb.on_train_end(ok=False, error=str(e))

        async def stream():
            # Flush an initial event immediately
            yield f"data: {json.dumps({'type':'begin','job_id':job_id,'time':_now()})}\n\n"
            # Kick off training in the background only after streaming begins
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, run_train)
            while True:
                msg = await q.get()
                if msg.get("type") == "end":
                    final = {
                        "type": "done",
                        "job_id": job_id,
                        "ok": msg.get("ok", False),
                        "error": msg.get("error"),
                        "download": f"/download/{job_id}" if msg.get("ok") else None,
                        "time": _now(),
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    break
                yield f"data: {json.dumps(msg)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/download/{job_id}")
    async def download(job_id: str):
        assert isinstance(job_id, str) and len(job_id) > 0
        path = f"./outputs/{job_id}.tar.gz"
        if not os.path.exists(path):
            return JSONResponse({"error": "not_found"}, status_code=404)
        # Small assert that the tar isn't empty/corrupt (directory should also exist)
        out_dir = f"./outputs/{job_id}"
        assert os.path.isdir(out_dir), f"Expected output dir to exist: {out_dir}"
        return FileResponse(
            path, filename=f"{job_id}.tar.gz", media_type="application/gzip"
        )

    return app
