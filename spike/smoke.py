"""Load each model on MPS, check it talks, and time prefill/decode."""

import time

import torch

from common import DEVICE, MODELS, load, short


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def main():
    print("model,prefill_tok_s,decode_tok_s,peak_mem_gb")
    for model_id in MODELS:
        model, tok = load(model_id)
        msgs = [{"role": "user", "content": "Say hello in five words."}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(DEVICE) for k, v in ids.items()}
        out = model.generate(**ids, max_new_tokens=20, do_sample=False)
        reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"# {short(model_id)} reply: {reply.strip()!r}")

        long = torch.randint(1000, 20000, (1, 1024), device=DEVICE)
        model(long)  # warm-up
        sync(); t = time.perf_counter(); model(long); sync()
        prefill = 1024 / (time.perf_counter() - t)

        prompt = long[:, :64]
        sync(); t = time.perf_counter()
        model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=128, min_new_tokens=128, do_sample=False)
        sync()
        decode = 128 / (time.perf_counter() - t)
        mem = torch.mps.driver_allocated_memory() / 1e9 if DEVICE == "mps" else float("nan")
        print(f"{short(model_id)},{prefill:.0f},{decode:.1f},{mem:.1f}")
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
