# few-B pre-SFT comparison

> Weak-label agreement is not human accuracy. Invalid JSON is treated as abstention.

| Model | Params (B) | VRAM (GiB) | JSON coverage | P50 (s) | P95 (s) | video-h/wall-h | weak recall | weak F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-VL-2B-Instruct | 2.128 | 3.96 | 1.00 | 1.124 | 1.665 | 5.13 | 0.000 | NA |
| Qwen/Qwen3-VL-4B-Instruct | 4.438 | 8.27 | 0.35 | 3.152 | 3.347 | 2.58 | 0.000 | NA |
| Qwen/Qwen3-VL-8B-Instruct | 8.767 | 16.33 | 1.00 | 1.311 | 1.574 | 4.85 | 0.154 | 0.222 |
