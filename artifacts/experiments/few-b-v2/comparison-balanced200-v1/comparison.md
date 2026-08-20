# few-B pre-SFT comparison

> Weak-label agreement is not human accuracy. Invalid JSON is treated as abstention.

| Model | Params (B) | VRAM (GiB) | JSON coverage | P50 (s) | P95 (s) | video-h/wall-h | weak recall | weak F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-VL-2B-Instruct | 2.128 | 3.96 | 1.00 | 1.046 | 1.224 | 5.60 | 0.000 | NA |
| Qwen/Qwen3-VL-4B-Instruct | 4.438 | 8.27 | 0.51 | 1.826 | 3.314 | 2.79 | 0.000 | NA |
| Qwen/Qwen3-VL-8B-Instruct | 8.767 | 16.33 | 1.00 | 1.314 | 1.483 | 5.13 | 0.132 | 0.189 |
