# edge-compress

> **Work in progress.**

Building this to learn how ML models get made small and fast enough to run on a CPU.

Comparing three compression techniques, alone and combined:

- **Knowledge distillation:** train a small "student" model to copy a bigger "teacher" model.
- **INT8 quantization:** store weights and activations as 8-bit integers instead of 32-bit floats.
- **Structured pruning:** remove whole channels from the network so it does less work.