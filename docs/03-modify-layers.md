# Case 3: modify existing GGUF layers

## Goal

Measure how a controlled change to existing layer weights changes loading, memory, speed, and output quality. Preserve the number of blocks and the original downloads. Planning only in this task.

## Method

GGUF describes named tensors, their dimensions/types, and architecture metadata. A tensor change must respect the block encoding and alignment of its quantization type. Use a fresh output file and record which tensor elements changed. [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

Proposed first experiment: choose one small existing floating-point normalization tensor, then compare an exact no-change rewrite with a single documented scaling perturbation. This separates file-writing damage from an intentional weight change. A later selected quantized tensor experiment would dequantize only that tensor, apply a defined change, and requantize with a supported encoder. These are proposed controls, not validated implementations.

## Tools and versions

Python 3.12.13 and gguf 0.19.0 are installed. GGUFReader/GGUFWriter and llama.cpp's quantization utilities are candidate tools. Inference uses llama-cpp-python 0.3.36 and its vendored llama.cpp commit from case 1. Pin any separate standalone llama.cpp build to the same source revision before using it. [gguf-py tools](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)

## Feasibility on the 8 GB Air

Memory-mapped metadata inspection and one small tensor copy should fit more readily than loading the whole model as floats. This remains a budget to measure. A complete BF16 language weight copy is about 15.23 GB; full-model dequantization is outside the local budget. Keep one inference process and enough disk for the unchanged original plus a complete edited artifact. Rewriting a 3.81 GB file still requires substantial disk I/O.

## Concrete plan

1. Inventory original architecture metadata, tensor names, shapes, and quantization types without modifying them.
2. Choose the smallest meaningful target tensor and predeclare the operation and expected numerical range.
3. Write a no-change copy; validate tensor metadata and data hashes against the source. Confirm loading and output equivalence on a fixed smoke prompt.
4. Write one edited copy, preserving all unrelated tensor bytes and required metadata. Record source/output SHA256, exact operation, seed, and target tensor statistics.
5. Run the same evaluation protocol as case 1 with one artifact loaded at a time. Compare unchanged-copy controls before attributing effects to the perturbation.

## Risks and open questions

A model can load successfully while producing damaged text. Quantization scales and packing make byte edits unsafe. A no-change rewrite must preserve tokenizer and chat metadata too. Which weight to perturb, the magnitude, and encoder support for its actual quant type require inspection. This experiment does not substitute for meaningful training.

## Findings

Not run. Reserved: tensor inventory, no-change round-trip evidence, edit manifest, load errors, output differences, and memory/time table. No GGUF tensor has been edited.

## Sources consulted

The GGUF specification and gguf-py project linked above were consulted on 2026-11-11. The inference architecture reference is [llama.cpp Qwen2-VL](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2vl.cpp). Proposed experimental controls are project plans, not claims of upstream support.
