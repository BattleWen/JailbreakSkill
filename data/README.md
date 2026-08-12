# Benchmark data

> [!WARNING]
> These files contain harmful, illegal, or otherwise sensitive requests for
> controlled AI-safety evaluation. Do not execute them against systems without
> authorization and appropriate safeguards.

## Included files

| File | Rows | Provenance and transformation |
| --- | ---: | --- |
| `AdvBench.jsonl` | 520 | Queries exactly match the `goal` column of the official AdvBench harmful-behaviors CSV. `risk_category` is a project annotation. |
| `HarmBench.jsonl` | 400 | Project-normalized JSONL representation of the official HarmBench text behaviors. Original source fields are retained; contextual rows combine `ContextString` and `Behavior` into `query`. |
| `JBB-Behaviors-original.jsonl` | 55 | Rows whose official JBB-Behaviors `Source` is `Original`. `query` is the official `Goal`; source index, behavior label, and category are retained. |

The JSONL schema always provides a model-facing `query` field. Additional
metadata is retained where available so runs can be filtered and audited.

## Integrity

```text
d23eb1d36ed705bdfa3892cb4e949208115ee7dfde6143ff05c3c87f0c36503c  AdvBench.jsonl
07195abeb046e28526fee6add02d5fe50059b303dd9d235251b8c722a01ca87f  HarmBench.jsonl
1b0cd54cdb4c08e89feb3c8dfd35f0549be69732465b3d15dc93b9485a170ee8  JBB-Behaviors-original.jsonl
```

## Sources and licenses

- AdvBench: [official `llm-attacks` repository](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench) and [paper](https://arxiv.org/abs/2307.15043). The upstream repository is MIT licensed; see `licenses/AdvBench-LICENSE.txt`.
- HarmBench: [official repository](https://github.com/centerforaisafety/HarmBench) and [paper](https://arxiv.org/abs/2402.04249). The upstream repository is MIT licensed; see `licenses/HarmBench-LICENSE.txt`.
- JBB-Behaviors: [official dataset](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) and [paper](https://arxiv.org/abs/2404.01318). The dataset is MIT licensed; see `licenses/JailbreakBench-LICENSE.txt`.

Please cite the corresponding papers when using these benchmarks. The bundled
license notices must be preserved when redistributing the dataset files.
