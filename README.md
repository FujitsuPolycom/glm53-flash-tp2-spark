# GLM-5.3-Flash-NVFP4 on a DGX Spark GB10 pair

Serves `local-inference-lab/GLM-5.3-Flash-NVFP4` across two GB10 nodes as one
tensor-parallel-2 vLLM instance with multi-token-prediction speculative
decoding. Weights occupy 92.72 GiB per node.

## Baseline image

```
sparkring-glm53-official-spark:flashkda-rdma-v3
```

Required on both nodes, along with the checkpoint at an identical path.

## Patches

Four files, 64 changed lines total. Without them the engine fails at load or
at request time. Copy the files out of the image, patch the copies, and bind
mount them back over the originals; the image is unmodified.

| Patch | Fixes |
|---|---|
| `model.patch` | Checkpoint names hyper-connection parameters as submodules (`attn_hc.base`) and fuses the linear-attention q/k/v convolutions into one `[3C,1,4]` tensor; the image expects flat names and three convolutions. |
| `modelopt.patch` | Draft model resolves its quantization in a different module namespace than the target rewrote the shared config into, so its experts load unquantized. See [MTP](#mtp). |
| `sparse_attn_indexer.patch`, `sparse_attn_indexer_kpool.patch` | `persistent_topk` requests 62 thread blocks against 48 available on GB10, and its fallback needs 128 KB shared memory per block where GB10 has 99 KB. Disables it on compute capability 12.x in favour of `top_k_per_row_decode`. |

```bash
PATCHDIR=/var/tmp/glm53-nvfp4-patch
IMAGE=sparkring-glm53-official-spark:flashkda-rdma-v3
B=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p "$PATCHDIR"

cid=$(docker create "$IMAGE")
docker cp "$cid:$B/models/glm5next/nvidia/model.py"                    "$PATCHDIR/model.py"
docker cp "$cid:$B/model_executor/layers/quantization/modelopt.py"     "$PATCHDIR/modelopt.py"
docker cp "$cid:$B/model_executor/layers/sparse_attn_indexer.py"       "$PATCHDIR/sparse_attn_indexer.py"
docker cp "$cid:$B/model_executor/layers/sparse_attn_indexer_kpool.py" "$PATCHDIR/sparse_attn_indexer_kpool.py"
docker rm "$cid"

for f in model modelopt sparse_attn_indexer sparse_attn_indexer_kpool; do
  patch "$PATCHDIR/$f.py" < "patches/$f.patch"
done
python3 -m py_compile "$PATCHDIR"/*.py
```

Run this on both nodes. `scripts/launch_tp2.sh` carries the matching mounts.

## MTP

The checkpoint declares `MIXED_PRECISION`: NVFP4 for routed experts in layers
3 through 44, MXFP8 for the multi-token-prediction layer 45.

vLLM builds the draft model against the target model's quantization config
object. By then the target's weight mapper has rewritten that object's layer
keys into the target module namespace (`language_model.model.layers.N...`).
The draft has no mapper of its own and queries `model.layers.45...`, which
matches no key, so its experts instantiate unquantized and the MXFP8 scale
parameters are never created. Loading the checkpoint's scale tensor then fails:

```
KeyError: model.layers.45.mtp_block.mlp.experts.routed_experts.w2_weight_scale
```

`modelopt.patch` widens the prefix-candidate generator in
`ModelOptMixedPrecisionConfig._quantized_layer_prefix_candidates`: it strips
the draft's `.mtp_block.` segment and re-roots each candidate across
`model.`, `model.language_model.`, and `language_model.model.`. The draft then
matches the config's keys whether or not the target's mapper has rewritten
them, which makes the lookup independent of the order in which the target and
draft models are built.

Confirm it took, in the engine log:

```
[quantprobe] layer=RoutedExperts prefix=model.layers.45.mlp.experts algo=MXFP8
```

That line comes from a logging statement carried in the same patch. Delete the
statement if unwanted; it has no other effect. `algo=None` means the fix is not
active and load will fail.

## Launch config

`scripts/launch_tp2.sh` takes a node rank and that node's own address. Start
rank 1 first; rank 0 hosts the API on port 8000.

```bash
/var/tmp/launch_tp2.sh 1 <rank1-address>   # rank-1 node
/var/tmp/launch_tp2.sh 0 <rank0-address>   # rank-0 node
```

Edit the assignments at the top of the script for image tag, checkpoint path,
patch directory, and master address. Flags that are load-bearing rather than
tuning choices:

| Flag | Value | Reason |
|---|---|---|
| `--tensor-parallel-size` | 2 | Weights do not fit one node. |
| `--kv-cache-dtype` | `fp8` | The sparse-MLA path implements only the packed `fp8_ds_mla` layout. |
| `--kv-cache-memory-bytes` | `10737418240` | Pins the KV cache and overrides `--gpu-memory-utilization`. At 12 GiB, free memory falls below 1 GB during warmup and compilation runs at swap speed. |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` | Requires the MTP fix above. |
| `--max-model-len` | `524288` | 10 GiB of KV yields ~1.16M tokens, 2.22 concurrent requests at this length. |
| `--max-num-seqs` | `8` | Requests beyond this queue; aggregate throughput is flat above it. |
| `--load-format` | `instanttensor` | Optional. 184 GB in ~30 s against ~700 s sequential. Omit if the package is absent. |

`instanttensor` is source-only on PyPI and compiles on arm64. Install with
`--no-deps`; dependency resolution downgrades the image's NCCL from 2.30.7 to
2.29.7.

```dockerfile
FROM sparkring-glm53-official-spark:flashkda-rdma-v3
ENV MAX_JOBS=8
RUN pip install --no-deps --no-build-isolation instanttensor==0.1.9 && \
    python3 -c "import instanttensor"
```

## Requests

The model emits reasoning tokens before its answer. The chat template accepts
`reasoning_effort` of `low` or `high`; any other value, including omission,
selects maximum effort. Short `max_tokens` with high effort returns
`finish_reason: "length"` and empty content.

```bash
curl -s http://<rank0-address>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-nvfp4-tp2-mtp3",
       "messages":[{"role":"user","content":"What is 17 * 23? Show the arithmetic."}],
       "temperature":1,"max_tokens":1024,
       "chat_template_kwargs":{"reasoning_effort":"low"}}'
```
