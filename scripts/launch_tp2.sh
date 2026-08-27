#!/usr/bin/env bash
# Serve GLM-5.3-Flash-NVFP4 across the two-node DGX Spark pair as TP2.
#
# Derived from the ring's four-node GLM-5.3 launch: same image, same GB10
# (sm_121) build environment, same NCCL layout. Distributed bootstrap runs over
# the management interface (enP7s7) while collective data moves over RDMA, which
# is the arrangement already proven on the ring.
#
# HCA list is restricted to the first PCIe domain (rocep1s0f0, rocep1s0f1),
# matching the ring's proven configuration. The pair also has a second domain
# (roceP2p1s0f0, roceP2p1s0f1) on separate x4 links; adding it raises available
# egress bandwidth but is not exercised by this configuration yet.
#
# Weights are ~177 GiB over two nodes, so ~88.5 GiB of the 121 GiB unified
# memory per node. KV is capped explicitly rather than left to the utilization
# fraction because the weight footprint leaves little headroom.
# The NCCL/GLOO environment below is written for two directly-cabled nodes:
# /30 point-to-point links, RDMA and management device names as DGX Spark
# enumerates them. On a switched fabric, set NCCL_SOCKET_IFNAME/NCCL_IB_HCA to
# your devices, match NCCL_IB_SUBNET_PREFIX_LEN to your subnetting, and
# reconsider NCCL_ALGO=Ring and NCCL_SKIP_TREE_CONNECT.
set -euo pipefail

rank="${1:?node rank required (0 or 1)}"
host_ip="${2:?host IP for this node required}"

# Image built from image/Dockerfile. For the instanttensor weight loader, build
# the variant in README "Launch config", set its tag here, and add
# "--load-format instanttensor" to the vLLM flags below.
image="glm53-nvfp4-serving:local"
container="glm53-nvfp4-tp2-mtp3-r${rank}"
model_dir="/var/tmp/models/local-inference-lab--GLM-5.3-Flash-NVFP4/staging"
cache_dir="/var/tmp/glm53-nvfp4-tp2-jit-r${rank}"
master_addr="192.0.2.10"  # rank-0 node address; set to yours
master_port=29699

headless=()
[[ "${rank}" != "0" ]] && headless=(--headless)

mkdir -p "${cache_dir}"
docker rm -f "${container}" >/dev/null 2>&1 || true

exec docker run -d \
  --name "${container}" \
  --init \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --device /dev/infiniband:/dev/infiniband \
  --security-opt label=disable \
  -v "${model_dir}:/models/glm53nvfp4:ro" \
  -v "${cache_dir}:/cache/jit" \
  -v "/var/tmp/glm53-nvfp4-patch/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py:ro" \
  -v "/var/tmp/glm53-nvfp4-patch/modelopt.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro" \
  -v "/var/tmp/glm53-nvfp4-patch/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  -v "/var/tmp/glm53-nvfp4-patch/sparse_attn_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro" \
  -e TORCH_USE_RTLD_GLOBAL=1 \
  -e GLOO_SOCKET_IFNAME=enP7s7 \
  -e NCCL_SOCKET_IFNAME=enP7s7 \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_NET=IB \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -e NCCL_IB_SUBNET_PREFIX_LEN=30 \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CROSS_NIC=1 \
  -e NCCL_ALGO=Ring \
  -e NCCL_SKIP_TREE_CONNECT=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG=WARN \
  -e VLLM_HOST_IP="${host_ip}" \
  -e VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2 \
  -e LD_PRELOAD=/opt/sparkring/nccl/libnccl.so.2 \
  -e SPARKRING_SKIP_MM_RENDERER_WARMUP=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_USE_AOT_COMPILE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1f \
  -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CMAKE_CUDA_ARCHITECTURES=121 \
  -e XDG_CACHE_HOME=/cache/jit \
  -e TRITON_CACHE_DIR=/cache/jit/triton \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e VLLM_CACHE_ROOT=/cache/jit/vllm \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  "${image}" \
  /models/glm53nvfp4 \
  --served-model-name glm-5.3-flash-nvfp4-tp2-mtp3 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-memory-bytes 10737418240 \
  --max-model-len 524288 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --kda-prefill-backend flashkda \
  --disable-custom-all-reduce \
  --kernel-config '{"enable_cutedsl_warmup":false,"enable_flashinfer_autotune":false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --port 8000 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "${rank}" \
  --master-addr "${master_addr}" \
  --master-port "${master_port}" \
  "${headless[@]}"
