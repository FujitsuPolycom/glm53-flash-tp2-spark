# Serving image for GLM-5.3-Flash-NVFP4 on DGX Spark GB10 pairs.
#
# The base is public and pinned by digest. It already carries the SM120
# attention overlay from chriswritescode-dev/glm-5.3-flash-sm120 at commit
# dc6b4fdd68005ab6ee0b1decfa4ebb8384393d37.
#
# Two source changes are applied on top:
#   renderers-base.patch  - multimodal renderer
#   glm5next-kda.patch    - linear-attention (KDA) prefill path, selected at
#                           run time by --kda-prefill-backend flashkda
#
# nccl/libnccl.so.2.30.7 is the NCCL build the launch script selects through
# VLLM_NCCL_SO_PATH and LD_PRELOAD. It self-identifies as "NCCL version 2.30.7
# compiled with CUDA 13.0", targets arm64, and has sha256
# ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f. The build's
# source revision is unrecorded; its dynamic symbols and embedded strings match
# a stock NCCL build. License terms: THIRD_PARTY_NOTICES.md at the repository
# root.
FROM vllm/vllm-openai:glm53-flash@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4

ARG VLLM_DIR=/usr/local/lib/python3.12/dist-packages/vllm

COPY patches/ /tmp/patches/
RUN patch "${VLLM_DIR}/renderers/base.py"                 < /tmp/patches/renderers-base.patch \
 && patch "${VLLM_DIR}/models/glm5next/nvidia/kda.py"     < /tmp/patches/glm5next-kda.patch \
 && python3 -m py_compile \
      "${VLLM_DIR}/renderers/base.py" \
      "${VLLM_DIR}/models/glm5next/nvidia/kda.py" \
 && rm -rf /tmp/patches

COPY nccl/libnccl.so.2.30.7 /opt/sparkring/nccl/libnccl.so.2.30.7
RUN ln -sf libnccl.so.2.30.7 /opt/sparkring/nccl/libnccl.so.2  && ln -sf libnccl.so.2.30.7 /opt/sparkring/nccl/libnccl.so

LABEL org.opencontainers.image.base.name="vllm/vllm-openai:glm53-flash" \
      org.opencontainers.image.description="GLM-5.3-Flash-NVFP4 serving image for DGX Spark GB10 pairs"
