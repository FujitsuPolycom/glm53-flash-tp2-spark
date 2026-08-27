# Third-party notices

## NCCL

`image/nccl/libnccl.so.2.30.7` is a binary build of NVIDIA NCCL. It
self-identifies as `NCCL version 2.30.7 compiled with CUDA 13.0` and targets
arm64. NCCL is licensed under the Apache License 2.0, with portions under a
BSD license held by NVIDIA CORPORATION, Lawrence Berkeley National Laboratory,
and the U.S. Department of Energy; the canonical license text is `LICENSE.txt`
in the NCCL source repository, https://github.com/NVIDIA/nccl. The source
revision of this build is unrecorded. Its dynamic symbol table and embedded
strings match a stock NCCL build; no additional symbols or vendor markers are
present.

## vLLM

The patch files under `patches/` and `image/patches/` are unified diffs whose
context and modified lines derive from vLLM, licensed under the Apache License
2.0, https://github.com/vllm-project/vllm. The specific files patched are
those distributed in the public container image
`vllm/vllm-openai:glm53-flash`, pinned by digest in `image/Dockerfile`.

## SM120 attention overlay

The base container image incorporates the SM120 attention overlay from
https://github.com/chriswritescode-dev/glm-5.3-flash-sm120 at commit
`dc6b4fdd68005ab6ee0b1decfa4ebb8384393d37`. This repository does not
redistribute that overlay separately; it is consumed only through the pinned
base image.
