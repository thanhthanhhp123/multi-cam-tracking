#!/bin/bash
# Kiem PeopleNet Transformer chay duoc tren DeepStream 7.1 / TensorRT 10.3 — lam NGAY sau
# vast_bootstrap.sh, TRUOC khi chay WildTrack (phien 30, docs/worklog/2026-09-28-30-*).
# Chay TREN instance, cwd bat ky:  bash /workspace/mct-repo/docker/vast_peoplenet_check.sh
#
# Tra loi 3 cau ma may dev khong tra loi duoc:
#   1. libnvinfer_plugin cua TensorRT trong image co MultiscaleDeformableAttnPlugin_TRT
#      khong. Ma nguon TensorRT OSS nhanh 10.3 co plugin nay, nhung chua kiem ban binary.
#   2. trtexec build duoc engine FP16 tu ca hai file ONNX khong, va do throughput tho
#      (batch 1/4/7; qps x batch = anh/s). KHONG phai FPS cua pipeline: do bang ds_pipeline.
#   3. Ban nvinfer trong DeepStream 7.1 co cat topk ca khi cluster-mode=4 khong (da doc o
#      ma nguon DS 9). Neu khong cat thi topk=200 van vo hai.
set -euo pipefail
REPO=/workspace/mct-repo
DIR=$REPO/models/detector/peoplenet_transformer
DS=/opt/nvidia/deepstream/deepstream
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
mkdir -p /workspace/pnt_check

echo "=== 0. phien ban"
cat "$DS/version" 2>/dev/null || true
dpkg -l | grep -E "^ii +(libnvinfer10|tensorrt) " | awk '{print $2, $3}' || true

echo "=== 1. plugin trong libnvinfer_plugin"
PLUGIN_LIB=$(ldconfig -p | grep -m1 -o '/[^ ]*libnvinfer_plugin.so.10[^ ]*' || true)
echo "lib: ${PLUGIN_LIB:-KHONG THAY}"
n=$(strings "$PLUGIN_LIB" | grep -c MultiscaleDeformableAttnPlugin_TRT || true)
echo "so lan xuat hien ten plugin: $n (0 = phai build TensorRT OSS, xem deepstream_tao_apps/TRT-OSS)"

echo "=== 2. trtexec, FP16: build MOT lan moi model, luu dung ten engine ma nvinfer se doc"
# Profile min 1 / opt 7 / max 7 = dung cai nvinfer tu build cho batch 7. Luu vao
# <onnx>_b7_gpu0_fp16.engine (ten khai trong config_infer_peoplenet_transformer*.txt) de
# pipeline nap lai, khoi build lan hai. Throughput do lai tren CHINH engine do o batch 1/4/7.
for onnx in resnet50_peoplenet_transformer_op17.onnx dino_fan_small_astro_delta.onnx; do
  eng="$DIR/${onnx}_b7_gpu0_fp16.engine"
  log=/workspace/pnt_check/${onnx%.onnx}_build.log
  if [ -f "$eng" ]; then
    echo "--- da co $eng"
  else
    echo "--- build $onnx (vai phut, im lang KHONG phai treo — xem file .engine lon dan)"
    "$TRTEXEC" --onnx="$DIR/$onnx" --fp16 --saveEngine="$eng" \
      --minShapes=inputs:1x3x544x960 --optShapes=inputs:7x3x544x960 \
      --maxShapes=inputs:7x3x544x960 > "$log" 2>&1 \
      || { echo "BUILD THAT BAI, xem $log"; tail -25 "$log"; continue; }
    ls -la "$eng"
  fi
  for b in 1 4 7; do
    log=/workspace/pnt_check/${onnx%.onnx}_b${b}.log
    "$TRTEXEC" --loadEngine="$eng" --shapes=inputs:${b}x3x544x960 --duration=10 > "$log" 2>&1 \
      || { echo "chay batch $b that bai, xem $log"; continue; }
    echo "batch $b: $(grep -h -o 'Throughput: [0-9.]* qps' "$log") $(grep -h -o 'GPU Compute Time: min = [^,]*, max = [^,]*, mean = [^,]*' "$log")"
  done
done

echo "=== 3. nvinfer DS 7.1: topk khi cluster-mode=4"
SRC=$DS/sources/libs/nvdsinfer/nvdsinfer_context_impl_output_parsing.cpp
if [ -f "$SRC" ]; then
  grep -n -A8 "DetectPostprocessor::fillUnclusteredOutput" "$SRC" | grep -n "filterTopKOutputs" \
    && echo "=> CO cat topk khi khong gom cum: topk=200 la can thiet" \
    || echo "=> khong thay filterTopKOutputs trong fillUnclusteredOutput"
else
  echo "khong co $SRC"
fi
