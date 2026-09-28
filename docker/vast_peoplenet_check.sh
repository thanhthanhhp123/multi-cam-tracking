#!/bin/bash
# Kiem PeopleNet Transformer chay duoc tren DeepStream 7.1 / TensorRT 10.3 — lam NGAY sau
# vast_bootstrap.sh, TRUOC khi chay WildTrack (phien 30, docs/worklog/2026-09-28-30-*).
# Chay TREN instance, cwd bat ky:  bash /workspace/mct-repo/docker/vast_peoplenet_check.sh
#
# Tra loi 3 cau ma may dev khong tra loi duoc:
#   1. libnvinfer_plugin cua TensorRT trong image co MultiscaleDeformableAttnPlugin_TRT
#      khong. Ma nguon TensorRT OSS nhanh 10.3 co plugin nay, nhung chua kiem ban binary.
#   2. trtexec build duoc engine FP16 tu ca hai file ONNX khong, va do throughput tho
#      (batch 1 va 4). Day KHONG phai so FPS cua pipeline: FPS 4 luong do bang ds_pipeline.
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

echo "=== 2. trtexec, FP16"
for onnx in resnet50_peoplenet_transformer_op17.onnx dino_fan_small_astro_delta.onnx; do
  for b in 1 4; do
    log=/workspace/pnt_check/${onnx%.onnx}_b${b}.log
    echo "--- $onnx batch $b (build vai phut, im lang KHONG phai treo)"
    "$TRTEXEC" --onnx="$DIR/$onnx" --fp16 \
      --minShapes=inputs:1x3x544x960 --optShapes=inputs:${b}x3x544x960 --maxShapes=inputs:7x3x544x960 \
      --shapes=inputs:${b}x3x544x960 --duration=10 > "$log" 2>&1 || { echo "THAT BAI, xem $log"; tail -20 "$log"; continue; }
    grep -E "Throughput|GPU Compute Time: min" "$log" | sed 's/^.*\] //'
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
