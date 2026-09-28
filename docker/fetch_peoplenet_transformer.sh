#!/bin/bash
# Tai hai ban PeopleNet Transformer tu NGC (cong khai, khong can API key) vao
# models/detector/peoplenet_transformer/ va kiem sha256. Chay o goc repo, tren may dev
# hoac tren vast-gpu (mang vast nhanh hon, ~310 MB).
#
#   v1.1  peoplenet_transformer:deployable_v1.1     Deformable DETR + ResNet50 ("PNT" cua bai MV3DT)
#   v2    peoplenet_transformer_v2:deployable_v1.0  DINO + FAN-Small (ban reference app MV3DT tai)
#
# Hash ghi lai ngay 2026-09-28 (phien 30). NGC doi file ma giu nguyen ten version thi hash
# lech va script dung lai: khi do so lai bang config truoc khi sua hash.
set -euo pipefail
DIR=models/detector/peoplenet_transformer
NGC=https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao
mkdir -p "$DIR"

tai() {  # tai <url> <file dich>
  [ -f "$2" ] && { echo "da co: $2"; return; }
  curl -fsSL -o "$2.part" "$1" && mv "$2.part" "$2"
  echo "tai xong: $2 ($(du -h "$2" | cut -f1))"
}

tai "$NGC/peoplenet_transformer/deployable_v1.1/files?redirect=true&path=resnet50_peoplenet_transformer_op17.onnx" \
    "$DIR/resnet50_peoplenet_transformer_op17.onnx"
tai "$NGC/peoplenet_transformer_v2/deployable_v1.0/files?redirect=true&path=dino_fan_small_astro_delta.onnx" \
    "$DIR/dino_fan_small_astro_delta.onnx"
tai "https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt" \
    "$DIR/labels.txt"

(cd "$DIR" && sha256sum -c - <<'EOF'
b7c9e8159697c046bb5b6534b4fd7323626430f9ea8c7f4b16009b58bcbb6252  resnet50_peoplenet_transformer_op17.onnx
974d6a77b4fde5ad83b18bbabc5e3771220e6ee22224e98cdc0cbf733da06038  dino_fan_small_astro_delta.onnx
67f6f595cbfbd3180afadb2359f334d76947c33c7615f5547af55b7aaeb41cb6  labels.txt
EOF
)
