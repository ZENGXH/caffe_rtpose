#!/usr/bin/env sh
mkdir caffemodel
/data2/zengxiaohui/Multi-Person-Pose-Estimation/caffe_demo/build/tools/caffe \
    -v=0 \
    train \
    --solver=pose_solver.prototxt --gpu=4,5,6,7 \
    --snapshot="./caffemodel/pose_iter_145000.solverstate" \
    2>&1 | tee -a ./output.txt

    # --weights="../../pose_exp/VGG_ILSVRC_19_layers.caffemodel" \
