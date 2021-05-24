/home/martin/dev/openvslam/build/apps/SlamApplication \
    -v \
    "/home/martin/dev/openvslam/assets/OrbVocabulary.fbow" \
    -c \
    "/home/martin/dev/openvslam/assets/Underwater-UIENet.yaml" \
    -s \
    "/home/martin/Data/SVO/Ekne-01.svo" \
    --map \
    "/home/martin/Data/Output/SLAM-Debug/UIENet-Map.msg" \
    --frame \
    "/home/martin/Data/Output/SLAM-Debug/UIENet-Frames.csv" \
    --keyframe \
    "/home/martin/Data/Output/SLAM-Debug/UIENet-Keyframes.csv" \
    --track-times \
    "/home/martin/Data/Output/SLAM-Debug/UIENet-Track-Times.csv" \
    --process-times \
    "/home/martin/Data/Output/SLAM-Debug/UIENet-Process-Times.csv" \
    --frame-skip 1 \
    --frame-start 2500 \
    --eval-log --auto-term \
    --bias 51057.171 \
    --preprocessing 3 \
    --uienet \
    "/home/martin/dev/openvslam/assets/UIENet-CUDA-Script.pt" \
    --filter \
    --filter-dia 7 \
    --filter-color 60 \
    --filter-space 20 \
