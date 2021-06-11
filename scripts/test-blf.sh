/home/martin/dev/openvslam/build/apps/SlamApplication \
    -v \
    "/home/martin/dev/openvslam/assets/OrbVocabulary.fbow" \
    -c \
    "/home/martin/dev/openvslam/assets/Underwater-BLF.yaml" \
    -s \
    "/home/martin/Data/SVO/Ekne-01.svo" \
    --map \
    "/home/martin/Data/Output/SLAM-Debug/BLF-Map.msg" \
    --frame \
    "/home/martin/Data/Output/SLAM-Debug/BLF-Frames.csv" \
    --keyframe \
    "/home/martin/Data/Output/SLAM-Debug/BLF-Keyframes.csv" \
    --profiling \
    "/home/martin/Data/Output/SLAM-Debug/BLF-Profiling.csv" \
    --keypoint \
    "/home/martin/Data/Output/SLAM-Debug/BLF-Keypoints.json" \
    --frame-skip 1 \
    --frame-start 4500 \
    --eval-log --auto-term \
    --bias 54142.8 \
    --preprocessing 0 \
    --filter \
    --filter-dia 7 \
    --filter-color 60 \
    --filter-space 20 \
