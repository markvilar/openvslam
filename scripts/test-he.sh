/home/martin/dev/openvslam/build/apps/SlamApplication \
    -v \
    "/home/martin/dev/openvslam/assets/OrbVocabulary.fbow" \
    -c \
    "/home/martin/dev/openvslam/assets/Underwater-HE.yaml" \
    -s \
    "/home/martin/Data/SVO/Ekne-01.svo" \
    --map \
    "/home/martin/Data/Output/SLAM-Debug/HE-Map.msg" \
    --frame \
    "/home/martin/Data/Output/SLAM-Debug/HE-Frames.csv" \
    --keyframe \
    "/home/martin/Data/Output/SLAM-Debug/HE-Keyframes.csv" \
    --profiling \
    "/home/martin/Data/Output/SLAM-Debug/HE-Profiling.csv" \
    --keypoint \
    "/home/martin/Data/Output/SLAM-Debug/HE-Keypoints.csv" \
    --frame-skip 1 \
    --frame-start 2500 \
    --eval-log --auto-term \
    --bias 51057.171 \
    --preprocessing 1 \
