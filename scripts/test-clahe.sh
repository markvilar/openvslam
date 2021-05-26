/home/martin/dev/openvslam/build/apps/SlamApplication \
    -v \
    "/home/martin/dev/openvslam/assets/OrbVocabulary.fbow" \
    -c \
    "/home/martin/dev/openvslam/assets/Underwater-CLAHE.yaml" \
    -s \
    "/home/martin/Data/SVO/Ekne-01.svo" \
    --map \
    "/home/martin/Data/Output/SLAM-Debug/CLAHE-Map.msg" \
    --frame \
    "/home/martin/Data/Output/SLAM-Debug/CLAHE-Frames.csv" \
    --keyframe \
    "/home/martin/Data/Output/SLAM-Debug/CLAHE-Keyframes.csv" \
    --track-times \
    "/home/martin/Data/Output/SLAM-Debug/CLAHE-Track-Times.csv" \
    --process-times \
    "/home/martin/Data/Output/SLAM-Debug/CLAHE-Process-Times.csv" \
    --frame-skip 1 \
    --frame-start 4500 \
    --eval-log --auto-term \
    --bias 51057.171 \
    --preprocessing 2 \
    --clahe-clip 2.0 \
    --clahe-size 20 \
    --filter \
    --filter-dia 10 \
    --filter-color 60 \
    --filter-space 20 \
