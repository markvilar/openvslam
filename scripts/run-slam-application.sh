/home/martin/Development/openvslam/build/apps/SlamApplication \
    -v \
    "/home/martin/Development/openvslam/assets/OrbVocabulary.dbow2" \
    -c \
    "/home/martin/Development/openvslam/assets/Underwater-Ekne.yaml" \
    -s \
    "/home/martin/Data/SVO/Ekne-01.svo" \
    --map \
    "/home/martin/Data/SLAM/Map.msg" \
    --frame \
    "/home/martin/Data/SLAM/Frames.csv" \
    --keyframe \
    "/home/martin/Data/SLAM/Keyframes.csv" \
    --track-times \
    "/home/martin/Data/SLAM/Track-Times.csv" \
    --process-times \
    "/home/martin/Data/SLAM/Process-Times.csv" \
    --frame-skip 1 \
    --frame-start 2300 \
    --eval-log --auto-term \
    --bias 51057.171 \
    --preprocessing 0 \
    --filter \
    --filter-dia 7 \
    --filter-color 20 \
    --filter-space 20 \
