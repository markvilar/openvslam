gdb --args \
    /home/martin/dev/openvslam/build/apps/SlamApplication \
    -v \
    "/home/martin/dev/openvslam/assets/OrbVocabulary.fbow" \
    -c \
    "/home/martin/dev/openvslam/assets/Underwater-Raw.yaml" \
    -s \
    "/home/martin/Data/SVO/Ekne-01.svo" \
    --map \
    "/home/martin/Data/Output/SLAM-Debug/Map.msg" \
    --frame \
    "/home/martin/Data/Output/SLAM-Debug/Frames.csv" \
    --keyframe \
    "/home/martin/Data/Output/SLAM-Debug/KeyFrames.csv" \
    --track-times \
    "/home/martin/Data/Output/SLAM-Debug/Track-Times.csv" \
    --eval-log \
    --frame-skip 1 \
    --frame-start 2500 \
    --eval-log --auto-term --debug \
    --bias 51057.171 \
    --preprocessing 1 \
    --filter \
    --filter-dia 7 \
    --filter-color 40 \
    --filter-space 40 \
