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
    --track-times \
    "/home/martin/Data/Output/SLAM-Debug/HE-Track-Times.csv" \
    --process-times \
    "/home/martin/Data/Output/SLAM-Debug/HE-Process-Times.csv" \
    --frame-skip 1 \
    --frame-start 2500 \
    --eval-log --auto-term \
    --bias 51057.171 \
    --preprocessing 1 \
