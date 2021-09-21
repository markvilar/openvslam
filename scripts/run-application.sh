/home/martin/Development/openvslam/build/apps/SlamApplication \
    -v /home/martin/Development/openvslam/assets/OrbVocabulary.fbow \
    -c /home/martin/Development/openvslam/assets/Underwater-Ekne.yaml \
    -s /home/martin/Data/ZED/SVO/Ekne-01.svo \
    --map \
    "/home/martin/Data/OpenVSLAM/Map.msg" \
    --frame \
    "/home/martin/Data/OpenVSLAM/Frames.csv" \
    --keyframe \
    "/home/martin/Data/OpenVSLAM/Keyframes.csv" \
    --track-times \
    "/home/martin/Data/OpenVSLAM/Track-Times.csv" \
    --process-times \
    "/home/martin/Data/OpenVSLAM/Process-Times.csv" \
    --frame-skip 3 \
    --frame-start 2500 \
    --eval-log --auto-term \
    --bias 51057.171 \
    --preprocessing 0 \
    --filter \
    --filter-dia 7 \
    --filter-color 20 \
    --filter-space 20 \
