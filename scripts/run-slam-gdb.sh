gdb --args \
    /home/martin/dev/openvslam/build/apps/SlamApplication \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2 \
    -c /home/martin/dev/openvslam/configurations/zed/test.yaml \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo \
    --map /home/martin/output/SLAM/Debug/Map.msg \
    --frame /home/martin/output/SLAM/Debug/Frames.csv \
    --keyframe /home/martin/output/SLAM/Debug/KeyFrames.csv \
    --track-times /home/martin/output/SLAM/Debug/Track-Times.csv \
    --frame-skip 1 \
    --frame-start 2500 \
    --eval-log \
