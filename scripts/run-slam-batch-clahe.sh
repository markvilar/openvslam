# Segment 1
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-01-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-01-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-01-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-01-Times.txt    \
    --frame-skip 1 --frame-start 2500 --eval-log --auto-term              \
    --preprocessing 2

# Segment 2 
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-02-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-02-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-02-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-02-Times.txt    \
    --frame-skip 1 --frame-start 8800 --eval-log --auto-term              \
    --preprocessing 2

# Segment 3
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-03-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-03-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-03-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-03-Track-times.txt    \
    --frame-skip 1 --frame-start 9600 --eval-log --auto-term              \
    --preprocessing 2

# Segment 4
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-04-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-04-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-04-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-04-Times.txt    \
    --frame-skip 1 --frame-start 12500 --eval-log --auto-term             \
    --preprocessing 2

# Segment 5
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-02.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-05-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-05-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-05-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-05-Times.txt    \
    --frame-skip 1 --frame-start 0 --eval-log --auto-term                 \
    --preprocessing 2

# Segment 6
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-02.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-06-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-06-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-06-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-06-Times.txt    \
    --frame-skip 1 --frame-start 1600 --eval-log --auto-term              \
    --preprocessing 2

# Segment 7
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-03.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-07-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-07-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-07-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-07-Times.txt    \
    --frame-skip 1 --frame-start 200 --eval-log --auto-term               \
    --preprocessing 2

# Segment 8
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-03.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-08-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-08-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-08-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-08-Times.txt    \
    --frame-skip 1 --frame-start 2600 --eval-log --auto-term              \
    --preprocessing 2

# Segment 9
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-05.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-09-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-09-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-09-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-09-Times.txt    \
    --frame-skip 1 --frame-start 200 --eval-log --auto-term               \
    --preprocessing 2

# Segment 10
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-05.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-10-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-10-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-10-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-10-Times.txt    \
    --frame-skip 1 --frame-start 1100 --eval-log --auto-term              \
    --preprocessing 2

# Segment 11
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-05.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-11-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-11-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-11-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-11-Times.txt    \
    --frame-skip 1 --frame-start 3000 --eval-log --auto-term              \
    --preprocessing 2

# Segment 12
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-06.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-12-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-12-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-12-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-12-Times.txt    \
    --frame-skip 1 --frame-start 200 --eval-log --auto-term               \
    --preprocessing 2

# Segment 13
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-06.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-13-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-13-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-13-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-13-Times.txt    \
    --frame-skip 1 --frame-start 800 --eval-log --auto-term              \
    --preprocessing 2

# Segment 14
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-06.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-14-Map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-14-Frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-14-Keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-14-Times.txt    \
    --frame-skip 1 --frame-start 1700 --eval-log --auto-term              \
    --preprocessing 2
