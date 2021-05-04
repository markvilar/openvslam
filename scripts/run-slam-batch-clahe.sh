# Segment 1
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-01/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-01/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-01/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-01/track-times.txt    \
    --frame-skip 1 --frame-start 2500 --eval-log --auto-term              \
    --preprocessing 2

# Segment 2 
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-02/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-02/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-02/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-02/track-times.txt    \
    --frame-skip 1 --frame-start 8800 --eval-log --auto-term              \
    --preprocessing 2

# Segment 3
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-03/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-03/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-03/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-03/track-times.txt    \
    --frame-skip 1 --frame-start 9600 --eval-log --auto-term              \
    --preprocessing 2

# Segment 4
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-01.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-04/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-04/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-04/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-04/track-times.txt    \
    --frame-skip 1 --frame-start 12500 --eval-log --auto-term             \
    --preprocessing 2

# Segment 5
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-02.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-05/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-05/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-05/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-05/track-times.txt    \
    --frame-skip 1 --frame-start 0 --eval-log --auto-term                 \
    --preprocessing 2

# Segment 6
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-02.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-06/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-06/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-06/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-06/track-times.txt    \
    --frame-skip 1 --frame-start 1600 --eval-log --auto-term              \
    --preprocessing 2

# Segment 7
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-03.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-07/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-07/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-07/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-07/track-times.txt    \
    --frame-skip 1 --frame-start 400 --eval-log --auto-term               \
    --preprocessing 2

# Segment 8
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-03.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-08/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-08/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-08/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-08/track-times.txt    \
    --frame-skip 1 --frame-start 2800 --eval-log --auto-term              \
    --preprocessing 2

# Segment 9
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-05.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-09/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-09/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-09/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-09/track-times.txt    \
    --frame-skip 1 --frame-start 400 --eval-log --auto-term               \
    --preprocessing 2

# Segment 10
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-05.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-10/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-10/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-10/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-10/track-times.txt    \
    --frame-skip 1 --frame-start 1300 --eval-log --auto-term              \
    --preprocessing 2

# Segment 11
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-05.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-11/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-11/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-11/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-11/track-times.txt    \
    --frame-skip 1 --frame-start 3200 --eval-log --auto-term              \
    --preprocessing 2

# Segment 12
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-06.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-12/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-12/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-12/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-12/track-times.txt    \
    --frame-skip 1 --frame-start 400 --eval-log --auto-term               \
    --preprocessing 2

# Segment 13
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-06.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-13/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-13/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-13/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-13/track-times.txt    \
    --frame-skip 1 --frame-start 1000 --eval-log --auto-term              \
    --preprocessing 2

# Segment 14
/home/martin/dev/openvslam/build/apps/SlamApplication                          \
    -v /home/martin/dev/openvslam/configurations/vocabularies/orb_vocab.dbow2  \
    -c /home/martin/dev/openvslam/configurations/zed/underwater-01-clahe.yaml  \
    -s /home/martin/data/Ekne-Survey/SVO/Ekne-06.svo                           \
    --map         /home/martin/output/SLAM/CLAHE/Segment-14/map.msg            \
    --frame       /home/martin/output/SLAM/CLAHE/Segment-14/frames.txt         \
    --keyframe    /home/martin/output/SLAM/CLAHE/Segment-14/keyframes.txt      \
    --track-times /home/martin/output/SLAM/CLAHE/Segment-14/track-times.txt    \
    --frame-skip 1 --frame-start 1900 --eval-log --auto-term              \
    --preprocessing 2
