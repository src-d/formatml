FROM srcd/formatml:0.3.0

RUN mkdir /codrep /data /metadata

WORKDIR /codrep

ADD configs ./training-configs
ADD outputs/train/checkpoints/s0.5968-e9-i720.tar checkpoint.tar
ADD outputs/instance.pickle.bz2 .

ENTRYPOINT formatml codrep run \
    --raw-dir /data \
    --uasts-dir uasts \
    --instance-file instance.pickle.bz2 \
    --tensors-dir tensors \
    --checkpoint-file checkpoint.tar \
    --training-configs-dir training-configs \
    --configs-dir configs \
    --metadata-dir /metadata \
    --prefix $DATASET
