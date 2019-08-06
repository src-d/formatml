# CodRep 2019 Visualizer

To use the visualizer docker image, you need to:

- obtain a results file by writing the standard output of your competing program to a file
- mount the dataset you want to visualize to `/codrep/data`
- mount your results file to `/codrep/results`
- publish port 5001

Reminder: docker mounts require **absolute** paths on the host side.

Complete command:

```
docker run \
    --volume /absolute/path/to/data:/codrep/data \
    --volume /absolute/path/to/results:/codrep/results \
    --publish 5001:5001 \
    --rm \
    --tty \
    --interactive \
    srcd/codrep2019-visualizer:0.1.0
```
