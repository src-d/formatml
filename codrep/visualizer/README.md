# CodRep 2019 Visualizer

To use the visualizer docker image, you need to:

- obtain a results file by writing the standard output of your competing program to a file
- mount the dataset you want to visualize to `/codrep/data`
- mount your results file to `/codrep/results`
- optionally, output a json file containing metadata with the structure:

  ```json
  {
    "columns": ["Probability"],
    "metadata":
    {
      "/path/to/task/0.txt":
      {
        "0": [0.9],
        "968": [0.1]
      },
      "/path/to/task/1.txt":
      {
        "3": [0.2],
        "1085": [0.3],
        "20987": [0.5],
      },
      …
    }
  }
  ```

  and mount it to `/codrep/metadata`

- publish port 5001

Reminder: docker mounts require **absolute** paths on the host side.

Complete command:

```
docker run \
    --volume /absolute/path/to/data:/codrep/data \
    --volume /absolute/path/to/results.txt:/codrep/results \
    --volume /absolute/path/to/metadata.json:/codrep/metadata \
    --publish 5001:5001 \
    --rm \
    --tty \
    --interactive \
    srcd/codrep2019-visualizer:0.2.0
```
