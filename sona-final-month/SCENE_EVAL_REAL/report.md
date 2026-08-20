# Scene classifier evaluation

Overall accuracy: **0.4694** (98 clips)

## Per-class metrics

| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| speech | 0.5172 | 1.0 | 0.6818 | 30 |
| music | 0.0 | 0.0 | 0.0 | 0 |
| traffic | 0.2105 | 0.8 | 0.3333 | 5 |
| fan | 0.9167 | 0.2895 | 0.44 | 38 |
| crowd | 1.0 | 0.04 | 0.0769 | 25 |
| mixed | 0.0 | 0.0 | 0.0 | 0 |
| unknown | 0.0 | 0.0 | 0.0 | 0 |

## Confusion (rows = true, cols = predicted)

| true\pred | speech | music | traffic | fan | crowd | mixed | unknown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| speech | 30 | 0 | 0 | 0 | 0 | 0 | 0 |
| music | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| traffic | 1 | 0 | 4 | 0 | 0 | 0 | 0 |
| fan | 7 | 1 | 12 | 11 | 0 | 7 | 0 |
| crowd | 20 | 0 | 3 | 1 | 1 | 0 | 0 |
| mixed | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
