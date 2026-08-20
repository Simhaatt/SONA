# Scene classifier evaluation

Overall accuracy: **0.7667** (60 clips)

## Per-class metrics

| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| speech | 0.75 | 0.75 | 0.75 | 8 |
| music | 0.6 | 0.6 | 0.6 | 5 |
| traffic | 0.8889 | 0.8 | 0.8421 | 10 |
| fan | 0.7143 | 0.5 | 0.5882 | 10 |
| crowd | 0.7143 | 0.9091 | 0.8 | 11 |
| mixed | 0.75 | 0.6 | 0.6667 | 5 |
| unknown | 0.8462 | 1.0 | 0.9167 | 11 |

## Confusion (rows = true, cols = predicted)

| true\pred | speech | music | traffic | fan | crowd | mixed | unknown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| speech | 6 | 0 | 1 | 0 | 1 | 0 | 0 |
| music | 1 | 3 | 0 | 0 | 1 | 0 | 0 |
| traffic | 0 | 0 | 8 | 0 | 1 | 1 | 0 |
| fan | 0 | 2 | 0 | 5 | 1 | 0 | 2 |
| crowd | 0 | 0 | 0 | 1 | 10 | 0 | 0 |
| mixed | 1 | 0 | 0 | 1 | 0 | 3 | 0 |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
