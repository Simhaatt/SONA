# Scene classifier evaluation

Overall accuracy: **0.4094** (320 clips)

## Per-class metrics

| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| speech | 0.2703 | 1.0 | 0.4255 | 40 |
| music | 0.0 | 0.0 | 0.0 | 0 |
| traffic | 0.6 | 0.075 | 0.1333 | 40 |
| fan | 0.678 | 0.2632 | 0.3791 | 152 |
| crowd | 0.4615 | 0.5455 | 0.5 | 88 |
| mixed | 0.0 | 0.0 | 0.0 | 0 |
| unknown | 0.0 | 0.0 | 0.0 | 0 |

## Confusion (rows = true, cols = predicted)

| true\pred | speech | music | traffic | fan | crowd | mixed | unknown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| speech | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| music | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| traffic | 14 | 0 | 3 | 11 | 12 | 0 | 0 |
| fan | 65 | 2 | 1 | 40 | 44 | 0 | 0 |
| crowd | 29 | 2 | 1 | 8 | 48 | 0 | 0 |
| mixed | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
