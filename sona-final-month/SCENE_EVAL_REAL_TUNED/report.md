# Scene classifier evaluation

Overall accuracy: **0.7857** (98 clips)

## Per-class metrics

| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| speech | 0.7317 | 1.0 | 0.8451 | 30 |
| music | 0.0 | 0.0 | 0.0 | 0 |
| traffic | 0.375 | 0.6 | 0.4615 | 5 |
| fan | 0.9259 | 0.6579 | 0.7692 | 38 |
| crowd | 0.9048 | 0.76 | 0.8261 | 25 |
| mixed | 0.0 | 0.0 | 0.0 | 0 |
| unknown | 0.0 | 0.0 | 0.0 | 0 |

## Confusion (rows = true, cols = predicted)

| true\pred | speech | music | traffic | fan | crowd | mixed | unknown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| speech | 30 | 0 | 0 | 0 | 0 | 0 | 0 |
| music | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| traffic | 1 | 0 | 3 | 1 | 0 | 0 | 0 |
| fan | 7 | 0 | 3 | 25 | 2 | 1 | 0 |
| crowd | 3 | 0 | 2 | 1 | 19 | 0 | 0 |
| mixed | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
