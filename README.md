# vwsd
SemEval2023-Visual Word Sense Disambiguation Task: https://raganato.github.io/vwsd/

# Get Results
For given task; (i) trial, (ii) train, (iii) test and prediction file path, gives the accuracy.
```
python eval.py --dataset <task> --input_path </path/to/file>
```
Example:
```
python eval.py --dataset trial --input_path /kuacc/users/eacikgoz17/semeval/results_trial_right.txt
```

## Current Results
| Model         | Task            | Context       | Score         | Score         |
| :------------ |:---------------:|:-------------:|:-------------:|:-------------:|
| Clip          | Trial           |         Full  |      9/16     |     56.25%    |
| Clip          | Trial           |     Left-only |               |               |
| Clip          | Trial           |    Right-only |               |               |
| Clip          | Train           |         Full  |               |               |
| Clip          | Train           |     Left-only |               |               |
| Clip          | Train           |    Right-only |               |               |
| :------------ |:---------------:|:-------------:|:-------------:|:-------------:|
| Blip          | Trial           |         Full  |      9/16     |     56.25%    |
| Blip          | Trial           |     Left-only |      7/16     |     43.75%    |
| Blip          | Trial           |    Right-only |      6/16     |     37.50%    |
| Blip          | Train           |         Full  |   7793/12869  |     60.55%    |
| Blip          | Train           |     Left-only |               |               |
| Blip          | Train           |    Right-only |               |               |

