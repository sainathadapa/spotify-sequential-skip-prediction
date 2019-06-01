# WSDM Cup 2019 - Spotify - Sequential Skip Prediction Challenge - 7th place solution

You can find a report about my solution [here](workshop-paper-source/paper.pdf)

To know more about the challenge, refer the following links:
- https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge
- http://www.wsdm-conference.org/2019/wsdm-cup-2019.php

This repository's contents are shared under the Apache License 2.0.

To reproduce, follow the steps in order:
1. Download the dataset from the CrowdAI website.
2. (Optional) Create sample datasets using the [04-create-samples.ipynb](04-create-samples.ipynb) notebook.
3. Train the models using the [05-train-each-len.py](05-train-each-len.py) script.
4. (Optional) Evaluate the models on the validation set using [06-predict-each-len-val-set.py](06-predict-each-len-val-set.py).
5. Process the test data using the [07-process-test-file.py](07-process-test-file.py).
6. Finally, the submission file can be created by using the [09-create-submission-one-test-file.py](09-create-submission-one-test-file.py) script.

## Citing

```
@article{adapa2019sequential,
  title={Sequential modeling of Sessions using Recurrent Neural Networks for Skip Prediction},
  author={Adapa, Sainath},
  journal={arXiv preprint arXiv:1904.10273},
  year={2019}
}
```
