# Part B — Summary Report: ANN, CNN, RNN, YOLO

## Results at a glance

| Architecture | Task                                           | Test Accuracy |
|--------------|-------------------------------------------------|----------------|
| ANN          | Tabular binary classification (breast cancer)    | 0.9561 |
| CNN          | Image classification, 10 classes (digits)        | 0.9861 |
| SimpleRNN    | Sequence classification (synthetic sentiment)    | 0.9988 |
| LSTM         | Sequence classification (synthetic sentiment)    | 1.0000 |

YOLOv4-tiny detected objects correctly across all 8 test images (people, dogs, vehicles, animals,
kites), with per-image CPU inference time between **93 ms and 463 ms** (see
`part_b_yolo_detections.csv`).

## Complexity, training time, and use-case fit

- **ANN** — simplest architecture, fewest parameters, trains in seconds on CPU. Best suited to
  clean tabular data where features are already meaningful on their own (medical measurements,
  financial ratios, churn indicators). No spatial or sequential structure to exploit, so a dense
  network is both sufficient and the cheapest option.
- **CNN** — moderate complexity (convolution + pooling layers add parameters and compute vs. ANN,
  but far fewer than an equivalently-sized dense network would need). Training took longer than the
  ANN due to convolution cost and data augmentation, but convergence was still fast on this small
  8x8 image dataset. Best suited to any grid-structured data — images, and by extension
  spectrograms or spatial sensor grids — where local pixel patterns matter more than global
  position.
- **RNN / LSTM** — sequential processing makes training inherently slower per epoch than ANN/CNN
  (no full parallelism across the time dimension), though on this short-sequence
  (max length 60) synthetic dataset both SimpleRNN and LSTM converged quickly. Best suited to
  variable-length ordered data — text, sensor time-series, audio — where the meaning depends on
  sequence order and long-range dependencies. LSTM's gating slightly outperformed the plain
  SimpleRNN here and would show a larger advantage on longer, noisier sequences where
  vanishing gradients become a bigger problem for the plain RNN.
- **YOLO (YOLOv4-tiny)** — the most specialized architecture: a single forward pass predicts all
  bounding boxes, classes, and confidences at once (as opposed to classification-only CNNs), making
  it far more expensive per-image than a classification CNN, but still fast enough (under half a
  second per image on CPU with a "tiny" backbone) for near-real-time use. Best suited to any task
  needing localization, not just labeling — surveillance, autonomous driving, retail analytics.

## Overall takeaway

Model complexity should track data structure, not just task difficulty: the fully-connected ANN
was the *right-sized* model for flat tabular data, not an under-powered one — adding convolution or
recurrence there would only add cost with no accuracy benefit. CNN and RNN/LSTM earn their extra
complexity specifically because they encode assumptions (translation invariance, sequential order)
that match their respective data. YOLO sits a level above plain classification because its task
(localize *and* classify multiple objects per image) is strictly harder, which is reflected in its
noticeably higher per-image inference cost.
