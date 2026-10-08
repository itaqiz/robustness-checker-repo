# Produce freshness: robustness checker

A ResNet-18 fine-tuned to classify fruit as **fresh or rotten**, then stress-tested
against blur, low light, noise, JPEG compression and a blocked region. When it is
unsure, it routes the photo to a human instead of guessing.

Most image classifiers are reported with one accuracy number on clean photos. Factory
cameras and clinic scanners do not produce clean photos, so this project measures how
accuracy falls as conditions get worse, and tests a "send this to a human" rule.

## Demo

![Stress test at severity 3](docs/screenshot-1.png)
![Stress test at severity 5](docs/screenshot-2.png)
![Accuracy vs severity](docs/screenshot-3.png)

The app has two tabs. The first takes an uploaded photo and shows the prediction and
confidence for the original and for five degraded versions, plus a review verdict.
The second shows accuracy against severity for the whole test set, stored in SQLite
and queried with SQL.

The pictures below the verdict are a **stress test**: what the model would say if the
photo were degraded further. The verdict itself comes from the validated review rule
described below.

## Results (900 held-out test images)

Clean accuracy is 99.4% (5 errors in 900). Accuracy at the highest severity (level 5):

| Corruption | Accuracy at level 5 |
|---|---|
| Blur | 63.7% |
| Low light | 60.7% |
| Noise | 67.1% |
| JPEG compression | 89.8% |
| Blocked region | 88.6% |

## Human-review rule

An image is sent to review if the model's confidence is below **0.95**, or if its
prediction flips under one mild extra blur. The threshold and the probe were chosen
on a validation set. The numbers below are from the untouched test set, on images
degraded at severity 3 and 5.

| Rule | Clean images sent to review | Accuracy on images the model keeps | Confident errors caught by the probe |
|---|---|---|---|
| Threshold only | 2.0% | 95.9% | 0% |
| Threshold + blur probe (used) | 4.8% | 97.0% | 31.2% |
| Threshold + all 5 probes | 12.3% | 97.9% | 60.6% |

"Confident errors" are wrong predictions made with at least 95% confidence, which the
threshold cannot catch. The noise probe catches many of them but flags about 10% of
clean images, so it was rejected on cost.

## How it was built

- **Data:** the Fruits folder of the Kaggle "Fruits and Vegetables dataset" by
  Mukhriddin Mukhiddinov (CC0): 5,997 images, 10 classes (fresh and rotten apple,
  banana, mango, orange, strawberry). Labels are fresh or rotten, taken from the
  folder names. Stratified 70/15/15 split with a fixed seed: 4,197 / 900 / 900.
- **Model:** torchvision ResNet-18, ImageNet-pretrained, final layer replaced with two
  outputs. Fine-tuned for 5 epochs (AdamW, learning rate 1e-4, batch size 32) with
  light augmentation (random crop and flip). The weights from the best validation
  epoch are kept.
- **Corruptions:** five types at five severity levels, applied after resizing and
  cropping. The severity values were checked by eye on a sample image and then fixed
  before the model was tested on them.
- **Selection discipline:** the threshold and the probe were chosen on the validation
  set. The test set was used once for the final numbers.

## Run it yourself

    git clone https://github.com/itaqiz/robustness-checker.git
    cd robustness-checker
    pip install -r requirements.txt
    python app.py

Then open the local address that Gradio prints. The model (45 MB) and the results
database are in the repo, so no training is needed. The training and experiment code
is in `notebook/robustness_checker_training.ipynb`.

## Demo images

These are test images, in `demo_images/`:

| File | What it shows |
|---|---|
| `fails_blur.jpg` | Rotten. Correct at 0.999 when clean. Under severity-5 blur the model says fresh at 0.999. |
| `fails_jpeg.jpg` | Fresh. Correct at 1.000 when clean. Under severity-5 JPEG it says rotten at 0.997. |
| `fails_low_light.jpg` | Fresh. Correct at 1.000 when clean. Under severity-5 low light it says rotten at 0.994. |
| `stays_correct.jpg` | Rotten. Correct at 0.988 or higher under all five corruptions. |
| `needs_review.jpg` | The clean test image the model is least sure about. The review rule sends it to a human. |

The three `fails_` images were picked as the most confident failures among the 900
test images, so they are worst cases and not typical behaviour.

## Limitations

- About 69% of the model's confident errors (wrong but at least 95% sure) still get
  through the review rule.
- The corruptions are synthetic. Dust and glare were not tested, and no real
  factory or clinic camera data was used.
- Training and test images come from the same web-sourced dataset, so clean
  accuracy is likely optimistic for new cameras.
- The model knows five fruits only and will answer confidently about anything else.
- For an uploaded photo, the noise and blocked-region pictures use one random draw.

## Possible next steps

- Train with noise and blur augmentation, which should help the weakest corruptions.
- Add dust and glare, and test on photos from real cameras.
- Export the model to ONNX to run it on cheaper or serverless hosting.

## Data and citation

The dataset page says the images were gathered from online sources. The CC0 label is
the uploader's declaration. Only five sample images are included in this repo.

Mukhiddinov, M., Muminov, A., Cho, J. (2022). Improved Classification Approach for
Fruits and Vegetables Freshness Based on Deep Learning. Sensors 22(21), 8192.
https://doi.org/10.3390/s22218192
