# Food Recognition & Calorie Estimation — Intern Project

A beginner-friendly, CPU-friendly ML project that:
1. Classifies a food photo into one of 11 broad food categories
2. Looks up its average calorie content
3. Lets the user pick a portion size to estimate total calories
4. Runs as a simple web app (Streamlit)

Scoped deliberately small so it trains in a reasonable time on a laptop CPU,
while still being a genuine end-to-end ML project.

---
<img width="1462" height="807" alt="image" src="https://github.com/user-attachments/assets/7a1bb69f-28ad-43d1-a84a-64c99127c7ea" />

## 1. Project Plan (2 weeks)

| Days | Task |
|------|------|
| 1–2  | Set up environment, download dataset, understand folder structure |
| 3–5  | Train the classifier (`train_model.py`) |
| 6–7  | Evaluate accuracy, fix issues, retrain if needed |
| 8–9  | Build calorie lookup + portion logic |
| 10–12| Build & test the Streamlit app |
| 13–14| Polish, write report/slides, record a demo |

---

## 2. Why 11 classes, not 101?

Datasets like Food-101 have 101 fine-grained classes (e.g. "pad thai" vs
"lasagna") and need a GPU + hours of training to do well. Since you're on
CPU with 2 weeks, we instead use the **Food-11** dataset (11 broad
categories), and freeze the pretrained CNN so we only train a small
classifier head. This is a completely standard and respectable approach —
transfer learning is literally how most production food-recognition
systems are built.

The 11 categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat,
Noodles-Pasta, Rice, Seafood, Soup, Vegetable-Fruit.

---

## 3. Setup

```bash
cd food-calorie-project
pip install -r requirements.txt
```

> Note: `requirements.txt` doesn't pin an exact TensorFlow version, since the
> right build depends on your Python version. pip will install whichever
> compatible version it can find. The model is saved/loaded as
> `food_classifier.keras` (the modern format), not `.h5`.

### Get the dataset
Download the **Food-11** dataset from Kaggle:
https://www.kaggle.com/datasets/vermaavi/food11

After downloading, arrange it like this (the Kaggle version is usually
already close to this structure — just rename folders if needed):

```
data/
  train/
    Bread/
    Dairy product/
    Dessert/
    Egg/
    Fried food/
    Meat/
    Noodles-Pasta/
    Rice/
    Seafood/
    Soup/
    Vegetable-Fruit/
  validation/
    (same 11 subfolders)
```

If you're on Colab instead of your laptop (recommended if training feels
slow — free GPU, same code works), upload the dataset to your Drive and
mount it; everything else is identical.

---

## 4. Train the model

```bash
python model/train_model.py
```

This will:
- Load MobileNetV2 pretrained on ImageNet, freeze its weights
- Add a small trainable classifier head (11 outputs)
- Train only that head on your data (fast, even on CPU)
- Save the trained model to `model/food_classifier.h5`
- Print final validation accuracy

Expect somewhere in the 75–90% validation accuracy range depending on data
quality/size — that's a solid result to report for this kind of project.

---

## 5. Calorie data

`data/calorie_data.csv` maps each of the 11 categories to average calories
per 100g (sourced from USDA-style averages — approximate by design, since
these are broad categories). Feel free to refine these numbers and cite
your sources in your report.

---

## 6. Run the app

```bash
streamlit run app/app.py
```

Upload a food photo, pick a portion size, and get a predicted category +
estimated calories.

---

## 7. What to put in your report

- Problem statement & why classification+lookup (not raw regression) was
  chosen
- Dataset description (Food-11, 11 classes, size)
- Model architecture (MobileNetV2 transfer learning, frozen base + custom
  head) with a diagram
- Training details (epochs, batch size, CPU training time)
- Accuracy / confusion matrix (add code to `train_model.py` if asked for
  more evaluation depth)
- Limitations: broad categories not specific dishes, portion size is
  self-reported not vision-estimated, calorie values are averages
- Future work: portion-size estimation via depth/reference object, larger
  fine-grained dataset, mobile deployment

## 8. Stretch goals (only if time allows)
- Swap manual portion-size dropdown for a reference-object-based size
  estimate
- Fine-tune (unfreeze) the last few CNN layers for a few epochs to push
  accuracy higher
- Add a food logging/history feature to the Streamlit app for real "diet
  tracking"
