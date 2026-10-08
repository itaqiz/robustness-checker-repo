import os
import sqlite3
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import robustness as rb

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = rb.load_config(os.path.join(HERE, 'config.json'))
MODEL = rb.load_model(os.path.join(HERE, 'resnet18_best.pt'))
DB_PATH = os.path.join(HERE, 'robustness.db')

SQL = """
SELECT kind, level,
       ROUND(AVG(correct), 3) AS accuracy,
       ROUND(AVG(conf < ?), 3) AS review_rate,
       COUNT(*) AS n
FROM predictions
GROUP BY kind, level
ORDER BY kind, level
"""


def run_analysis(img, level):
    if img is None:
        return "Upload a photo first.", []
    rows, needs_review, reasons = rb.analyze(MODEL, img, CONFIG, level=int(level))
    if needs_review:
        verdict = "### 🔴 Needs human review\n" + "\n".join(f"- {r}" for r in reasons)
    else:
        verdict = ("### 🟢 Model decision accepted\n"
                   "Confidence is above the threshold and the prediction is stable "
                   "under a mild blur.")
    verdict += (f"\n\nThe pictures below are a **stress test**: what the model would say "
                f"if this photo were degraded further (severity {int(level)} of 5). "
                f"The verdict above uses the validated review rule, not these rows.")
    gallery = [(r['image'],
                f"{r['name']}: {r['prediction']} ({r['confidence']:.0%}) | {r['note']}")
               for r in rows]
    return verdict, gallery


def curves(threshold):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(SQL, con, params=(float(threshold),))
    con.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, g in df.groupby('kind'):
        axes[0].plot(g['level'], g['accuracy'], marker='o', label=rb.LABELS[kind])
        axes[1].plot(g['level'], g['review_rate'], marker='o', label=rb.LABELS[kind])
    axes[0].set_title('Accuracy vs severity')
    axes[0].set_ylim(0.4, 1.02)
    axes[1].set_title(f'Share sent to review (confidence < {float(threshold):.2f})')
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel('severity level')
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    plt.close(fig)
    return fig, df


with gr.Blocks(title="Produce freshness: robustness checker") as demo:
    gr.Markdown(
        "# Produce freshness: robustness checker\n"
        "A ResNet-18 that classifies fruit as **fresh or rotten**, tested against "
        "blur, low light, noise, JPEG compression and blocked regions. When it is "
        "unsure, it routes the photo to a human. Trained on apple, banana, mango, "
        "orange and strawberry only, so it will give a confident-looking answer for "
        "anything else too."
    )
    with gr.Tab("Stress-test an image"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type='pil', label='Upload a photo of a fruit')
                level = gr.Slider(1, 5, value=3, step=1, label='Degradation severity')
                btn = gr.Button('Run', variant='primary')
            with gr.Column():
                verdict = gr.Markdown()
        gallery = gr.Gallery(label='Stress test', columns=3)
        btn.click(run_analysis, [inp, level], [verdict, gallery])

    with gr.Tab("Accuracy vs severity"):
        gr.Markdown(
            "Precomputed on the 900 held-out test images (SQLite, queried with SQL). "
            "The review rate here uses the confidence rule only. The live app also "
            "uses a mild-blur probe."
        )
        thr = gr.Slider(0.5, 0.99, value=CONFIG['threshold'], step=0.01,
                        label='Confidence threshold')
        plot = gr.Plot()
        table = gr.Dataframe()
        thr.change(curves, thr, [plot, table])
        demo.load(curves, thr, [plot, table])

if __name__ == '__main__':
    demo.launch()
