from pathlib import Path

import gradio as gr
import pandas as pd


ROOT = Path(__file__).parent
DATA = ROOT / "data"


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


accuracy = load_csv("accuracy.csv")
throughput = load_csv("throughput.csv")
inventory = load_csv("model_inventory.csv")
accuracy_leaderboard = load_csv("accuracy_leaderboard.csv")
throughput_leaderboard = load_csv("throughput_leaderboard.csv")


def overview_html() -> str:
    model_count = inventory["model"].nunique()
    config_count = len(inventory)
    families = inventory["model_family"].nunique()
    quants = ", ".join(sorted(inventory["quant"].unique()))
    tasks = ", ".join(sorted(accuracy["task"].unique()))
    rates = ", ".join(str(x) for x in sorted(throughput["request_rate"].unique()))

    return f"""
    <div style="padding: 1rem 0;">
      <h2>H200 Quantization Benchmarks</h2>
      <p>
        Final benchmark dashboard for quantized instruction-tuned LLMs served with vLLM on an NVIDIA H200 MIG setup.
      </p>
      <ul>
        <li><b>Model-quantization configurations:</b> {config_count}</li>
        <li><b>Unique models:</b> {model_count}</li>
        <li><b>Model families:</b> {families}</li>
        <li><b>Quantizations:</b> {quants}</li>
        <li><b>Quality tasks:</b> {tasks}</li>
        <li><b>Throughput request rates:</b> {rates}</li>
      </ul>
    </div>
    """


def filter_accuracy(model_family, quant, task):
    df = accuracy.copy()
    if model_family != "All":
        df = df[df["model_family"] == model_family]
    if quant != "All":
        df = df[df["quant"] == quant]
    if task != "All":
        df = df[df["task"] == task]
    return df.sort_values(["task", "accuracy"], ascending=[True, False])


def filter_throughput(quant, request_rate):
    df = throughput.copy()
    if quant != "All":
        df = df[df["quant"] == quant]
    if request_rate != "All":
        df = df[df["request_rate"].astype(str) == str(request_rate)]
    return df.sort_values("output_tok_throughput", ascending=False)


families = ["All"] + sorted(accuracy["model_family"].unique())
quants = ["All"] + sorted(accuracy["quant"].unique())
tasks = ["All"] + sorted(accuracy["task"].unique())
rates = ["All"] + [str(x) for x in sorted(throughput["request_rate"].unique())]


with gr.Blocks(title="H200 Quantization Dashboard") as demo:
    gr.HTML(overview_html())

    with gr.Tab("Accuracy Leaderboard"):
        gr.Dataframe(accuracy_leaderboard, interactive=False)

    with gr.Tab("Throughput Leaderboard"):
        gr.Dataframe(throughput_leaderboard, interactive=False)

    with gr.Tab("Accuracy Explorer"):
        with gr.Row():
            family_filter = gr.Dropdown(families, value="All", label="Model family")
            quant_filter = gr.Dropdown(quants, value="All", label="Quantization")
            task_filter = gr.Dropdown(tasks, value="All", label="Task")
        accuracy_table = gr.Dataframe(
            filter_accuracy("All", "All", "All"),
            interactive=False,
            label="Accuracy rows",
        )
        for component in [family_filter, quant_filter, task_filter]:
            component.change(
                filter_accuracy,
                inputs=[family_filter, quant_filter, task_filter],
                outputs=accuracy_table,
            )

    with gr.Tab("Throughput Explorer"):
        with gr.Row():
            throughput_quant_filter = gr.Dropdown(quants, value="All", label="Quantization")
            rate_filter = gr.Dropdown(rates, value="All", label="Request rate")
        throughput_table = gr.Dataframe(
            filter_throughput("All", "All"),
            interactive=False,
            label="Throughput rows",
        )
        for component in [throughput_quant_filter, rate_filter]:
            component.change(
                filter_throughput,
                inputs=[throughput_quant_filter, rate_filter],
                outputs=throughput_table,
            )

    with gr.Tab("Model Inventory"):
        gr.Dataframe(inventory, interactive=False)


if __name__ == "__main__":
    demo.launch()
