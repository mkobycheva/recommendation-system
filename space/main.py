"""HF Spaces entry point: a single Gradio process (Spaces' Gradio SDK expects
one process, not a separate FastAPI backend). Recommenders are called
in-process here -- the local dev setup (app/api/main.py + app/frontend/app.py,
talking over HTTP) is untouched and still the reference for that boundary.

Model artifacts (~4GB) don't fit in the Space's own git repo (1GB cap on the
free tier), so they're downloaded at startup from a companion public HF
Dataset repo -- see the "Publish artifacts" cell at the end of
notebooks/00_export_metadata.ipynb for how that repo gets populated.
"""

import os

import gradio as gr
import pandas as pd
from huggingface_hub import snapshot_download
from rapidfuzz import fuzz, process

from app.recommenders.item2vec import Item2VecRecommender
from app.recommenders.sequential import BERT4RecRecommender, SASRecRecommender
from app.recommenders.svd_als import FoldInRecommender

ARTIFACTS_REPO = os.environ.get("ARTIFACTS_REPO", "<your-hf-username>/recsys-artifacts")
MODEL_CHOICES = ["svd", "als", "item2vec", "sasrec", "bert4rec"]
DOMAIN_CHOICES = ["books", "movies"]

MODEL_LOADERS = {
    "svd": FoldInRecommender.load,
    "als": FoldInRecommender.load,
    "item2vec": Item2VecRecommender.load,
    "sasrec": SASRecRecommender.load,
    "bert4rec": BERT4RecRecommender.load,
}

artifacts_dir = snapshot_download(repo_id=ARTIFACTS_REPO, repo_type="dataset")
items_metadata = pd.read_parquet(f"{artifacts_dir}/items_metadata.parquet")
recommenders = {name: load(f"{artifacts_dir}/{name}") for name, load in MODEL_LOADERS.items()}


def _items_to_records(item_ids: list[str]) -> list[dict]:
    results = items_metadata.set_index("item_id").reindex(item_ids).dropna(subset=["title"])
    results = results.reset_index()
    return results[["item_id", "title", "domain", "image_url"]].to_dict("records")


def search(q: str, domain: str) -> list[dict]:
    domain_items = items_metadata[items_metadata["domain"] == domain]
    if domain_items.empty or not q:
        return []

    choices = dict(zip(domain_items["item_id"], domain_items["title"]))
    matches = process.extract(q, choices, scorer=fuzz.WRatio, limit=10)
    matched_ids = [item_id for _, _score, item_id in matches]
    return _items_to_records(matched_ids)


def recommend(selected_items: list[str], target_domain: str, model_name: str, k: int = 10) -> list[dict]:
    recommender = recommenders[model_name]
    try:
        item_ids = recommender.recommend(selected_items, target_domain, k=k)
    except ValueError:
        return []
    return _items_to_records(item_ids)


# --- Gradio UI: same layout as app/frontend/app.py, callbacks call
# search()/recommend() directly instead of requests.get/post over HTTP. ---


def do_search(query, domain):
    if not query:
        return [], gr.update(choices=[], value=None)

    results = search(query, domain)
    choices = [(f"[{item['domain']}] {item['title']}", item["item_id"]) for item in results]
    return results, gr.update(choices=choices, value=None)


def add_to_cart(selected_item_id, search_results, cart):
    cart = dict(cart or {})
    if selected_item_id:
        match = next((r for r in (search_results or []) if r["item_id"] == selected_item_id), None)
        if match:
            cart[selected_item_id] = f"[{match['domain']}] {match['title']}"

    choices = [(title, item_id) for item_id, title in cart.items()]
    return cart, gr.update(choices=choices)


def remove_from_cart(to_remove, cart):
    cart = dict(cart or {})
    for item_id in to_remove or []:
        cart.pop(item_id, None)

    choices = [(title, item_id) for item_id, title in cart.items()]
    return cart, gr.update(choices=choices, value=[])


def _call_recommend(cart, target_domain, model_name, k):
    item_ids = list((cart or {}).keys())
    if not item_ids:
        return "Кошик порожній — додай хоча б один тайтл перед рекомендацією."

    results = recommend(item_ids, target_domain, model_name, k=int(k))
    if not results:
        return "Рекомендацій не знайдено (можливо, жоден обраний тайтл не відомий цій моделі)."

    return "\n".join(f"- **{item['title']}** ({item['domain']})" for item in results)


def get_recommendations(cart, target_domain, model_name, k):
    return _call_recommend(cart, target_domain, model_name, k)


def compare_recommendations(cart, target_domain, model_a, model_b, k):
    return (
        _call_recommend(cart, target_domain, model_a, k),
        _call_recommend(cart, target_domain, model_b, k),
    )


with gr.Blocks(title="Cross-domain Recommender") as demo:
    cart_state = gr.State({})
    search_results_state = gr.State([])

    gr.Markdown("# Крос-доменні рекомендації (books + movies)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Пошук і кошик")
            search_domain = gr.Radio(DOMAIN_CHOICES, value="books", label="Домен пошуку")
            search_box = gr.Textbox(label="Назва")
            search_button = gr.Button("Шукати")
            search_radio = gr.Radio(choices=[], label="Результати пошуку")
            add_button = gr.Button("Додати в кошик")
            cart_checkboxgroup = gr.CheckboxGroup(choices=[], label="Кошик (познач, щоб видалити)")
            remove_button = gr.Button("Видалити позначене")

        with gr.Column():
            gr.Markdown("## Рекомендації")
            target_domain = gr.Radio(DOMAIN_CHOICES, value="movies", label="Цільовий домен")
            k_slider = gr.Slider(1, 20, value=10, step=1, label="K")

            with gr.Tab("Рекомендувати"):
                model_dropdown = gr.Dropdown(MODEL_CHOICES, value="svd", label="Модель")
                recommend_button = gr.Button("Рекомендувати")
                recommend_output = gr.Markdown()

            with gr.Tab("Порівняти"):
                model_a_dropdown = gr.Dropdown(MODEL_CHOICES, value="svd", label="Модель A")
                model_b_dropdown = gr.Dropdown(MODEL_CHOICES, value="als", label="Модель B")
                compare_button = gr.Button("Порівняти")
                with gr.Row():
                    compare_output_a = gr.Markdown()
                    compare_output_b = gr.Markdown()

    search_button.click(
        do_search, inputs=[search_box, search_domain], outputs=[search_results_state, search_radio]
    )
    add_button.click(
        add_to_cart,
        inputs=[search_radio, search_results_state, cart_state],
        outputs=[cart_state, cart_checkboxgroup],
    )
    remove_button.click(
        remove_from_cart, inputs=[cart_checkboxgroup, cart_state], outputs=[cart_state, cart_checkboxgroup]
    )
    recommend_button.click(
        get_recommendations,
        inputs=[cart_state, target_domain, model_dropdown, k_slider],
        outputs=[recommend_output],
    )
    compare_button.click(
        compare_recommendations,
        inputs=[cart_state, target_domain, model_a_dropdown, model_b_dropdown, k_slider],
        outputs=[compare_output_a, compare_output_b],
    )


if __name__ == "__main__":
    demo.launch()
