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
# Fetch more matches than we display so the gallery can pull in the next
# candidate as each visible one gets clicked into the cart, instead of
# leaving a gap or requiring a re-search.
SEARCH_FETCH_COUNT = 30
SEARCH_DISPLAY_COUNT = 10

# Shown in place of a cover when items_metadata has no image_url for an item
# -- avoids depending on an external placeholder service at request time.
NO_IMAGE_PLACEHOLDER = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='300'%3E"
    "%3Crect width='200' height='300' fill='%23ddd'/%3E"
    "%3Ctext x='50%25' y='50%25' text-anchor='middle' fill='%23888' "
    "font-size='16' font-family='sans-serif'%3Eno image%3C/text%3E%3C/svg%3E"
)

MODEL_LOADERS = {
    "svd": FoldInRecommender.load,
    "als": FoldInRecommender.load,
    "item2vec": Item2VecRecommender.load,
    "sasrec": SASRecRecommender.load,
    "bert4rec": BERT4RecRecommender.load,
}

artifacts_dir = snapshot_download(repo_id=ARTIFACTS_REPO, repo_type="dataset")
items_metadata = pd.read_parquet(f"{artifacts_dir}/items_metadata.parquet")
# The raw scraped metadata can have more than one row for the same item_id
# (e.g. heavily-reviewed titles like Harry Potter) -- set_index/reindex below
# raises ValueError on a non-unique index, so collapse duplicates up front.
items_metadata = items_metadata.drop_duplicates(subset="item_id", keep="first")
recommenders = {name: load(f"{artifacts_dir}/{name}") for name, load in MODEL_LOADERS.items()}


def _items_to_records(item_ids: list[str]) -> list[dict]:
    results = items_metadata.set_index("item_id").reindex(item_ids).dropna(subset=["domain"])
    results = results.reset_index()
    results["title"] = results["title"].replace("", pd.NA).fillna(results["item_id"])
    records = results[["item_id", "title", "domain", "image_url"]].to_dict("records")
    # image_url is missing as NaN (float), not None, for items without a cover
    # -- NaN is truthy in Python, so downstream "image_url or placeholder"
    # checks need a real None. Normalized here, after to_dict, because
    # reassigning a pandas column with a mix of None/str can get its dtype
    # re-inferred as a nullable string type whose missing marker round-trips
    # back to a bare NaN float on the next to_dict() call.
    for record in records:
        if not isinstance(record["image_url"], str) or not record["image_url"]:
            record["image_url"] = None
    return records


def _to_gallery(records) -> list[tuple[str, str]]:
    # No domain prefix in the caption -- gallery captions truncate, and the
    # prefix was eating the space needed to show the actual title.
    return [(r["image_url"] or NO_IMAGE_PLACEHOLDER, r["title"]) for r in records]


def search(q: str, domain: str) -> list[dict]:
    domain_items = items_metadata[items_metadata["domain"] == domain]
    if domain_items.empty or not q:
        return []

    choices = dict(zip(domain_items["item_id"], domain_items["title"]))
    matches = process.extract(q, choices, scorer=fuzz.WRatio, limit=SEARCH_FETCH_COUNT)
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
        return [], gr.update(value=[])

    results = search(query, domain)
    return results, gr.update(value=_to_gallery(results[:SEARCH_DISPLAY_COUNT]))


def add_to_cart_on_select(evt: gr.SelectData, search_results, cart):
    cart = dict(cart or {})
    search_results = list(search_results or [])
    if evt.index < len(search_results):
        item = search_results.pop(evt.index)
        cart[item["item_id"]] = item
    return (
        cart,
        gr.update(value=_to_gallery(cart.values())),
        search_results,
        gr.update(value=_to_gallery(search_results[:SEARCH_DISPLAY_COUNT])),
    )


def sync_cart_after_gallery_edit(cart_gallery_value, cart):
    # The cart gallery is interactive, so Gradio already lets the user remove
    # a thumbnail with the built-in per-item "x" -- this just syncs cart_state
    # to match afterwards. Diffing by caption (title) because the gallery's
    # own preprocess() hands back a locally-cached file path for the image,
    # not the original image_url, so that side can't be compared directly.
    cart = dict(cart or {})
    remaining_titles = {caption for _media, caption in (cart_gallery_value or [])}
    for item_id, item in list(cart.items()):
        if item["title"] not in remaining_titles:
            del cart[item_id]
    return cart


def clear_cart():
    return {}, gr.update(value=[])


def _call_recommend(cart, target_domain, model_name, k):
    item_ids = list((cart or {}).keys())
    if not item_ids:
        gr.Warning("Кошик порожній — додай хоча б один тайтл перед рекомендацією.")
        return [], ""

    results = recommend(item_ids, target_domain, model_name, k=int(k))
    if not results:
        gr.Warning("Рекомендацій не знайдено (можливо, жоден обраний тайтл не відомий цій моделі).")
        return [], ""

    text = "\n".join(f"- **{r['title']}** ({r['domain']})" for r in results)
    return _to_gallery(results), text


def get_recommendations(cart, target_domain, model_name, k):
    return _call_recommend(cart, target_domain, model_name, k)


def compare_recommendations(cart, target_domain, model_a, model_b, k):
    gallery_a, text_a = _call_recommend(cart, target_domain, model_a, k)
    gallery_b, text_b = _call_recommend(cart, target_domain, model_b, k)
    return gallery_a, text_a, gallery_b, text_b


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
            search_gallery = gr.Gallery(
                label="Результати пошуку (клікни, щоб додати в кошик)",
                columns=5,
                height=260,
                object_fit="contain",
                allow_preview=False,
                interactive=False,
            )
            clear_cart_button = gr.Button("Очистити кошик")
            cart_gallery = gr.Gallery(
                label="Кошик (клікни ×, щоб видалити)",
                columns=5,
                height=220,
                object_fit="contain",
                allow_preview=False,
                interactive=True,
            )

        with gr.Column():
            gr.Markdown("## Рекомендації")
            target_domain = gr.Radio(DOMAIN_CHOICES, value="movies", label="Цільовий домен")
            k_slider = gr.Slider(1, 20, value=10, step=1, label="K")

            with gr.Tab("Рекомендувати"):
                model_dropdown = gr.Dropdown(MODEL_CHOICES, value="svd", label="Модель")
                recommend_button = gr.Button("Рекомендувати")
                recommend_gallery = gr.Gallery(
                    columns=5, height=260, object_fit="contain", allow_preview=False, interactive=False
                )
                recommend_text = gr.Markdown()

            with gr.Tab("Порівняти"):
                model_a_dropdown = gr.Dropdown(MODEL_CHOICES, value="svd", label="Модель A")
                model_b_dropdown = gr.Dropdown(MODEL_CHOICES, value="als", label="Модель B")
                compare_button = gr.Button("Порівняти")
                with gr.Row():
                    with gr.Column():
                        compare_gallery_a = gr.Gallery(
                            label="Модель A", columns=3, height=260, object_fit="contain",
                            allow_preview=False, interactive=False,
                        )
                        compare_text_a = gr.Markdown()
                    with gr.Column():
                        compare_gallery_b = gr.Gallery(
                            label="Модель B", columns=3, height=260, object_fit="contain",
                            allow_preview=False, interactive=False,
                        )
                        compare_text_b = gr.Markdown()

    search_button.click(
        do_search, inputs=[search_box, search_domain], outputs=[search_results_state, search_gallery]
    )
    search_gallery.select(
        add_to_cart_on_select,
        inputs=[search_results_state, cart_state],
        outputs=[cart_state, cart_gallery, search_results_state, search_gallery],
    )
    cart_gallery.change(
        sync_cart_after_gallery_edit, inputs=[cart_gallery, cart_state], outputs=[cart_state]
    )
    clear_cart_button.click(clear_cart, outputs=[cart_state, cart_gallery])
    recommend_button.click(
        get_recommendations,
        inputs=[cart_state, target_domain, model_dropdown, k_slider],
        outputs=[recommend_gallery, recommend_text],
    )
    compare_button.click(
        compare_recommendations,
        inputs=[cart_state, target_domain, model_a_dropdown, model_b_dropdown, k_slider],
        outputs=[compare_gallery_a, compare_text_a, compare_gallery_b, compare_text_b],
    )


if __name__ == "__main__":
    # HF Spaces doesn't set PORT and is fine with the default bind; Cloud Run
    # (and most container hosts) require listening on 0.0.0.0:$PORT.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
