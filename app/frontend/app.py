"""Gradio MVP frontend. Talks to the FastAPI backend over HTTP only -- does not
import app.recommenders directly, to keep the same process boundary the real
deployment will have.
"""

import os

import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_CHOICES = ["svd", "als", "item2vec", "sasrec", "bert4rec"]
DOMAIN_CHOICES = ["books", "movies"]
# The backend returns SEARCH_FETCH_COUNT matches (app/api/main.py); only show
# this many at a time so the gallery can pull in the next candidate as each
# visible one gets clicked into the cart, instead of leaving a gap.
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


def _to_gallery(records: list[dict]) -> list[tuple[str, str]]:
    # No domain prefix in the caption -- gallery captions truncate, and the
    # prefix was eating the space needed to show the actual title.
    return [(r.get("image_url") or NO_IMAGE_PLACEHOLDER, r["title"]) for r in records]


def do_search(query, domain):
    if not query:
        return [], gr.update(value=[])

    try:
        response = requests.get(f"{API_URL}/search", params={"q": query, "domain": domain}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        gr.Warning(f"Пошук не вдався: {exc}")
        return [], gr.update(value=[])

    results = response.json()
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

    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"selected_items": item_ids, "target_domain": target_domain, "model": model_name, "k": int(k)},
            timeout=30,
        )
    except requests.RequestException as exc:
        gr.Warning(f"Помилка з'єднання з бекендом: {exc}")
        return [], ""

    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        gr.Warning(f"Помилка ({response.status_code}): {detail}")
        return [], ""

    results = response.json()
    if not results:
        gr.Warning("Рекомендацій не знайдено.")
        return [], ""

    text = "\n".join(f"- **{r['title']}** ({r['domain']})" for r in results)
    return _to_gallery(results), text


def get_recommendations(cart, target_domain, model_name, k):
    return _call_recommend(cart, target_domain, model_name, k)


def compare_recommendations(cart, target_domain, model_a, model_b, k):
    gallery_a, text_a = _call_recommend(cart, target_domain, model_a, k)
    gallery_b, text_b = _call_recommend(cart, target_domain, model_b, k)
    return gallery_a, text_a, gallery_b, text_b


with gr.Blocks(title="Cross-domain Recommender MVP") as demo:
    cart_state = gr.State({})
    search_results_state = gr.State([])

    gr.Markdown("# Книго-фільмо-чаклун ")

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
    demo.launch(theme=gr.themes.Soft())
