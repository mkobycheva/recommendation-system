"""Gradio MVP frontend. Talks to the FastAPI backend over HTTP only -- does not
import app.recommenders directly, to keep the same process boundary the real
deployment will have.
"""

import os

import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_CHOICES = ["svd", "als", "item2vec"]
DOMAIN_CHOICES = ["books", "movies"]


def do_search(query, domain):
    if not query:
        return [], gr.update(choices=[], value=None)

    try:
        response = requests.get(f"{API_URL}/search", params={"q": query, "domain": domain}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        gr.Warning(f"Пошук не вдався: {exc}")
        return [], gr.update(choices=[], value=None)

    results = response.json()
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

    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"selected_items": item_ids, "target_domain": target_domain, "model": model_name, "k": int(k)},
            timeout=30,
        )
    except requests.RequestException as exc:
        return f"Помилка з'єднання з бекендом: {exc}"

    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        return f"Помилка ({response.status_code}): {detail}"

    results = response.json()
    if not results:
        return "Рекомендацій не знайдено."

    return "\n".join(f"- **{item['title']}** ({item['domain']})" for item in results)


def get_recommendations(cart, target_domain, model_name, k):
    return _call_recommend(cart, target_domain, model_name, k)


def compare_recommendations(cart, target_domain, model_a, model_b, k):
    return (
        _call_recommend(cart, target_domain, model_a, k),
        _call_recommend(cart, target_domain, model_b, k),
    )


with gr.Blocks(title="Cross-domain Recommender MVP") as demo:
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
