import streamlit as st
from PIL import Image
import os

# Установка настроек страницы
st.set_page_config(page_title="Дашборд анализа данных", layout="wide")

# Заголовок страницы
st.title("📊 Дашборд анализа данных и моделирования")
st.markdown("""
На этой странице представлены основные результаты анализа данных и оценки моделей машинного обучения для задачи прогнозирования длительности поездки.
""")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "../Pictures"

# Функция для отображения изображения с подписью
def display_image_with_caption(image_path, caption):
    image = Image.open(os.path.join(image_dir, image_path))
    st.image(image, caption=caption, use_container_width=True)

# Создание сетки для изображений
col1, col2, col3 = st.columns(3)

# Размещение графиков в сетке
with col1:
    # Bagging.png
    display_image_with_caption(
        "Bagging.png",
        "3 дерева из Bagging Regressor\n"
    )

    # GradientBoosting.png
    display_image_with_caption(
        "GradientBoosting.png",
        "Первое дерево в ансамбле Gradient Boosting\n"
    )

    # Linear_Regression.png
    display_image_with_caption(
        "Linear_Regression.png",
        "Точность предсказаний модели линейной регрессии\n"
    )

with col2:
    # EDA1.png
    display_image_with_caption(
        "EDA1.png",
        "Данная схема показывает зависимость целевого признака от остальных признаков\n"
    )

    # EDA2.png
    display_image_with_caption(
        "EDA2.png",
        "Матрица Scatter Plot\n"
    )

    # Stacking.png
    display_image_with_caption(
        "Stacking.png",
        "Дерево решений в составе Stacking Regressor\n"
    )

with col3:
    # mlp.png
    display_image_with_caption(
        "mlp.png",
        "График потерь MLPRegressor\n"
    )
