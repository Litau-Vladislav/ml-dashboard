import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from catboost import CatBoostRegressor
from io import StringIO
import time

# Настройки страницы
st.set_page_config(
    page_title="Прогнозирование цен на недвижимость",
    page_icon="🏠",
    layout="wide"
)

# Заголовок
st.title("🏠 Прогнозирование цен на недвижимость в King County")
st.markdown("---")

# 1. Загрузка моделей
@st.cache_resource  # Кэширование моделей для ускорения
def load_models():
    models = {}
    try:
        # Загрузка всех 6 моделей
        with open('models/linear_regression.pkl', 'rb') as f:
            models['Linear Regression'] = pickle.load(f)
        
        models['Gradient Boosting'] = joblib.load('models/gradient_boosting.joblib')
        models['Bagging'] = joblib.load('models/bagging.joblib')
        models['Stacking'] = joblib.load('models/stacking.joblib')
        
        models['CatBoost'] = CatBoostRegressor().load_model('models/catboost.cbm')
        models['Neural Network'] = joblib.load('models/mlp.joblib')
        
        st.success("Все модели успешно загружены!")
        return models
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        return None

models = load_models()

if not models:
    st.stop()

# 2. Выбор способа ввода данных
st.header("1. Способ ввода данных")
input_method = st.radio(
    "Выберите способ ввода данных:",
    ["Загрузить CSV файл", "Ручной ввод"],
    horizontal=True
)

X_input = None

if input_method == "Загрузить CSV файл":
    # 3. Загрузка файла
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл с данными для прогнозирования", 
        type=['csv']
    )
    
    if uploaded_file:
        try:
            X_input = pd.read_csv(uploaded_file)
            X_input = X_input.drop('price', axis=1)

            # Удаляем столбец City, если он есть
            if 'City' in X_input.columns:
                X_input = X_input.drop('City', axis=1)
            
            st.success("Файл успешно загружен!")
            st.dataframe(X_input.head())
            
            # Проверка столбцов
            expected_cols = [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'condition', 'grade', 'sqft_above', 'sqft_basement', 
                'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15',
                'day_posted', 'month_posted', 'year_posted'
            ]
            
            missing_cols = set(expected_cols) - set(X_input.columns)
            if missing_cols:
                st.error(f"Отсутствуют необходимые столбцы: {missing_cols}")
                X_input = None
            
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            X_input = None

else:
    # 4. Форма для ручного ввода
    st.header("2. Введите параметры недвижимости")
    
    # Создаем колонки для лучшего отображения
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bedrooms = st.number_input("Количество спален", min_value=1, max_value=15, value=3)
        bathrooms = st.number_input("Количество ванных", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        sqft_living = st.number_input("Жилая площадь (кв.футы)", min_value=300, max_value=15000, value=2000)
    
    with col2:
        sqft_lot = st.number_input("Площадь участка (кв.футы)", min_value=500, max_value=1000000, value=5000)
        floors = st.number_input("Количество этажей", min_value=1.0, max_value=5.0, value=1.5, step=0.5)
        condition = st.selectbox("Состояние", options=[1, 2, 3, 4, 5], index=2)
    
    with col3:
        grade = st.selectbox("Качество строительства", options=list(range(1, 14)), index=7)
        sqft_above = st.number_input("Площадь над землей (кв.футы)", min_value=300, max_value=10000, value=1500)
        sqft_basement = st.number_input("Площадь подвала (кв.футы)", min_value=0, max_value=5000, value=500)
    
    # Вторая строка параметров
    col4, col5, col6 = st.columns(3)
    
    with col4:
        yr_built = st.number_input("Год постройки", min_value=1900, max_value=2023, value=1990)
        yr_renovated = st.number_input("Год ремонта (0 если не было)", min_value=0, max_value=2023, value=0)
    
    with col5:
        sqft_living15 = st.number_input("Средняя жилая площадь соседей (кв.футы)", min_value=300, max_value=10000, value=2000)
        sqft_lot15 = st.number_input("Средняя площадь участка соседей (кв.футы)", min_value=500, max_value=1000000, value=5000)
    
    with col6:
        day_posted = st.number_input("День публикации", min_value=1, max_value=31, value=15)
        month_posted = st.number_input("Месяц публикации", min_value=1, max_value=12, value=6)
        year_posted = st.number_input("Год публикации", min_value=1900, max_value=2023, value=2015)
    
    # Собираем ввод в DataFrame
    X_input = pd.DataFrame([[
        bedrooms, bathrooms, sqft_living, sqft_lot, floors,
        condition, grade, sqft_above, sqft_basement,
        yr_built, yr_renovated, sqft_living15, sqft_lot15,
        day_posted, month_posted, year_posted
    ]], columns=[
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'condition', 'grade', 'sqft_above', 'sqft_basement',
        'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15',
        'day_posted', 'month_posted', 'year_posted'
    ])
    
    st.success("Данные готовы к прогнозированию!")
    st.dataframe(X_input)

# 5. Прогнозирование
if X_input is not None:
    st.header("3. Выполнение прогноза")
    
    # Выбор модели
    model_name = st.selectbox(
        "Выберите модель для прогнозирования",
        options=list(models.keys())
    )
    
    # Дополнительные параметры
    show_details = st.checkbox("Показать детали прогноза")
    
    if st.button("Выполнить прогноз"):
        with st.spinner("Выполняется прогнозирование..."):
            try:
                model = models[model_name]
                
                # Прогноз
                start_time = time.time()
                prediction = abs(model.predict(X_input))
                prediction_time = time.time() - start_time
                
                # Отображение результатов
                st.success("Прогноз успешно выполнен!")
                
                # Красивый вывод результата
                st.markdown(f"""
                ### 🎯 Результат прогноза
                **Модель:** {model_name}  
                **Время выполнения:** {prediction_time:.4f} секунд  
                **Прогнозируемая цена дома:**  
                """)
                
                # Стилизованный вывод
                st.markdown(f"""
                <div style="
                    background: #f0f2f6;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    text-align: center;
                ">
                    <h2 style="color: #ff4b4b;">${prediction[0]:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Детали прогноза
                if show_details:
                    st.subheader("Детали прогноза")
                    
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("**Важность признаков:**")
                        importances = pd.DataFrame({
                            'Признак': X_input.columns,
                            'Важность': model.feature_importances_
                        }).sort_values('Важность', ascending=False)
                        
                        st.dataframe(importances)
                    
                    st.markdown("**Использованные данные:**")
                    st.dataframe(X_input)
                
            except Exception as e:
                st.error(f"Ошибка при прогнозировании: {e}")

# 6. Примеры прогнозов
st.markdown("---")
st.header("📌 Примеры прогнозов")

example_tab1, example_tab2 = st.tabs(["Корректные данные", "Данные с выбросами"])

with example_tab1:
    st.markdown("""
    ### Пример 1: Стандартные данные
    ```python
    bedrooms = 3
    bathrooms = 2.5
    sqft_living = 2500
    sqft_lot = 6000
    floors = 2.0
    condition = 3
    grade = 8
    sqft_above = 2000
    sqft_basement = 500
    yr_built = 1995
    yr_renovated = 2010
    sqft_living15 = 2400
    sqft_lot15 = 6500
    day_posted = 15
    month_posted = 5
    year_posted = 2015
    ```
    """)
    st.markdown("**Ожидаемый результат:** Цена в диапазоне $400,000 - $600,000")

with example_tab2:
    st.markdown("""
    ### Пример 2: Данные с выбросами
    ```python
    bedrooms = 20
    bathrooms = 10.0
    sqft_living = 50000
    sqft_lot = 1000000
    floors = 10.0
    condition = 5
    grade = 13
    sqft_above = 50000
    sqft_basement = 0
    yr_built = 2025
    yr_renovated = 0
    sqft_living15 = 50000
    sqft_lot15 = 1000000
    day_posted = 32
    month_posted = 13
    year_posted = 2025
    ```
    """)
    st.markdown("**Ожидаемый результат:** Модель может показать некорректный прогноз")

st.markdown("---")
st.info("ℹ️ Для получения точных прогнозов используйте данные в том же формате, что и при обучении моделей.")
