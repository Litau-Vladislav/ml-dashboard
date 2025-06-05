import streamlit as st

# Убедись, что set_page_config первая!
st.set_page_config(page_title="Информация о студенте", layout="wide")

# Заголовок
st.markdown("<h1 style='text-align: center;'>👤 Информация о студенте</h1>", unsafe_allow_html=True)

# Основная информация
st.markdown("""
<p style='font-size: 20px;'>
<strong>ФИО:</strong> Литау Владислав Денисович<br>
<strong>Группа:</strong> МО-231<br>
<strong>Тема РГР:</strong> Разработка Web-приложения для инференса моделей ML и анализа данных
</p>
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 16px;'>Омск 2025</p>",
            unsafe_allow_html=True)