# git remote add origin https://github.com/LuCryp/friends_finder.git
# https://friendsfinder-encrypted-lk.streamlit.app/

import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
from cryptography.fernet import Fernet
from io import StringIO
import os
import plotly.express as px
import tempfile

st.title("Znajdz znajomych")
env = os.environ

# Nazwy plików
MODEL_NAME = "welcome_survey_clustering_pipeline_v2.enc"
DATA = "welcome_survey_simple_v2.csv.enc"                # zaszyfrowany
CLUSTER_NAMES_AND_DESCRIPTIONS = "welcome_survey_cluster_names_and_descriptions_v2.json.enc"  # zaszyfrowany

if not st.session_state.get('fernet_key'):
    if 'FERNET_KEY' in env:
        st.session_state['fernet_key'] = env['FERNET_KEY']
    else:
        st.info('Dodaj klucz szyfrowania aplikacji (FERNET_KEY)')
        st.session_state['fernet_key'] = st.text_input(
            'Podaj klucz',
            type='password'
        )
        if st.session_state['fernet_key']:
            st.rerun()

if not st.session_state.get("fernet_key"):
    st.stop()  # dopóki nie ma klucza, nic się nie wykonuje

# --- Fernet ---
fernet = Fernet(st.session_state["fernet_key"].encode())

# --- Funkcje do odszyfrowania ---
def load_encrypted(path: str) -> bytes:
    with open(path, "rb") as f:
        return fernet.decrypt(f.read())

def get_model():
    # odszyfrowanie pliku do bytes
    raw = load_encrypted(MODEL_NAME)
    if not raw:
        raise ValueError("Odszyfrowany model jest pusty!")

    # Tworzymy tymczasowy plik .pkl z kontrolowaną nazwą
    tmp_path = os.path.join(tempfile.gettempdir(), "tmp_model")

    try:
        with open(tmp_path, "wb") as f:
            f.write(raw)

        # Wczytanie modelu z PyCaret
        model = load_model(tmp_path)

    finally:
        # Sprzątanie pliku – usuń, jeśli plik istnieje
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return model

@st.cache_data
def get_cluster_names_and_descriptions():
    raw = load_encrypted(CLUSTER_NAMES_AND_DESCRIPTIONS)
    return json.loads(raw.decode("utf-8"))

@st.cache_data
def get_all_participants():
    model = get_model()
    
    # CSV odszyfrowany w pamięci
    raw = load_encrypted(DATA)
    all_df = pd.read_csv(StringIO(raw.decode("utf-8")), sep=';')
    
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])
    
    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])


model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))
same_cluster_df_pl = same_cluster_df.rename(columns={
    "age": "Wiek",
    "edu_level": "Wykształcenie",
    "fav_animals": "Zwierzęta",
    "fav_place": "Miejsce",
    "gender": "Płeć",
    "Cluster": "Grupa"
})

st.dataframe(same_cluster_df_pl.sample(5), hide_index=True)


col = st.selectbox(
    "Wybierz cechę",
    ["Wiek", "Wykształcenie", "Zwierzęta", "Miejsce", "Płeć"]
)

counts = same_cluster_df_pl[col].value_counts()

fig = px.pie(values=counts.values, names=counts.index)
st.plotly_chart(fig)

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)


