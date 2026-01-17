
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from pathlib import Path

from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet



st.set_page_config(page_title="Análise SIH", layout="wide")
st.title("Análise e Previsão de Internações no SUS de SC")
FIGSIZE = (5, 2.4)



def normalize_text(text):
    if pd.isna(text):
        return text

    text = str(text)

    try:
        fixed = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        if fixed.count("Ã") < text.count("Ã"):
            text = fixed
    except Exception:
        pass

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    return text.strip().lower()


def read_csv_multi_encoding(path, sep=";", encodings=("utf-8", "latin1", "cp1252")):
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err

BASE_DIR = Path(__file__).resolve().parent

base_dados = "https://drive.google.com/uc?id=1ac0WCV0oVLyTnbzK6XbK-Nc8hQIm_Zfy&export=download"
base_cid = "https://drive.google.com/uc?id=13oKKjPD-n4EKwV6cMgTsGcxjm_gtFd1S&export=download"
base_ranking = "https://drive.google.com/uc?id=1lZVh1XDRMpOKnt1Uhj7Y4Bdt9Hy_3r67&export=download"


@st.cache_data
def carregar_dados():
    df = read_csv_multi_encoding(base_dados)
    df["DT_INTER"] = pd.to_datetime(df["DT_INTER"], errors="coerce")
    df["DIAG_PRINC"] = df["DIAG_PRINC"].astype(str).str.strip().str.lower()
    return df


@st.cache_data
def carregar_cids():
    cid = read_csv_multi_encoding(base_cid)
    cid["codigo"] = cid["codigo"].astype(str).str.strip().str.lower()
    cid["descricao"] = cid["descricao"].apply(normalize_text)
    cid["opcao"] = cid["codigo"] + " — " + cid["descricao"]
    return cid


@st.cache_data
def carregar_ranking():
    df = read_csv_multi_encoding(base_ranking)
    df["DIAG_PRINC"] = df["DIAG_PRINC"].astype(str).str.strip().str.lower()
    return df


dados = carregar_dados()
cid_dict = carregar_cids()



st.sidebar.header("Pesquisa da doença")

busca = st.sidebar.selectbox(
    "Digite o CID ou o nome da doença",
    options=cid_dict["opcao"].tolist(),
    index=None,
    placeholder="Ex: j18 ou pneumonia por microorganismo nao especificada"
)



if busca is None:
    st.subheader("Ranking de internações por CID no período 2019–2024")

    ranking_df = carregar_ranking()

    ranking = ranking_df["DIAG_PRINC"].value_counts().reset_index()
    ranking.columns = ["CID", "Total de internações"]
    ranking["CID"] = ranking["CID"].str.lower()

    ranking = ranking.merge(
        cid_dict[["codigo", "descricao"]],
        left_on="CID",
        right_on="codigo",
        how="left"
    ).drop(columns="codigo")

    ranking = ranking.rename(columns={"descricao": "Doença"})
    ranking = ranking[["CID", "Doença", "Total de internações"]]

    st.markdown(
        """
        <style>
        th, td { text-align: center; }
        td { white-space: normal; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(ranking.to_html(index=False), unsafe_allow_html=True)
    st.stop()



cid = busca.split(" — ")[0].strip().lower()

dados_cid = dados[
    (dados["DIAG_PRINC"] == cid) &
    (dados["DT_INTER"].dt.year.between(2019, 2024))
].copy()

dados_cid["ANO"] = dados_cid["DT_INTER"].dt.year
dados_cid["MES"] = dados_cid["DT_INTER"].dt.month


st.subheader("Média de internações por mês (2019–2024)")

media_mensal_hist = (
    dados_cid.groupby(["ANO", "MES"])
    .size()
    .groupby("MES")
    .mean()
    .reindex(range(1, 13), fill_value=0)
)

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(media_mensal_hist.index, media_mensal_hist.values, marker="o")
ax.set_xticks(range(1, 13))
ax.grid(True)

for x, y in zip(media_mensal_hist.index, media_mensal_hist.values):
    ax.text(x, y, f"{y:.1f}", fontsize=7, ha="center")

st.pyplot(fig)



serie_mensal = (
    dados_cid
    .set_index("DT_INTER")
    .resample("MS")
    .size()
    .astype(float)
)

serie_mensal = serie_mensal.reindex(
    pd.date_range("2019-01-01", "2024-12-01", freq="MS"),
    fill_value=0
)


# PREVISÕES (HOLT / HW / SARIMA / PROPHET)

h = 24
idx_futuro = pd.date_range("2025-01-01", "2026-12-01", freq="MS")

holt = Holt(serie_mensal).fit().forecast(h)
hw = ExponentialSmoothing(
    serie_mensal,
    trend="add",
    seasonal="add",
    seasonal_periods=12
).fit().forecast(h)

sarima = SARIMAX(
    serie_mensal,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False).get_forecast(steps=h).predicted_mean

# Prophet
df_prophet = pd.DataFrame({
    "ds": serie_mensal.index,
    "y": serie_mensal.values
})

m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=h, freq="MS")
forecast = m.predict(future)
prophet_vals = forecast.tail(h)["yhat"].values


prev = pd.DataFrame({
    "Holt": holt.values,
    "Holt-Winters": hw.values,
    "SARIMA": sarima.values,
    "Prophet": prophet_vals
}, index=idx_futuro)

prev["Média"] = prev.mean(axis=1)



st.subheader("Previsões mensais por modelo (2025–2026)")
st.dataframe(prev.round(1), use_container_width=True)


for ano, cor in [(2025, "red"), (2026, "orange")]:
    st.subheader(f"Previsão mensal {ano} (média dos modelos)")
    dados_ano = prev.loc[str(ano)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dados_ano.index.month, dados_ano["Média"], marker="o", color=cor)
    ax.set_xticks(range(1, 13))
    ax.grid(True)

    for x, y in zip(dados_ano.index.month, dados_ano["Média"]):
        ax.text(x, y, f"{y:.1f}", fontsize=7, ha="center")

    st.pyplot(fig)



st.subheader("Total anual + previsões")

total_real = serie_mensal.groupby(serie_mensal.index.year).sum()
total_2025 = prev.loc["2025"]["Média"].sum()
total_2026 = prev.loc["2026"]["Média"].sum()

anos = list(total_real.index) + [2025, 2026]
valores = list(total_real.values) + [total_2025, total_2026]

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(anos, valores, marker="o", linestyle="--")
ax.scatter(2025, total_2025, color="red", s=80)
ax.scatter(2026, total_2026, color="orange", s=80)
ax.set_xticks(anos)
ax.grid(True)

for x, y in zip(anos, valores):
    ax.text(x, y, f"{int(y)}", fontsize=7, ha="center")

st.pyplot(fig)



st.subheader("Variação percentual anual (%)")

serie_total = pd.Series(valores, index=anos)
variacao = serie_total.pct_change() * 100
variacao = variacao.dropna()

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.bar(
    variacao.index,
    variacao.values,
    color=["red" if a >= 2025 else "steelblue" for a in variacao.index]
)
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(True, axis="y")

for x, y in zip(variacao.index, variacao.values):
    ax.text(x, y, f"{y:.1f}%", fontsize=7, ha="center")

st.pyplot(fig)


# pip install -r requirements.txt
# streamlit run app_cid_streamlit_teste10.py