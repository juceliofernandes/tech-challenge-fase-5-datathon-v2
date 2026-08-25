# -*- coding: utf-8 -*-
"""Streamlit — Passos Mágicos | Risco de defasagem (PEDE)

Este app consome o artefato gerado pelo notebook 02_modelo_preditivo_risco.ipynb:
  outputs/modelo/modelo_risco_defasagem.joblib

Se o arquivo ainda não existir (você abriu o app antes de rodar o notebook),
ele tenta treinar um modelo rápido com a planilha DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx
e avisa no topo. Para deploy no Community Cloud, mantenha o .joblib no repositório.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Pacotes de modelagem (presentes em requirements.txt)
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTEFATO = ROOT / "outputs" / "modelo" / "modelo_risco_defasagem.joblib"
CARTAO = ROOT / "outputs" / "modelo" / "cartao_do_modelo.md"
PLANILHA_CANDIDATOS = [
    ROOT / "DATATHON" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx",
    ROOT / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx",
]
for alt in ROOT.rglob("BASE DE DADOS PEDE 2024 - DATATHON.xlsx"):
    if alt not in PLANILHA_CANDIDATOS:
        PLANILHA_CANDIDATOS.append(alt)

NUM = [
    "idade",
    "ano_ingresso",
    "anos_na_pm",
    "fase_num",
    "inde",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipv",
    "ian",
    "defasagem",
    "risco_defasagem_atual",
    "gap_iaa_ida",
    "gap_ieg_ida",
    "media_indicadores",
    "indicadores_baixos_qtd",
]
CAT = ["fase_grupo"]

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Passos Mágicos — Risco de defasagem", layout="wide", page_icon="📚")


def localizar_planilha() -> Path | None:
    for p in PLANILHA_CANDIDATOS:
        if p.exists():
            return p
    hits = list(ROOT.rglob("BASE DE DADOS PEDE 2024 - DATATHON.xlsx"))
    return hits[0] if hits else None


def como_texto(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def corrigir_idade(serie: pd.Series) -> pd.Series:
    def um(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, pd.Timestamp):
            return float(v.dayofyear) if v.year in (1899, 1900) else np.nan
        n = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        return float(n) if pd.notna(n) else np.nan

    idade = serie.map(um).astype(float)
    return idade.where((idade >= 4) & (idade <= 30))


def fase_para_numero(v) -> float:
    if pd.isna(v):
        return np.nan
    t = str(v).strip().upper()
    if t in {"0", "ALFA", "ALF", "FASE 0", "FASE ALFA"}:
        return 0.0
    m = re.search(r"(\d+)", t)
    return float(m.group()) if m else np.nan


def primeira_coluna(df: pd.DataFrame, candidatos: list[str]) -> pd.Series:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for nome in candidatos:
        if nome.lower() in cols_lower:
            return df[cols_lower[nome.lower()]]
        for c in df.columns:
            if str(c).strip().lower() == nome.lower():
                return df[c]
    for c in df.columns:
        nl = str(c).lower()
        for nome in candidatos:
            if nome.lower() in nl:
                return df[c]
    return pd.Series(np.nan, index=df.index)


def padronizar_aba(bruto: pd.DataFrame, nome_aba: str) -> pd.DataFrame:
    ano = int(re.search(r"(20\d{2})", nome_aba).group(1))
    sufixo = str(ano)[-2:]
    out = pd.DataFrame(
        {
            "ra": primeira_coluna(bruto, ["RA"]),
            "fase": primeira_coluna(bruto, ["Fase"]),
            "idade": primeira_coluna(bruto, ["Idade", f"Idade {sufixo}", "Idade 22"]),
            "ano_ingresso": primeira_coluna(bruto, ["Ano ingresso"]),
            "iaa": primeira_coluna(bruto, ["IAA"]),
            "ieg": primeira_coluna(bruto, ["IEG"]),
            "ips": primeira_coluna(bruto, ["IPS"]),
            "ida": primeira_coluna(bruto, ["IDA"]),
            "ipv": primeira_coluna(bruto, ["IPV"]),
            "ian": primeira_coluna(bruto, ["IAN"]),
            "defasagem": primeira_coluna(bruto, ["Defasagem", "Defas"]),
        }
    )
    out["inde"] = primeira_coluna(bruto, [f"INDE {ano}", f"INDE {sufixo}", "INDE 22", "INDE 23"])
    out["ano"] = ano
    out["ra"] = como_texto(out["ra"])
    out["fase"] = como_texto(out["fase"])
    out["idade"] = corrigir_idade(out["idade"])
    for c in ["ano_ingresso", "inde", "iaa", "ieg", "ips", "ida", "ipv", "ian", "defasagem"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["fase_num"] = out["fase"].map(fase_para_numero)
    out["anos_na_pm"] = out["ano"] - out["ano_ingresso"]
    out.loc[out["anos_na_pm"] < 0, "anos_na_pm"] = np.nan
    out["risco_defasagem_atual"] = (out["defasagem"] < 0).astype(float)
    return out


def engenharia(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["gap_iaa_ida"] = d["iaa"] - d["ida"]
    d["gap_ieg_ida"] = d["ieg"] - d["ida"]
    inds = ["iaa", "ieg", "ips", "ida", "ipv", "ian"]
    d["media_indicadores"] = d[inds].mean(axis=1)
    disponiveis = d[inds].notna().sum(axis=1)
    baixos = d[inds].lt(6).sum(axis=1)
    d["indicadores_baixos_qtd"] = baixos.where(disponiveis > 0, np.nan)
    d["fase_grupo"] = np.where(
        d["fase_num"] == 0, "ALFA", d["fase_num"].map(lambda x: str(int(x)) if pd.notna(x) else "NI")
    )
    return d


def treinar_modelo_completo(planilha: Path) -> dict:
    """Replica o pipeline do notebook 02: split temporal, 4 modelos, CV (PR-AUC),
    limiar F2 e modelo de produção em todos os pares rotulados."""
    partes = [padronizar_aba(pd.read_excel(planilha, sheet_name=a), a) for a in ["PEDE2022", "PEDE2023", "PEDE2024"]]
    painel = pd.concat(partes, ignore_index=True)
    feats = engenharia(painel)
    blocos = []
    for origem in sorted(feats["ano"].unique())[:-1]:
        hoje = feats.loc[feats["ano"] == origem].copy()
        futuro = feats.loc[feats["ano"] == origem + 1, ["ra", "defasagem"]].copy()
        futuro["risco_proximo_ano"] = (futuro["defasagem"] < 0).astype(int)
        j = hoje.merge(futuro[["ra", "risco_proximo_ano"]], on="ra", how="inner", validate="one_to_one")
        j["ano_origem"] = origem
        blocos.append(j)
    pares = pd.concat(blocos, ignore_index=True)
    COLS = NUM + CAT
    treino = pares[pares["ano_origem"] == 2022]
    teste = pares[pares["ano_origem"] == 2023]
    X_tr, y_tr = treino[COLS], treino["risco_proximo_ano"]
    X_te, y_te = teste[COLS], teste["risco_proximo_ano"]

    def prep():
        num = SkPipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
        cat = SkPipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
        return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])

    candidatos = {
        "Regressão Logística": LogisticRegression(max_iter=2500, C=0.7, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    melhor_nome, melhor_score, melhor_thr = None, -1.0, 0.5
    ajustados = {}
    for nome, est in candidatos.items():
        passos = [("prep", prep())]
        if nome != "HistGradientBoosting":
            passos.append(("bal", RandomOverSampler(random_state=42)))
        passos.append(("modelo", est))
        pipe = ImbPipeline(passos)
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        pr = average_precision_score(y_tr, oof)
        prec, rec, thr = precision_recall_curve(y_tr, oof)
        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-12)
        t = float(thr[int(np.nanargmax(f2[:-1]))]) if len(thr) else 0.5
        ajustados[nome] = clone(pipe).fit(X_tr, y_tr)
        if pr > melhor_score:
            melhor_score, melhor_nome, melhor_thr = pr, nome, t

    pipe_final = ajustados[melhor_nome]
    p_te = pipe_final.predict_proba(X_te)[:, 1]
    metricas = {
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "precision": float(precision_score(y_te, (p_te >= melhor_thr).astype(int), zero_division=0)),
        "recall": float(recall_score(y_te, (p_te >= melhor_thr).astype(int), zero_division=0)),
        "f1": float(f1_score(y_te, (p_te >= melhor_thr).astype(int), zero_division=0)),
        "threshold": float(melhor_thr),
    }
    # modelo de produção em todos os pares rotulados (usa história conhecida)
    pipe_prod = clone(ajustados[melhor_nome]).fit(pares[COLS], pares["risco_proximo_ano"])
    prep_step = pipe_prod.named_steps["prep"]
    modelo = pipe_prod.named_steps["modelo"]
    nomes = pd.Series(prep_step.get_feature_names_out()).str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    vals = modelo.feature_importances_ if hasattr(modelo, "feature_importances_") else np.abs(modelo.coef_).ravel()
    imp = pd.DataFrame({"variavel": nomes, "importancia_global": vals}).sort_values("importancia_global", ascending=False)
    return {
        "pipeline": pipe_prod,
        "threshold": float(melhor_thr),
        "feature_columns": COLS,
        "numerical_features": NUM,
        "categorical_features": CAT,
        "model_name": melhor_nome,
        "test_metrics": metricas,
        "global_importances": imp.head(15).to_dict(orient="records"),
        "target_definition": "Defasagem < 0 no PEDE seguinte",
        "reference_year": 2024,
    }


@st.cache_resource(show_spinner=False)
def carregar_artefato():
    # 1) tenta o artefato salvo (pode falhar por versão de numpy/sklearn)
    aviso = None
    if ARTEFATO.exists():
        try:
            pacote = joblib.load(ARTEFATO)
            if isinstance(pacote, dict) and "pipeline" in pacote and "feature_columns" in pacote:
                return pacote, None
        except Exception as exc:  # pragma: no cover
            aviso = f"Não consegui carregar o modelo salvo ({exc}). Vou treinar um na inicialização."
    # 2) fallback: treina o pipeline completo com a planilha
    loc = localizar_planilha()
    if loc is None or not loc.exists():
        return None, "Planilha ausente. Envie DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx e rode o app novamente."
    try:
        pacote = treinar_modelo_completo(loc)
        return pacote, aviso or "Modelo treinado na inicialização (modelo salvo indisponível). Métricas no painel lateral."
    except Exception as exc:  # pragma: no cover
        return None, f"Falha no treino: {exc}"


PACOTE, AVISO = carregar_artefato()

# ---------------------------------------------------------------------------
st.title("📚 Passos Mágicos — Predição de risco de defasagem")
st.caption("PEDE 2022–2024 · O modelo prevê **Defasagem < 0 no PEDE seguinte** com indicadores do ano atual. Uso como triagem — não decide bolsa ou desligamento sozinho.")

if AVISO:
    st.warning(AVISO)
if PACOTE is None:
    st.stop()

PIPE = PACOTE["pipeline"]
THR = float(PACOTE.get("threshold", 0.5))
MODELO_NOME = PACOTE.get("model_name", "modelo")
METRICAS = PACOTE.get("test_metrics", {})

with st.sidebar:
    st.header("Como usar")
    st.markdown(
        "- Preencha os **indicadores do ano atual** na aba *Predição individual*.\n"
        "- Ou envie um **CSV** com as mesmas colunas na aba *Lote*.\n"
        "- O score é uma **probabilidade**; o limiar vem do treino."
    )
    st.divider()
    st.metric("Modelo em produção", MODELO_NOME)
    st.metric("Limiar (treino, F2)", f"{THR:.3f}")
    if METRICAS:
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC (teste 2023→2024)", f"{METRICAS.get('roc_auc', 0):.3f}")
        c2.metric("PR-AUC (teste)", f"{METRICAS.get('pr_auc', 0):.3f}")
        c3, c4 = st.columns(2)
        c3.metric("Recall", f"{METRICAS.get('recall', 0):.3f}")
        c4.metric("Precisão", f"{METRICAS.get('precision', 0):.3f}")
    if CARTAO.exists():
        with st.expander("Cartão do modelo"):
            st.markdown(CARTAO.read_text(encoding="utf-8"))
    st.divider()
    st.caption("Dicionário: IAN adequação de nível · IDA desempenho · IEG engajamento · IAA autoavaliação · IPS psicossocial · IPV ponto de virada · INDE índice geral.")

tab_ind, tab_lote, tab_sobre = st.tabs(["Predição individual", "Lote (CSV)", "Sobre / Deploy"])

with tab_ind:
    st.subheader("Entrada — indicadores do ano atual (t)")
    c1, c2, c3 = st.columns(3)
    fase_sel = c1.selectbox("Fase (grupo) — use ALFA para 1º/2º ano", ["ALFA", "1", "2", "3", "4", "5", "6", "7"], index=2)
    fase_num_map = {"ALFA": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "6": 6.0, "7": 7.0}
    idade = c2.number_input("Idade", 5, 30, 12, step=1)
    ano_ing = c3.number_input("Ano de ingresso", 2015, 2024, 2021, step=1)
    anos_pm = 2024 - int(ano_ing)

    def sli(label, default=7.0, key=None):
        return st.slider(label, 0.0, 10.0, float(default), 0.1, key=key)

    r1, r2, r3 = st.columns(3)
    iaa = r1.slider("IAA — autoavaliação", 0.0, 10.0, 8.5, 0.1)
    ieg = r2.slider("IEG — engajamento", 0.0, 10.0, 7.8, 0.1)
    ips = r3.slider("IPS — psicossocial", 0.0, 10.0, 6.5, 0.1)
    r4, r5, r6 = st.columns(3)
    ida = r4.slider("IDA — desempenho (média Mat/Port/Ing)", 0.0, 10.0, 6.2, 0.1)
    ipv = r5.slider("IPV — ponto de virada", 0.0, 10.0, 7.3, 0.1)
    ian = r6.slider("IAN — adequação de nível", 2.5, 10.0, 5.0, 0.5)
    r7, r8, r9 = st.columns(3)
    inde = r7.slider("INDE — índice geral", 0.0, 10.0, 7.0, 0.1)
    defas = r8.selectbox("Defasagem (fase efetiva − ideal)", [-3, -2, -1, 0, 1, 2], index=2, help="Negativo = atrasado")
    st.caption("Se a fase ideal não for conhecida, use defasagem 0 e ajuste o IAN.")

    gap_iaa_ida = iaa - ida
    gap_ieg_ida = ieg - ida
    media_inds = float(np.mean([iaa, ieg, ips, ida, ipv, ian]))
    baixos = int(sum(v < 6 for v in [iaa, ieg, ips, ida, ipv, ian]))

    linha = pd.DataFrame(
        [
            {
                "idade": float(idade),
                "ano_ingresso": float(ano_ing),
                "anos_na_pm": float(anos_pm),
                "fase_num": float(fase_num_map[fase_sel]),
                "inde": float(inde),
                "iaa": float(iaa),
                "ieg": float(ieg),
                "ips": float(ips),
                "ida": float(ida),
                "ipv": float(ipv),
                "ian": float(ian),
                "defasagem": float(defas),
                "risco_defasagem_atual": float(1 if defas < 0 else 0),
                "gap_iaa_ida": float(gap_iaa_ida),
                "gap_ieg_ida": float(gap_ieg_ida),
                "media_indicadores": float(media_inds),
                "indicadores_baixos_qtd": float(baixos),
                "fase_grupo": str(fase_sel),
            }
        ]
    )

    if st.button("Calcular risco", type="primary", use_container_width=True):
        prob = float(PIPE.predict_proba(linha[PACOTE["feature_columns"]])[:, 1][0])
        risco = prob >= THR
        colA, colB, colC = st.columns([1.2, 1, 1])
        colA.metric("Probabilidade de defasagem no próximo PEDE", f"{prob:.1%}", delta=f"limiar {THR:.2f}")
        colB.metric("Classificação (limiar do treino)", "EM RISCO" if risco else "SEM RISCO")
        colC.metric("Defasagem atual", "Sim" if defas < 0 else "Não")

        # barra
        st.progress(min(1.0, prob))
        if risco:
            st.error("**Sinalização:** acima do limiar. Priorize escuta pedagógica/psicológica e plano de acompanhamento. O modelo não vê contexto familiar ou de saúde.")
        else:
            st.success("Abaixo do limiar. Mantenha acompanhamento de rotina.")

        with st.expander("Ver features derivadas usadas"):
            st.dataframe(linha.T.rename(columns={0: "valor"}).style.format("{:.3f}", subset=pd.IndexSlice[NUM, "valor"]))

        # importâncias globais, se houver
        imp = PACOTE.get("global_importances") or []
        if imp:
            df_imp = pd.DataFrame(imp)
            st.bar_chart(df_imp.set_index("variavel")["importancia_global"])

    st.divider()
    st.markdown(
        "**Campos esperados no CSV de lote:** `idade, ano_ingresso, anos_na_pm, fase_num, inde, iaa, ieg, ips, ida, ipv, ian, defasagem, risco_defasagem_atual, gap_iaa_ida, gap_ieg_ida, media_indicadores, indicadores_baixos_qtd, fase_grupo`  "
        "Você pode baixar o template com o botão abaixo e preencher a partir da exportação do PEDE."
    )
    tmpl = pd.DataFrame([{"idade": 12, "ano_ingresso": 2021, "anos_na_pm": 3, "fase_num": 2, "inde": 7.0, "iaa": 8.5, "ieg": 7.8, "ips": 6.5, "ida": 6.2, "ipv": 7.3, "ian": 5.0, "defasagem": -1, "risco_defasagem_atual": 1, "gap_iaa_ida": 2.3, "gap_ieg_ida": 1.6, "media_indicadores": 6.9, "indicadores_baixos_qtd": 1, "fase_grupo": "2"}])
    st.download_button("Baixar template CSV", tmpl.to_csv(index=False).encode("utf-8-sig"), file_name="template_lote_passos.csv", mime="text/csv")

with tab_lote:
    st.subheader("Predição em lote")
    st.markdown("Envie um CSV com as colunas do template (cabeçalho igual). O app devolve o mesmo CSV com `prob_risco` e `classificacao`.")
    arq = st.file_uploader("CSV", type=["csv"])
    if arq is not None:
        try:
            df = pd.read_csv(arq)
            cols_needed = PACOTE["feature_columns"]
            faltando = [c for c in cols_needed if c not in df.columns]
            if faltando:
                st.error(f"Colunas faltando: {faltando}")
            else:
                probs = PIPE.predict_proba(df[cols_needed])[:, 1]
                out = df.copy()
                out["prob_risco"] = probs
                out["classificacao"] = np.where(probs >= THR, "EM RISCO", "SEM RISCO")
                st.dataframe(out.head(20), use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Linhas", len(out))
                c2.metric("Em risco", int((out["prob_risco"] >= THR).sum()))
                c3.metric("Prevalência prevista", f"{(out['prob_risco'] >= THR).mean():.1%}")
                st.download_button("Baixar resultado", out.to_csv(index=False).encode("utf-8-sig"), file_name="predicoes_risco.csv", mime="text/csv")
                # histograma
                st.bar_chart(out["prob_risco"])
        except Exception as exc:
            st.error(f"Erro ao processar CSV: {exc}")

with tab_sobre:
    st.subheader("Sobre e deploy")
    st.markdown(
        """
**O que este app faz:** recebe indicadores do PEDE do ano *t* e estima a probabilidade de **defasagem (<0) no PEDE *t+1***.

**O que não faz:** não diagnostica, não substitui a escuta da equipe. Use como fila de triagem.

**Deploy no Streamlit Community Cloud**
1. Crie um repositório GitHub com `app.py`, `requirements.txt` e a pasta `outputs/modelo/modelo_risco_defasagem.joblib` (gerada pelo notebook 02).
2. Em https://share.streamlit.io → *New app* → aponte para `app.py`.
3. Python 3.12 (obrigatório: `numpy==2.5.2` exige Python ≥ 3.12). Não é necessário `secrets.toml`.

**Reproduzir localmente**
```bash
pip install -r requirements.txt
streamlit run app.py
```
"""
    )
    if METRICAS:
        st.json(METRICAS)
    st.caption("Passos Mágicos · PEDE · Datathon FIAP Postech — Fase 5")
