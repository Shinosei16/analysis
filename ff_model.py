import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

# ---------- ページ設定 ----------
st.set_page_config(
    page_title="Fama-French Factor Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Fama-French Factor Analysis")
st.caption("ファーマ・フレンチモデルで銘柄のリターンを分解します")

# ---------- サイドバー ----------
st.sidebar.header("設定")

ticker = st.sidebar.text_input("銘柄ティッカー（例：AAPL）", value="AAPL")
start_year = st.sidebar.slider("開始年", 2000, 2023, 2018)
end_year = st.sidebar.slider("終了年", 2001, 2024, 2023)
model = st.sidebar.selectbox("モデル", ["3ファクター", "4ファクター（モメンタム）"])

start = f"{start_year}-01-01"
end = f"{end_year}-12-31"

# ---------- データ取得関数 ----------
#データをキャッシュするため!st.cache_dataをつけている。データが取れない時exceptでメッセージを返す。
@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        px = yf.download(ticker, start=start, end=end, auto_adjust=True)["Close"]
        if px.empty:
            st.error(f"銘柄 {ticker} のデータが取得できませんでした。ティッカーを確認してください。")
            return None
        px_m = px.resample("ME").last()
        ret_m = px_m.pct_change().dropna()
        ret_m.index = ret_m.index.to_period("M")
        return ret_m
    except Exception as e:
        st.error(f"株価データの取得中にエラーが発生しました：{e}")
        return None

@st.cache_data
def get_ff_data(start_year, end_year):
    try:
        ff = web.DataReader("F-F_Research_Data_Factors", "famafrench", start=f"{start_year}-01-01", end=f"{end_year}-12-31")[0]
        ff = ff / 100
        return ff
    except Exception as e:
        st.error(f"FF3ファクターデータの取得中にエラーが発生しました：{e}")
        return None


@st.cache_data
def get_mom_data(start_year, end_year):
    try:
        mom = web.DataReader("F-F_Momentum_Factor", "famafrench", start=f"{start_year}-01-01", end=f"{end_year}-12-31")[0]
        mom = mom / 100
        return mom
    except Exception as e:
        st.error(f"モメンタムデータの取得中にエラーが発生しました：{e}")
        return None
    
# ---------- 分析関数 ----------
def run_regression(ret_m, ff, mom=None):
    if isinstance(ret_m, pd.DataFrame):
        ret_m = ret_m.iloc[:, 0]
    
    df = pd.merge(ret_m.rename("stock"), ff, left_index=True, right_index=True)
    
    # momがある場合は結合する
    if mom is not None:
        df = pd.merge(df, mom, left_index=True, right_index=True)
    
    df["excess_ret"] = df["stock"] - df["RF"]
    
    if mom is not None:
        # 4factors
        X = df[["Mkt-RF", "SMB", "HML", "Mom"]]
    else:
        # 3factors
        X = df[["Mkt-RF", "SMB", "HML"]]
    
    # 定数項を追加（αの推定に必要）
    X = sm.add_constant(X)
    y = df["excess_ret"]
    
    # OLS回帰
    result = sm.OLS(y, X).fit()
    
    return result, df

# ---------- メイン処理 ----------
if st.sidebar.button("分析する"):
    with st.spinner("データ取得中..."):
        ret_m = get_stock_data(ticker, start, end)
        ff = get_ff_data(start_year, end_year)
        
        if ret_m is None or ff is None:
            st.stop()
        
        if model == "4ファクター（モメンタム）":
            mom = get_mom_data(start_year, end_year)
            if mom is None:
                st.stop()
            result, df = run_regression(ret_m, ff, mom)
        else:
            result, df = run_regression(ret_m, ff)
    
    # ---------- モデルの説明 ----------
    with st.expander("モデルの説明"):
        st.markdown(r"""
        ### Fama-French 3ファクターモデル
        
        $$R_i - R_f = \alpha + \beta_1(R_m - R_f) + \beta_2 SMB + \beta_3 HML + \varepsilon$$
        
        | 記号 | 説明 |
        |---|---|
        | $R_i$ | 銘柄iの月次リターン |
        | $R_f$ | リスクフリーレート |
        | $\alpha$ | 切片（ファクターで説明できない超過リターン） |
        | $\beta_1$ | 市場ファクターの感応度 |
        | $\beta_2$ | 規模ファクターの感応度 |
        | $\beta_3$ | バリューファクターの感応度 |
        | $\varepsilon$ | 誤差項 |
        
        ### 各ファクターの意味
        
        **SMB (Small Minus Big)**  
        規模ファクター。小型株ポートフォリオと大型株ポートフォリオのリターンの差。$\beta_2 > 0$ なら小型株的な特性を持つ。
        
        **HML (High Minus Low)**  
        バリューファクター。割安株と割高株のリターンの差。$\beta_3 < 0$ なら成長株的な特性を持つ。
        
        ### 統計的検定
        
        各係数について以下の仮説検定を行う。
        
        $$H_0: \beta_i = 0 \quad H_1: \beta_i \neq 0$$
        
        $p < 0.05$ のとき帰無仮説を棄却し、そのファクターがリターンの説明に有意に寄与していると判断する。
        
        $\alpha$ については以下の検定を行う。
        
        $$H_0: \alpha = 0 \quad H_1: \alpha \neq 0$$
        
        $p < 0.05$ のとき、3つのファクターで説明できない統計的に有意な超過リターンが存在することを意味する。
        
        ### 決定係数 $R^2$
        
        モデルの当てはまりの良さを示す指標。値が高いほど3つのファクターで銘柄のリターンをよく説明できていることを示す。
        """)
        
    # ---------- 結果の表示 ----------
    st.subheader(f"{ticker} の分析結果（{start_year}〜{end_year}）")
    
    if model == "4ファクター（モメンタム）":
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("α（月次）", f"{result.params['const']:.4f}",
                    f"p値: {result.pvalues['const']:.3f}")
        col2.metric("β（市場）", f"{result.params['Mkt-RF']:.4f}",
                    f"p値: {result.pvalues['Mkt-RF']:.3f}")
        col3.metric("SMB", f"{result.params['SMB']:.4f}",
                    f"p値: {result.pvalues['SMB']:.3f}")
        col4.metric("HML", f"{result.params['HML']:.4f}",
                    f"p値: {result.pvalues['HML']:.3f}")
        col5.metric("Mom", f"{result.params['Mom']:.4f}",
                    f"p値: {result.pvalues['Mom']:.3f}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("α（月次）", f"{result.params['const']:.4f}",
                    f"p値: {result.pvalues['const']:.3f}")
        col2.metric("β（市場）", f"{result.params['Mkt-RF']:.4f}",
                    f"p値: {result.pvalues['Mkt-RF']:.3f}")
        col3.metric("SMB", f"{result.params['SMB']:.4f}",
                    f"p値: {result.pvalues['SMB']:.3f}")
        col4.metric("HML", f"{result.params['HML']:.4f}",
                    f"p値: {result.pvalues['HML']:.3f}")

    st.metric("R²", f"{result.rsquared:.4f}")
    
    # 詳細な回帰結果
    with st.expander("詳細な回帰結果を見る"):
        st.text(result.summary())
    
    # ---------- グラフ ----------
    st.subheader("実際の超過リターン vs モデルの予測値")

    fig, ax = plt.subplots(figsize=(7, 7))

    # 散布図
    ax.scatter(result.fittedvalues, df["excess_ret"], alpha=0.6, label="各月のデータ")

    # y=xの対角線
    min_val = min(result.fittedvalues.min(), df["excess_ret"].min())
    max_val = max(result.fittedvalues.max(), df["excess_ret"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x（完全予測）")

    ax.set_xlabel("モデルの予測値")
    ax.set_ylabel("実際の超過リターン")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)