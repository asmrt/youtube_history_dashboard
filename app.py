import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import calplot
from matplotlib.colors import LinearSegmentedColormap

# --- 日本語フォント設定（Streamlit Cloud対応） ---
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    st.warning(f"日本語フォント設定に失敗: {e}")
    plt.rcParams['font.family'] = 'sans-serif'

# --- データ読み込み関数 ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return [None] * 9

    try:
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)

        # video_id抽出
        df['video_id'] = df['titleUrl'].str.extract(r'v=([^&]+)')
        df = df.dropna(subset=['video_id'])

        # 日時変換（tz-aware 対応）
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        df['time'] = df['time'].dt.tz_convert('Asia/Tokyo')

        # 日次集計
        df_daily = (
            df.groupby([df['time'].dt.date, 'video_id'])
              .size()
              .reset_index(name='daily_watch_count')
              .assign(time=lambda x: pd.to_datetime(x['time']))
        )

        # 累積
        df_cumulative = df_daily.sort_values(['video_id', 'time'])
        df_cumulative['cumulative_watch_count'] = df_cumulative.groupby('video_id')['daily_watch_count'].cumsum()

        # 全体日次 & 月次
        df_daily_total = (
            df['time'].dt.date.value_counts()
              .sort_index()
              .reset_index()
              .rename(columns={'index': 'date', 'time': 'total_watch_count'})
        )
        df_monthly_total = (
            df['time'].dt.to_period('M').value_counts()
              .sort_index()
              .reset_index()
              .rename(columns={'index': 'month', 'time': 'total_watch_count'})
        )
        df_monthly_total['month'] = df_monthly_total['month'].astype(str)

        # 動画情報
        video_info_dict = {
            vid: {
                'title': str(title).replace(" を視聴しました", "").strip(),
                'thumbnail_url': f"http://img.youtube.com/vi/{vid}/hqdefault.jpg"
            }
            for vid, title in zip(df['video_id'], df['title'])
        }

        most_watched_day = df_daily_total.loc[df_daily_total['total_watch_count'].idxmax()] if not df_daily_total.empty else None
        most_watched_month = df_monthly_total.loc[df_monthly_total['total_watch_count'].idxmax()] if not df_monthly_total.empty else None

        return df, df, df_daily, df_cumulative, df_daily_total, df_monthly_total, video_info_dict, most_watched_day, most_watched_month

    except Exception as e:
        st.error(f"データ処理中にエラー: {e}")
        return [None] * 9

# --- ヒートマップ表示 ---
def plot_calendar_heatmap(df_daily_total):
    if df_daily_total.empty:
        st.warning("日次データがありません")
        return
    cmap = LinearSegmentedColormap.from_list("black_grey_red", ["black", "grey", "red"])
    s = pd.Series(df_daily_total['total_watch_count'].values, index=pd.to_datetime(df_daily_total['date']))
    fig, ax = calplot.calplot(
        s,
        cmap=cmap,
        edgecolor='white',
        linewidth=0.5,
        colorbar=False,
        figsize=(12, 3)
    )
    st.pyplot(fig)

# --- 累積スコアボード表示 ---
def display_cumulative_score(df_cumulative):
    if df_cumulative.empty:
        st.warning("累積データがありません")
        return
    total_views = int(df_cumulative['cumulative_watch_count'].max())
    st.markdown(
        f"<h1 style='text-align: center; color: red; font-size: 80px;'>{total_views:,}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center;'>累積視聴回数</h3>",
        unsafe_allow_html=True
    )

# --- Streamlit アプリ ---
st.set_page_config(layout="wide")
st.title('📺 YouTube視聴履歴分析ダッシュボード')
st.sidebar.header('設定')
uploaded_file = st.sidebar.file_uploader("watch-history.jsonファイルをアップロードしてください", type=["json"])

df, df_processed, df_daily, df_cumulative, df_daily_total, df_monthly_total, video_info_dict, most_watched_day, most_watched_month = load_data(uploaded_file)

if df is not None:
    unique_video_ids = df_processed['video_id'].unique().tolist() if df_processed is not None else []
    unique_video_ids.insert(0, '--- 全体統計を表示 ---')
    selected_video_id = st.sidebar.selectbox('表示したい動画IDを選択してください:', unique_video_ids)

    if selected_video_id == '--- 全体統計を表示 ---':
        st.subheader("📅 日次集計")
        plot_calendar_heatmap(df_daily_total)

        st.subheader("🏆 累積スコアボード（全動画合計）")
        total_cumulative = df_cumulative.groupby('time')['daily_watch_count'].sum().cumsum().reset_index()
        total_cumulative['cumulative_watch_count'] = total_cumulative['daily_watch_count']
        display_cumulative_score(total_cumulative)

    else:
        video_info = video_info_dict.get(selected_video_id)
        if video_info:
            st.markdown(f"### {video_info['title']}")
            st.image(video_info['thumbnail_url'], caption=video_info['title'])

        st.subheader("📅 日次集計（対象動画）")
        df_video_daily = df_cumulative[df_cumulative['video_id'] == selected_video_id][['time', 'daily_watch_count']]
        if not df_video_daily.empty:
            df_video_daily_total = df_video_daily.groupby(df_video_daily['time'].dt.date)['daily_watch_count'].sum().reset_index()
            df_video_daily_total.columns = ['date', 'total_watch_count']
            plot_calendar_heatmap(df_video_daily_total)

        st.subheader("🏆 累積スコアボード（対象動画）")
        df_filtered = df_cumulative[df_cumulative['video_id'] == selected_video_id]
        display_cumulative_score(df_filtered)

    with st.expander("元のデータ (プレビュー)"):
        st.dataframe(df.head())

else:
    st.info("サイドバーからwatch-history.jsonファイルをアップロードしてください。")
