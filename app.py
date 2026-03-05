# -*- coding: utf-8 -*-
# Streamlit YouTube視聴履歴ダッシュボード（単一ファイル・関数化版）
# 全体統計 / 動画別の可視化を video_id の有無で切替
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ---------------------------------------
# Matplotlib 日本語フォント設定（Cloud環境考慮）
# ---------------------------------------
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    st.warning(f"日本語フォント設定に失敗しました: {e}")
    plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------
# 共通ピボット＆描画ユーティリティ
# ---------------------------------------
def _weekday_pivot(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    曜日（行）× (年月, 日)（列）のヒートマップ用ピボットテーブルを作成。
    全体/動画別いずれにも使える共通化ユーティリティ。
    """
    tmp = df.copy()
    tmp['date'] = pd.to_datetime(tmp[date_col]).dt.date
    tmp['day'] = pd.to_datetime(tmp['date']).dt.day
    tmp['month_year'] = pd.to_datetime(tmp['date']).dt.to_period('M').astype(str)
    tmp['weekday_num'] = pd.to_datetime(tmp['date']).dt.dayofweek
    weekday_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    tmp['weekday'] = tmp['weekday_num'].map(weekday_map)
    pt = pd.pivot_table(
        tmp, values=value_col, index='weekday',
        columns=['month_year', 'day'], fill_value=0
    )
    pt = pt.reindex(list(weekday_map.values()))
    return pt

def render_heatmap(pivot_table: pd.DataFrame, title: str, cbar_label: str = '視聴回数', figsize=(20,8)):
    """ヒートマップ描画（共通）。"""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot_table, ax=ax, cmap='YlGnBu', linewidths=.5, cbar_kws={'label': cbar_label})
    ax.set_title(title)
    ax.set_xlabel('年月 - 日')
    ax.set_ylabel('曜日')
    plt.xticks(rotation=90)
    return fig

def render_line(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, figsize=(14,7)):
    """折れ線描画（共通）。"""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    sns.lineplot(data=df, x=x, y=y, ax=ax, color='red')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.xticks(rotation=90)
    return fig

def render_bar(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, figsize=(14,7), color='skyblue'):
    """棒グラフ描画（共通）。"""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    sns.barplot(data=df, x=x, y=y, color=color, ax=ax)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.xticks(rotation=90)
    for c in ax.containers:
        ax.bar_label(c)
    return fig

# ---------------------------------------
# データ読み込み・前処理・スコアボード
# ---------------------------------------
@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file):
    """
    JSONの視聴履歴を読み込み、JSTで日次/累積/全体集計を返す。
    戻り値:
      df, df_processed, df_daily, df_cumulative, df_daily_total, df_monthly_total,
      video_info_dict, most_watched_day, most_watched_month
    """
    if uploaded_file is None:
        return (None,)*9

    data = json.load(uploaded_file)
    df = pd.DataFrame(data)

    # UTCとしてパース → JSTへ変換（tz_convertはtz-aware必須）
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df['video_id'] = df['titleUrl'].str.extract(r'v=([^&]+)')
    df_processed = df.dropna(subset=['video_id']).copy()
    df_processed['time_jst'] = df_processed['time'].dt.tz_convert('Asia/Tokyo')

    # 日次（動画別）
    df_daily = (
        df_processed
        .groupby([df_processed['time_jst'].dt.date, 'video_id'])
        .size()
        .reset_index(name='daily_watch_count')
    )
    df_daily['time'] = pd.to_datetime(df_daily['time_jst'])

    # 累積（動画別）
    df_cumulative = df_daily.sort_values(['video_id', 'time']).copy()
    df_cumulative['cumulative_watch_count'] = (
        df_cumulative.groupby('video_id')['daily_watch_count'].cumsum()
    )

    # 全体：日次
    df_daily_total = (
        df_processed['time_jst'].dt.date.value_counts().sort_index().reset_index()
    )
    df_daily_total.columns = ['date', 'total_watch_count']
    df_daily_total['date'] = pd.to_datetime(df_daily_total['date'])

    # 全体：月次
    df_monthly_total = (
        df_processed['time_jst'].dt.to_period('M').value_counts().sort_index().reset_index()
    )
    df_monthly_total.columns = ['month', 'total_watch_count']
    df_monthly_total['month'] = df_monthly_total['month'].astype(str)

    # 動画タイトル/サムネ
    video_info_dict = {}
    for _, row in df_processed.iterrows():
        vid = row['video_id']
        title = row['title'].replace(" を視聴しました", "").strip()
        thumb_url = f"http://img.youtube.com/vi/{vid}/hqdefault.jpg"
        video_info_dict[vid] = {'title': title, 'thumbnail_url': thumb_url}

    # 最大日/最大月
    most_watched_day = None
    if not df_daily_total.empty:
        most_watched_day = df_daily_total.loc[df_daily_total['total_watch_count'].idxmax()]
    most_watched_month = None
    if not df_monthly_total.empty:
        most_watched_month = df_monthly_total.loc[df_monthly_total['total_watch_count'].idxmax()]

    return (df, df_processed, df_daily, df_cumulative,
            df_daily_total, df_monthly_total,
            video_info_dict, most_watched_day, most_watched_month)

@st.cache_data(show_spinner=False)
def build_scoreboard(df_cumulative: pd.DataFrame, video_info_dict: dict) -> pd.DataFrame:
    """
    最新時点の累積視聴回数でスコアボードを作成。
    """
    if df_cumulative is None or df_cumulative.empty:
        return pd.DataFrame(columns=['video_id','cumulative_watch_count','title','thumbnail_url'])

    latest_idx = df_cumulative.groupby('video_id')['time'].idxmax()
    df_latest = df_cumulative.loc[latest_idx, ['video_id', 'cumulative_watch_count']]

    df_info = pd.DataFrame.from_dict(video_info_dict, orient='index').reset_index()
    df_info.columns = ['video_id', 'title', 'thumbnail_url']

    out = pd.merge(df_latest, df_info, on='video_id', how='left')
    # sort in descending order of cumulative views and keep only the top 5 entries
    out = out.sort_values('cumulative_watch_count', ascending=False).reset_index(drop=True)
    return out.head(20)

# ---------------------------------------
# たった1つの公開API：全体 / 動画別の統一描画
# ---------------------------------------
def show_dashboard(
    df_daily_total: pd.DataFrame,        # 全体日次（date, total_watch_count）
    df_monthly_total: pd.DataFrame,      # 全体月次（month, total_watch_count）
    df_daily: pd.DataFrame,              # 動画別日次（time_jst, video_id, daily_watch_count）
    df_cumulative: pd.DataFrame,         # 動画別累積（time, video_id, daily_watch_count, cumulative_watch_count）
    video_info_dict: dict,               # {video_id: {title, thumbnail_url}}
    df_scoreboard: pd.DataFrame | None = None,
    most_day: pd.Series | None = None,
    most_month: pd.Series | None = None,
    video_id: str | None = None
):
    """
    video_id が None → 全体統計、指定あり → 動画別。
    中身は共通ユーティリティ関数のみで描画。
    """
    if video_id is None:
        # ---- 全体統計 ----
        st.subheader('📈 全体の視聴統計')

        if df_scoreboard is not None and not df_scoreboard.empty:
            # Display a scoreboard of the top videos by cumulative views (latest)
            st.subheader('Top Videos by Cumulative Views (Latest)')
            # Only show the video title and cumulative views in the table
            st.dataframe(
                df_scoreboard[['title', 'cumulative_watch_count']],
                use_container_width=True, hide_index=True
            )

        if df_daily_total is not None and not df_daily_total.empty:
            st.markdown("---")
            # Heatmap of daily total watch counts
            st.subheader('Daily Total Views Heatmap')
            pt = _weekday_pivot(
                df_daily_total.rename(columns={'date': 'date_for_pivot'}),
                date_col='date_for_pivot', value_col='total_watch_count'
            )
            st.pyplot(render_heatmap(pt, 'Daily Total Views Heatmap'))

            st.markdown("---")
            # Bar chart of daily total watch counts
            st.subheader('Daily Total Views')
            st.pyplot(render_bar(
                df_daily_total, x='date', y='total_watch_count',
                title='Daily Total Views', xlabel='Date', ylabel='Total Views'
            ))
            if most_day is not None and 'date' in most_day:
                st.markdown(f"**💡 Most watched day:** `{pd.to_datetime(most_day['date']).date()}` ({most_day['total_watch_count']} views)")

        if df_monthly_total is not None and not df_monthly_total.empty:
            st.markdown("---")
            # Bar chart of monthly total watch counts
            st.subheader('Monthly Total Views')
            st.pyplot(render_bar(
                df_monthly_total, x='month', y='total_watch_count',
                title='Monthly Total Views', xlabel='Month', ylabel='Total Views',
                figsize=(12,6), color='salmon'
            ))
            if most_month is not None and 'month' in most_month:
                st.markdown(f"**💡 Most watched month:** `{most_month['month']}` ({most_month['total_watch_count']} views)")

    else:
        # ---- 動画別 ----
        info = video_info_dict.get(video_id, {'title': f'動画ID: {video_id}', 'thumbnail_url': None})
        title = info['title']
        st.subheader(f'🎥 {title}')
        if info.get('thumbnail_url'):
            st.image(info['thumbnail_url'], caption=title)

        heat_src = df_daily[df_daily['video_id'] == video_id].copy()
        if not heat_src.empty:
            st.markdown("---")
            # Heatmap for daily watch counts of a single video
            st.subheader('Daily View Count Heatmap')
            pt = _weekday_pivot(
                heat_src.rename(columns={'time_jst':'date_for_pivot'}),
                date_col='date_for_pivot', value_col='daily_watch_count'
            )
            st.pyplot(render_heatmap(pt, f'Daily Views Heatmap'))
        else:
            st.info('No data available for heatmap.')

        df_v = df_cumulative[df_cumulative['video_id'] == video_id].copy()
        if df_v.empty:
            st.warning('対象動画の集計データが見つかりませんでした。')
            return
        df_v['date'] = df_v['time'].dt.date
        
        st.markdown("---")
        # Line chart for cumulative watch counts of a single video
        st.subheader('Cumulative Views')
        st.pyplot(render_line(
            df_v, x='date', y='cumulative_watch_count',
            title=f'Cumulative Views', xlabel='Date', ylabel='Cumulative Views'
        ))

        st.markdown("---")
        st.subheader('詳細データ')
        st.dataframe(
            df_v[['date', 'daily_watch_count', 'cumulative_watch_count']],
            use_container_width=True, hide_index=True
        )

# ---------------------------------------
# メインアプリ
# ---------------------------------------
def main():
    st.set_page_config(layout="wide")
    st.title('📺 YouTube視聴履歴分析ダッシュボード')
    st.sidebar.header('設定')

    uploaded_file = st.sidebar.file_uploader("watch-history.jsonファイルをアップロードしてください", type=["json"])

    (df, df_processed, df_daily, df_cumulative,
     df_daily_total, df_monthly_total,
     video_info_dict, most_day, most_month) = load_and_process_data(uploaded_file)

    if df is None:
        st.info("サイドバーからwatch-history.jsonファイルをアップロードしてください。")
        return

    # スコアボード
    df_scoreboard = build_scoreboard(df_cumulative, video_info_dict)

    # セレクタ：先頭は全体統計
    unique_ids = df_processed['video_id'].unique().tolist() if df_processed is not None else []
    unique_ids.insert(0, '--- 全体統計を表示 ---')
    selected = st.sidebar.selectbox('表示したい動画IDを選択してください:', unique_ids)

    if selected == '--- 全体統計を表示 ---':
        show_dashboard(
            df_daily_total=df_daily_total,
            df_monthly_total=df_monthly_total,
            df_daily=df_daily,
            df_cumulative=df_cumulative,
            video_info_dict=video_info_dict,
            df_scoreboard=df_scoreboard,
            most_day=most_day,
            most_month=most_month,
            video_id=None
        )
    else:
        show_dashboard(
            df_daily_total=df_daily_total,
            df_monthly_total=df_monthly_total,
            df_daily=df_daily,
            df_cumulative=df_cumulative,
            video_info_dict=video_info_dict,
            df_scoreboard=df_scoreboard,
            most_day=most_day,
            most_month=most_month,
            video_id=selected
        )

    with st.expander("元のデータ (プレビュー)"):
        st.dataframe(df.head(), use_container_width=True)

# エントリポイント
if __name__ == "__main__":
    main()
