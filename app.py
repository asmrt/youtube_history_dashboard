# -*- coding: utf-8 -*-
# Streamlit YouTube視聴履歴ダッシュボード（単一ファイル・関数化版）
# 全体統計 / 動画別の可視化を video_id の有無で切替
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import json

# ---------------------------------------
# カラーパレット定義（落ち着いた赤・灰色ベース）
# ---------------------------------------
COLOR_RED        = '#A63228'
COLOR_RED_LIGHT  = '#C9736B'
COLOR_GRAY       = '#6B6B6B'
COLOR_GRAY_LIGHT = '#B0B0B0'

HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'gray_red', ['#F5F5F5', '#B0B0B0', '#A63228']
)

# ---------------------------------------
# Matplotlib 日本語フォント設定
# ---------------------------------------
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------
# 共通ピボット＆描画ユーティリティ
# ---------------------------------------
def _weekday_pivot(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
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
    return pt.reindex(list(weekday_map.values()))

def render_heatmap(pivot_table, title, cbar_label='視聴回数', figsize=(20, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    sns.heatmap(pivot_table, ax=ax, cmap=HEATMAP_CMAP,
                linewidths=.5, linecolor='#E0E0E0', cbar_kws={'label': cbar_label})
    ax.set_title(title, color=COLOR_GRAY, fontsize=13)
    ax.set_xlabel('年月 - 日', color=COLOR_GRAY)
    ax.set_ylabel('曜日', color=COLOR_GRAY)
    ax.tick_params(colors=COLOR_GRAY)
    plt.xticks(rotation=90)
    return fig

def render_line(df, x, y, title, xlabel, ylabel, figsize=(14, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    sns.lineplot(data=df, x=x, y=y, ax=ax, color=COLOR_RED, linewidth=2)
    ax.set_title(title, color=COLOR_GRAY, fontsize=13)
    ax.set_xlabel(xlabel, color=COLOR_GRAY)
    ax.set_ylabel(ylabel, color=COLOR_GRAY)
    ax.tick_params(colors=COLOR_GRAY)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRAY_LIGHT)
    ax.spines['bottom'].set_color(COLOR_GRAY_LIGHT)
    plt.xticks(rotation=90)
    return fig

def render_bar(df, x, y, title, xlabel, ylabel, figsize=(14, 7), color=COLOR_RED):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    sns.barplot(data=df, x=x, y=y, color=color, ax=ax)
    ax.set_title(title, color=COLOR_GRAY, fontsize=13)
    ax.set_xlabel(xlabel, color=COLOR_GRAY)
    ax.set_ylabel(ylabel, color=COLOR_GRAY)
    ax.tick_params(colors=COLOR_GRAY)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRAY_LIGHT)
    ax.spines['bottom'].set_color(COLOR_GRAY_LIGHT)
    plt.xticks(rotation=90)
    for c in ax.containers:
        ax.bar_label(c, color=COLOR_GRAY, fontsize=8)
    return fig

# ---------------------------------------
# データ読み込み・前処理
# ---------------------------------------
@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file):
    if uploaded_file is None:
        return None, None

    data = json.load(uploaded_file)
    df = pd.DataFrame(data)

    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df['video_id'] = df['titleUrl'].str.extract(r'v=([^&]+)')

    # subtitles からチャンネル名を抽出（subtitles[0].name）
    def extract_channel(subtitles):
        if isinstance(subtitles, list) and len(subtitles) > 0:
            return subtitles[0].get('name', None)
        return None

    df['channel_name'] = df['subtitles'].apply(extract_channel)

    df_processed = df.dropna(subset=['video_id']).copy()
    df_processed['time_jst'] = df_processed['time'].dt.tz_convert('Asia/Tokyo')

    # video_info_dict：video_id → {title, thumbnail_url, channel_name}
    video_info_dict = {}
    for _, row in df_processed.iterrows():
        vid = row['video_id']
        title = row['title'].replace(" を視聴しました", "").strip()
        thumb_url = f"http://img.youtube.com/vi/{vid}/hqdefault.jpg"
        channel = row.get('channel_name', None)
        video_info_dict[vid] = {
            'title': title,
            'thumbnail_url': thumb_url,
            'channel_name': channel if pd.notna(channel) else '不明'
        }

    return df_processed, video_info_dict


def apply_filters(df_processed: pd.DataFrame, channel_filter: list, date_range: tuple) -> pd.DataFrame:
    """チャンネル絞り込み・期間絞り込みを適用して df_processed を返す。"""
    df = df_processed.copy()
    start_date, end_date = date_range
    df = df[
        (df['time_jst'].dt.date >= start_date) &
        (df['time_jst'].dt.date <= end_date)
    ]
    if channel_filter:
        df = df[df['channel_name'].isin(channel_filter)]
    return df


def build_aggregates(df_processed: pd.DataFrame):
    """フィルタ済み df_processed から各種集計を作成して返す。"""
    if df_processed is None or df_processed.empty:
        return None, None, None, None

    # 動画別日次
    df_daily = (
        df_processed
        .groupby([df_processed['time_jst'].dt.date, 'video_id'])
        .size()
        .reset_index(name='daily_watch_count')
    )
    df_daily.columns = ['time_jst', 'video_id', 'daily_watch_count']
    df_daily['time'] = pd.to_datetime(df_daily['time_jst'])

    # 動画別累積
    df_cumulative = df_daily.sort_values(['video_id', 'time']).copy()
    df_cumulative['cumulative_watch_count'] = (
        df_cumulative.groupby('video_id')['daily_watch_count'].cumsum()
    )

    # 全体日次
    df_daily_total = (
        df_processed['time_jst'].dt.date.value_counts().sort_index().reset_index()
    )
    df_daily_total.columns = ['date', 'total_watch_count']
    df_daily_total['date'] = pd.to_datetime(df_daily_total['date'])

    # 全体月次
    df_monthly_total = (
        df_processed['time_jst'].dt.to_period('M').value_counts().sort_index().reset_index()
    )
    df_monthly_total.columns = ['month', 'total_watch_count']
    df_monthly_total['month'] = df_monthly_total['month'].astype(str)

    return df_daily, df_cumulative, df_daily_total, df_monthly_total


def build_scoreboard(df_cumulative: pd.DataFrame, video_info_dict: dict) -> pd.DataFrame:
    if df_cumulative is None or df_cumulative.empty:
        return pd.DataFrame(columns=['video_id','cumulative_watch_count','title','thumbnail_url','channel_name'])

    latest_idx = df_cumulative.groupby('video_id')['time'].idxmax()
    df_latest = df_cumulative.loc[latest_idx, ['video_id', 'cumulative_watch_count']]

    df_info = pd.DataFrame.from_dict(video_info_dict, orient='index').reset_index()
    df_info.columns = ['video_id', 'title', 'thumbnail_url', 'channel_name']

    out = pd.merge(df_latest, df_info, on='video_id', how='left')
    out = out.sort_values('cumulative_watch_count', ascending=False).reset_index(drop=True)
    return out.head(50)

# ---------------------------------------
# ダッシュボード描画
# ---------------------------------------
def show_dashboard(
    df_daily_total, df_monthly_total, df_daily, df_cumulative,
    video_info_dict, df_scoreboard=None, video_id=None
):
    if video_id is None:
        # ---- 全体統計 ----
        st.subheader('📈 全体の視聴統計')

        if df_scoreboard is not None and not df_scoreboard.empty:
            st.subheader('Top Videos by Cumulative Views (Latest)')

            top3 = df_scoreboard.head(3)
            cols = st.columns(3)
            medals = ['🥇', '🥈', '🥉']
            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    st.markdown(f"**{medals[i]} {row['title']}**")
                    st.image(
                        row['thumbnail_url'],
                        caption=f"{row['cumulative_watch_count']} views  |  {row['video_id']}",
                        use_container_width=True
                    )

            st.markdown("")
            st.dataframe(
                df_scoreboard[['video_id', 'title', 'channel_name', 'cumulative_watch_count']],
                use_container_width=True, hide_index=True
            )

        if df_daily_total is not None and not df_daily_total.empty:
            st.markdown("---")
            st.subheader('Daily Total Views Heatmap')
            pt = _weekday_pivot(
                df_daily_total.rename(columns={'date': 'date_for_pivot'}),
                date_col='date_for_pivot', value_col='total_watch_count'
            )
            st.pyplot(render_heatmap(pt, 'Daily Total Views Heatmap'))

            st.markdown("---")
            st.subheader('Daily Total Views')
            most_day = df_daily_total.loc[df_daily_total['total_watch_count'].idxmax()]
            st.pyplot(render_bar(
                df_daily_total, x='date', y='total_watch_count',
                title='Daily Total Views', xlabel='Date', ylabel='Total Views'
            ))
            st.markdown(f"**💡 Most watched day:** `{pd.to_datetime(most_day['date']).date()}` ({most_day['total_watch_count']} views)")

        if df_monthly_total is not None and not df_monthly_total.empty:
            st.markdown("---")
            st.subheader('Monthly Total Views')
            most_month = df_monthly_total.loc[df_monthly_total['total_watch_count'].idxmax()]
            st.pyplot(render_bar(
                df_monthly_total, x='month', y='total_watch_count',
                title='Monthly Total Views', xlabel='Month', ylabel='Total Views',
                figsize=(12, 6), color=COLOR_GRAY
            ))
            st.markdown(f"**💡 Most watched month:** `{most_month['month']}` ({most_month['total_watch_count']} views)")

    else:
        # ---- 動画別 ----
        info = video_info_dict.get(video_id, {'title': f'動画ID: {video_id}', 'thumbnail_url': None, 'channel_name': '不明'})
        st.subheader(f'🎥 {info["title"]}')
        st.caption(f'📺 チャンネル: {info.get("channel_name", "不明")}　|　🆔 {video_id}')
        if info.get('thumbnail_url'):
            st.image(info['thumbnail_url'], caption=info['title'])

        heat_src = df_daily[df_daily['video_id'] == video_id].copy()
        if not heat_src.empty:
            st.markdown("---")
            st.subheader('Daily View Count Heatmap')
            pt = _weekday_pivot(
                heat_src.rename(columns={'time_jst': 'date_for_pivot'}),
                date_col='date_for_pivot', value_col='daily_watch_count'
            )
            st.pyplot(render_heatmap(pt, 'Daily Views Heatmap'))
        else:
            st.info('No data available for heatmap.')

        df_v = df_cumulative[df_cumulative['video_id'] == video_id].copy()
        if df_v.empty:
            st.warning('対象動画の集計データが見つかりませんでした。')
            return
        df_v['date'] = df_v['time'].dt.date

        st.markdown("---")
        st.subheader('Cumulative Views')
        st.pyplot(render_line(
            df_v, x='date', y='cumulative_watch_count',
            title='Cumulative Views', xlabel='Date', ylabel='Cumulative Views'
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

    uploaded_file = st.sidebar.file_uploader(
        "watch-history.jsonファイルをアップロードしてください", type=["json"]
    )

    df_processed, video_info_dict = load_and_process_data(uploaded_file)

    if df_processed is None:
        st.info("サイドバーからwatch-history.jsonファイルをアップロードしてください。")
        return

    # ---- サイドバー：期間フィルタ ----
    st.sidebar.markdown("---")
    st.sidebar.subheader('📅 期間絞り込み')
    min_date = df_processed['time_jst'].dt.date.min()
    max_date = df_processed['time_jst'].dt.date.max()
    date_start = st.sidebar.date_input('開始日', value=min_date, min_value=min_date, max_value=max_date)
    date_end   = st.sidebar.date_input('終了日', value=max_date, min_value=min_date, max_value=max_date)
    if date_start > date_end:
        st.sidebar.error('開始日は終了日より前に設定してください。')
        return

    # ---- サイドバー：チャンネルフィルタ ----
    st.sidebar.markdown("---")
    st.sidebar.subheader('📺 チャンネル絞り込み')
    all_channels = sorted(df_processed['channel_name'].dropna().unique().tolist())
    selected_channels = st.sidebar.multiselect(
        'チャンネルを選択（未選択=全件）', options=all_channels, default=[]
    )

    # ---- フィルタ適用 ----
    df_filtered = apply_filters(df_processed, selected_channels, (date_start, date_end))

    if df_filtered.empty:
        st.warning('選択された条件に一致するデータがありません。')
        return

    # ---- 集計 ----
    df_daily, df_cumulative, df_daily_total, df_monthly_total = build_aggregates(df_filtered)

    # フィルタ済みの video_info_dict（絞り込み後の動画のみ）
    filtered_ids = set(df_filtered['video_id'].unique())
    filtered_video_info = {k: v for k, v in video_info_dict.items() if k in filtered_ids}

    df_scoreboard = build_scoreboard(df_cumulative, filtered_video_info)

    # ---- サイドバー：動画セレクタ ----
    st.sidebar.markdown("---")
    st.sidebar.subheader('🎬 動画選択')
    OVERALL_LABEL = '--- 全体統計を表示 ---'
    title_to_id = {}
    for vid, info in filtered_video_info.items():
        label = info['title']
        if label in title_to_id:
            existing_vid = title_to_id[label]
            title_to_id[f"{label} [{existing_vid}]"] = existing_vid
            del title_to_id[label]
            title_to_id[f"{label} [{vid}]"] = vid
        else:
            title_to_id[label] = vid

    selector_options = [OVERALL_LABEL] + sorted(title_to_id.keys())
    selected_label = st.sidebar.selectbox('表示したい動画を選択してください:', selector_options)
    selected_video_id = None if selected_label == OVERALL_LABEL else title_to_id.get(selected_label)

    # ---- 絞り込み状況バッジ ----
    filter_notes = [f"期間: {date_start} 〜 {date_end}"]
    if selected_channels:
        filter_notes.append(f"チャンネル: {', '.join(selected_channels)}")
    st.caption('🔍 ' + '　|　'.join(filter_notes))

    # ---- メイン表示 ----
    show_dashboard(
        df_daily_total=df_daily_total,
        df_monthly_total=df_monthly_total,
        df_daily=df_daily,
        df_cumulative=df_cumulative,
        video_info_dict=filtered_video_info,
        df_scoreboard=df_scoreboard,
        video_id=selected_video_id
    )

    with st.expander("元のデータ (プレビュー)"):
        st.dataframe(df_filtered.head(), use_container_width=True)


if __name__ == "__main__":
    main()
