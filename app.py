import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import json

# --- データ読み込み関数 ---
# Streamlit Cloudではファイルパスが変わる可能性があるため、直接ファイルを読み込む関数を用意
@st.cache_data  # Cache the data loading
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            # Streamlit's uploaded_file is like a file-like object
            data = json.load(uploaded_file)
            df = pd.DataFrame(data)

            # データ前処理 (動画ID抽出と日時変換)
            df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')
            df['video_id'] = df['titleUrl'].str.extract(r'v=([^&]+)')
            df_processed = df.dropna(subset=['video_id']).copy()
            df_processed = df_processed[['time', 'video_id']]

            # 日次集計
            df_daily = df_processed.groupby([df_processed['time'].dt.date, 'video_id']).size().reset_index(name='daily_watch_count')
            df_daily['time'] = pd.to_datetime(df_daily['time'])  # Convert date back to datetime for sorting

            # 累積集計
            df_cumulative = df_daily.sort_values(by=['video_id', 'time']).copy()
            df_cumulative['cumulative_watch_count'] = df_cumulative.groupby('video_id')['daily_watch_count'].cumsum()

            # 全体の日次集計
            df_daily_total = df_processed['time'].dt.date.value_counts().sort_index().reset_index()
            df_daily_total.columns = ['date', 'total_watch_count']

            # 全体の月次集計
            df_monthly_total = df_processed['time'].dt.to_period('M').value_counts().sort_index().reset_index()
            df_monthly_total.columns = ['month', 'total_watch_count']
            df_monthly_total['month'] = df_monthly_total['month'].astype(str)

            # 動画情報辞書
            video_info_dict = {}
            for index, row in df.dropna(subset=['video_id']).iterrows():
                video_id = row['video_id']
                video_title = row['title'].replace(" を視聴しました", "").strip()
                thumbnail_url = f"http://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                video_info_dict[video_id] = {
                    'title': video_title,
                    'thumbnail_url': thumbnail_url
                }

            # 一番視聴回数が多かった日と月 (全体)
            most_watched_day = df_daily_total.loc[df_daily_total['total_watch_count'].idxmax()]
            most_watched_month = df_monthly_total.loc[df_monthly_total['total_watch_count'].idxmax()]

            return df, df_processed, df_daily, df_cumulative, df_daily_total, df_monthly_total, video_info_dict, most_watched_day, most_watched_month

        except Exception as e:
            st.error(f"データの読み込みまたは処理中にエラーが発生しました: {e}")
            return None, None, None, None, None, None, None, None, None
    else:
        return None, None, None, None, None, None, None, None, None


# --- Matplotlib 日本語フォント設定 ---
# Streamlit Cloud環境に依存するため、一般的なフォント設定を試みる
try:
    plt.rcParams['font.family'] = 'sans-serif'  # Fallback to a common font family
    plt.rcParams['axes.unicode_minus'] = False  # Avoid displaying minus sign as a box
except Exception as e:
    st.warning(f"Could not set font properties: {e}")


# --- Streamlit アプリケーション本体 ---

st.set_page_config(layout="wide")  # Use wide layout

st.title('📺 YouTube視聴履歴分析ダッシュボード')

st.sidebar.header('設定')

# Add file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("watch-history.jsonファイルをアップロードしてください", type=["json"])

# Load data using the uploaded file
df, df_processed, df_daily, df_cumulative, df_daily_total, df_monthly_total, video_info_dict, most_watched_day, most_watched_month = load_data(uploaded_file)

if df is not None:  # Proceed only if data loaded successfully

    # Get the list of unique video IDs from the processed DataFrame
    unique_video_ids = df_processed['video_id'].unique().tolist()

    # Add an option for "Overall Statistics"
    unique_video_ids.insert(0, '--- 全体統計を表示 ---')

    # Use a selectbox for the user to choose a video ID
    selected_video_id = st.sidebar.selectbox('表示したい動画IDを選択してください:', unique_video_ids)

    # Use columns for better layout in the main area
    col1, col2 = st.columns([2, 1])  # Example: Two columns, 2/3 and 1/3 width

    with col1:
        st.header('分析結果')

        # Display content based on the selection
        if selected_video_id == '--- 全体統計を表示 ---':
            st.subheader('📈 全体の視聴統計')

            st.markdown("---")  # Add a separator

            st.subheader('日ごとの総視聴回数')
            if df_daily_total is not None and not df_daily_total.empty:
                fig3, ax3 = plt.subplots(figsize=(14, 7))
                sns.lineplot(data=df_daily_total, x='date', y='total_watch_count', ax=ax3)
                ax3.set_title('日ごとの総視聴回数')
                ax3.set_xlabel('日付')
                ax3.set_ylabel('総視聴回数')
                plt.xticks(rotation=90)
                st.pyplot(fig3)
                plt.close(fig3)

                if most_watched_day is not None:
                    st.markdown(f"**💡 一番視聴回数が多かった日:** `{most_watched_day['date']}` ({most_watched_day['total_watch_count']} 回)")

            st.markdown("---")  # Add a separator

            st.subheader('月ごとの総視聴回数')
            if df_monthly_total is not None and not df_monthly_total.empty:
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                sns.barplot(data=df_monthly_total, x='month', y='total_watch_count', color='lightcoral', ax=ax4)
                ax4.set_title('月ごとの総視聴回数')
                ax4.set_xlabel('月')
                ax4.set_ylabel('総視聴回数')
                plt.xticks(rotation=45)
                st.pyplot(fig4)
                plt.close(fig4)

                if most_watched_month is not None:
                    st.markdown(f"**💡 一番視聴回数が多かった月:** `{most_watched_month['month']}` ({most_watched_month['total_watch_count']} 回)")

        else:
            # Display information for the specific video ID if selected
            video_id_input = selected_video_id
            st.subheader(f'🎥 動画ID: `{video_id_input}` の分析結果')

            # Get video title and thumbnail from the dictionary
            video_info = video_info_dict.get(video_id_input)

            if video_info:
                st.markdown(f"### {video_info['title']}", unsafe_allow_html=True)
                st.image(video_info['thumbnail_url'], caption=video_info['title'])

                # Filter data for the selected video ID
                df_filtered = df_cumulative[df_cumulative['video_id'] == video_id_input].copy()

                if not df_filtered.empty:
                    st.markdown("---")  # Add a separator

                    st.subheader('日次視聴回数')
                    df_filtered['date'] = df_filtered['time'].dt.date  # Convert time to date
                    fig1, ax1 = plt.subplots(figsize=(14, 7))
                    sns.barplot(data=df_filtered, x='date', y='daily_watch_count', color='skyblue', ax=ax1)
                    ax1.set_title(f'日次視聴回数: {video_info["title"]}')
                    ax1.set_xlabel('日付')
                    ax1.set_ylabel('日次視聴回数')
                    plt.xticks(rotation=90)
                    st.pyplot(fig1)
                    plt.close(fig1)

                    st.markdown("---")  # Add a separator

                    st.subheader('累積視聴回数')
                    fig2, ax2 = plt.subplots(figsize=(14, 7))
                    sns.lineplot(data=df_filtered, x='date', y='cumulative_watch_count', ax=ax2)
                    ax2.set_title(f'累積視聴回数: {video_info["title"]}')
                    ax2.set_xlabel('日付')
                    ax2.set_ylabel('累積視聴回数')
                    plt.xticks(rotation=90)
                    st.pyplot(fig2)
                    plt.close(fig2)

                    st.markdown("---")  # Add a separator

                    st.subheader('詳細データ')
                    st.dataframe(df_filtered[['date', 'daily_watch_count', 'cumulative_watch_count']])

                else:
                    st.warning(f"動画ID: {video_id_input} のデータは見つかりませんでした。")
            else:
                st.warning(f"動画ID: {video_id_input} の動画情報が見つかりませんでした。")

    with col2:
        st.header('補足情報')
        st.info("このダッシュボードはYouTubeの視聴履歴データに基づいています。投稿やショート動画は一部集計に含まれない場合があります。")

    # Optional: Add an expander for raw data preview
    with st.expander("元のデータ (プレビュー)"):
        st.dataframe(df.head())

else:
    st.info("サイドバーからwatch-history.jsonファイルをアップロードしてください。")