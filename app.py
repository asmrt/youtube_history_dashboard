import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import json

# --- データ読み込み関数 ---
# Streamlit Cloudではファイルパスが変わる可能性があるため、直接ファイルを読み込む関数を用意
@st.cache_data # Cache the data loading
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
            # Convert time to JST before daily aggregation
            df_processed['time_jst'] = df_processed['time'].dt.tz_convert('Asia/Tokyo')
            df_daily = df_processed.groupby([df_processed['time_jst'].dt.date, 'video_id']).size().reset_index(name='daily_watch_count')
            df_daily['time'] = pd.to_datetime(df_daily['time_jst']) # Convert date back to datetime for sorting

            # 累積集計
            df_cumulative = df_daily.sort_values(by=['video_id', 'time']).copy()
            df_cumulative['cumulative_watch_count'] = df_cumulative.groupby('video_id')['daily_watch_count'].cumsum()

            # 全体の日次集計
            df_daily_total = df_processed['time_jst'].dt.date.value_counts().sort_index().reset_index()
            df_daily_total.columns = ['date', 'total_watch_count']
            # Ensure 'date' column is datetime objects
            df_daily_total['date'] = pd.to_datetime(df_daily_total['date'])


            # 全体の月次集計
            df_monthly_total = df_processed['time_jst'].dt.to_period('M').value_counts().sort_index().reset_index()
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
            most_watched_day = None
            if not df_daily_total.empty:
                 most_watched_day = df_daily_total.loc[df_daily_total['total_watch_count'].idxmax()]

            most_watched_month = None
            if not df_monthly_total.empty:
                 most_watched_month = df_monthly_total.loc[df_monthly_total['total_watch_count'].idxmax()]

            # スコアボードデータの準備 (最新の累積視聴回数)
            if df_cumulative is not None and not df_cumulative.empty:
                latest_indices = df_cumulative.groupby('video_id')['time'].idxmax()
                df_latest_cumulative = df_cumulative.loc[latest_indices].copy()
                df_latest_cumulative = df_latest_cumulative[['video_id', 'cumulative_watch_count']]

                if video_info_dict:
                    df_video_info = pd.DataFrame.from_dict(video_info_dict, orient='index').reset_index()
                    df_video_info.columns = ['video_id', 'title', 'thumbnail_url']
                    df_scoreboard = pd.merge(df_latest_cumulative, df_video_info, on='video_id', how='left')
                else:
                    df_scoreboard = df_latest_cumulative.copy()
                    df_scoreboard['title'] = 'タイトル情報なし'
                    df_scoreboard['thumbnail_url'] = 'サムネイル情報なし'

                df_scoreboard = df_scoreboard.sort_values(by='cumulative_watch_count', ascending=False).reset_index(drop=True)
            else:
                 df_scoreboard = pd.DataFrame(columns=['video_id', 'cumulative_watch_count', 'title', 'thumbnail_url'])


            return df, df_processed, df_daily, df_cumulative, df_daily_total, df_monthly_total, video_info_dict, most_watched_day, most_watched_month, df_scoreboard

        except Exception as e:
            st.error(f"データの読み込みまたは処理中にエラーが発生しました: {e}")
            return None, None, None, None, None, None, None, None, None, None
    else:
        return None, None, None, None, None, None, None, None, None, None


# --- Matplotlib 日本語フォント設定 ---
# Streamlit Cloud環境で日本語フォントを適切に表示するための設定
# Streamlit Cloudではfonts-ipaexfont-gothicが利用可能
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
    plt.rcParams['axes.unicode_minus'] = False # マイナス記号を正しく表示
    print("Using IPAexGothic font for matplotlib.")
except Exception as e:
    st.warning(f"日本語フォントの設定に失敗しました。文字化けする可能性があります: {e}")
    plt.rcParams['font.family'] = 'sans-serif' # Fallback to a common font family


# --- Streamlit アプリケーション本体 ---

st.set_page_config(layout="wide") # Use wide layout

st.title('📺 YouTube視聴履歴分析ダッシュボード')

st.sidebar.header('設定')

# Add file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("watch-history.jsonファイルをアップロードしてください", type=["json"])

# Load data using the uploaded file
df, df_processed, df_daily, df_cumulative, df_daily_total, df_monthly_total, video_info_dict, most_watched_day, most_watched_month, df_scoreboard = load_data(uploaded_file)

if df is not None: # Proceed only if data loaded successfully

    # Get the list of unique video IDs from the processed DataFrame
    if df_processed is not None:
        unique_video_ids = df_processed['video_id'].unique().tolist()
    else:
        unique_video_ids = []


    # Add an option for "Overall Statistics"
    unique_video_ids.insert(0, '--- 全体統計を表示 ---')

    # Use a selectbox for the user to choose a video ID
    selected_video_id = st.sidebar.selectbox('表示したい動画IDを選択してください:', unique_video_ids)

    # Use columns for better layout in the main area
    col1, col2 = st.columns([2, 1]) # Example: Two columns, 2/3 and 1/3 width

    with col1:
        st.header('分析結果')

        # Display content based on the selection
        if selected_video_id == '--- 全体統計を表示 ---':
            st.subheader('📈 全体の視聴統計')

            st.markdown("---") # Add a separator

            st.subheader('動画別 累積視聴回数 スコアボード (最新)')
            if df_scoreboard is not None and not df_scoreboard.empty:
                 # Display scoreboard, including title and cumulative count
                 st.dataframe(df_scoreboard[['title', 'cumulative_watch_count']])
            else:
                 st.info("スコアボードデータが見つかりませんでした。")


            st.markdown("---") # Add a separator

            st.subheader('日ごとの視聴回数ヒートマップ')
            if df_daily_total is not None and not df_daily_total.empty:
                # Prepare data for heatmap
                heatmap_data = df_daily_total.copy()
                # Use day of the month and month-year for columns
                heatmap_data['day'] = heatmap_data['date'].dt.day
                heatmap_data['month_year'] = heatmap_data['date'].dt.to_period('M').astype(str)


                # Use dayofweek to get a numerical representation (Mon=0, Sun=6)
                heatmap_data['weekday_num'] = heatmap_data['date'].dt.dayofweek
                weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
                heatmap_data['weekday'] = heatmap_data['weekday_num'].map(weekday_map)

                # Create pivot table
                pivot_table = pd.pivot_table(
                    heatmap_data,
                    values='total_watch_count',
                    index='weekday',
                    columns=['month_year', 'day'], # Use month-year and day for columns
                    fill_value=0
                )

                # Define weekday order for correct display
                weekday_order = list(weekday_map.values())

                # Reindex to ensure correct weekday order
                pivot_table = pivot_table.reindex(weekday_order)

                fig_heatmap, ax_heatmap = plt.subplots(figsize=(20, 8)) # Adjust figure size for daily view
                sns.heatmap(
                    pivot_table,
                    ax=ax_heatmap,
                    cmap='YlGnBu',
                    linewidths=.5,
                    annot=False,
                    fmt='d',
                    cbar_kws={'label': '視聴回数'} # colorbarのラベルを追加
                )
                ax_heatmap.set_title('日ごとの総視聴回数ヒートマップ') # Updated title
                ax_heatmap.set_xlabel('年月 - 日') # Update xlabel
                ax_heatmap.set_ylabel('曜日')
                plt.xticks(rotation=90) # Rotate x-axis labels for readability
                st.pyplot(fig_heatmap)
                plt.close(fig_heatmap)

            st.markdown("---") # Add a separator

            st.subheader('日ごとの総視聴回数')
            if df_daily_total is not None and not df_daily_total.empty:
                fig3, ax3 = plt.subplots(figsize=(14, 7))
                # Change background color to white
                fig3.patch.set_facecolor('white')
                ax3.set_facecolor('white')
                # Change line color to red
                sns.lineplot(data=df_daily_total, x='date', y='total_watch_count', ax=ax3, color='red')
                ax3.set_title('日ごとの総視聴回数')
                ax3.set_xlabel('日付')
                ax3.set_ylabel('総視聴回数')
                plt.xticks(rotation=90)

                # Add data labels (optional, can be crowded)
                # for i, row in df_daily_total.iterrows():
                #     ax3.text(row['date'], row['total_watch_count'], round(row['total_watch_count'], 1), color='black', ha="center")

                st.pyplot(fig3)
                plt.close(fig3)

                # Safely access and format date for most_watched_day
                if most_watched_day is not None and isinstance(most_watched_day, pd.Series) and 'date' in most_watched_day:
                    date_value = most_watched_day['date']
                    most_watched_day_str = str(date_value) # Default to string representation
                    if isinstance(date_value, pd.Timestamp):
                        most_watched_day_str = date_value.strftime('%Y-%m-%d')
                    elif isinstance(date_value, pd.Period): # Handle Period objects if necessary
                         most_watched_day_str = str(date_value)
                    st.markdown(f"**💡 一番視聴回数が多かった日:** `{most_watched_day_str}` ({most_watched_day['total_watch_count']} 回)")
                else:
                     st.info("一番視聴回数が多かった日のデータが見つかりませんでした。")


            st.markdown("---") # Add a separator

            st.subheader('月ごとの総視聴回数')
            if df_monthly_total is not None and not df_monthly_total.empty:
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                # Change background color to white
                fig4.patch.set_facecolor('white')
                ax4.set_facecolor('white')
                # Change bar color to a shade of red
                sns.barplot(data=df_monthly_total, x='month', y='total_watch_count', color='salmon', ax=ax4)
                ax4.set_title('月ごとの総視聴回数')
                ax4.set_xlabel('月')
                ax4.set_ylabel('総視聴回数')
                plt.xticks(rotation=45)

                # Add data labels
                for container in ax4.containers:
                    ax4.bar_label(container)

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
                    st.markdown("---") # Add a separator

                    st.subheader('日次視聴回数ヒートマップ')
                    # Prepare data for heatmap for the specific video
                    heatmap_data_video = df_daily[df_daily['video_id'] == video_id_input].copy()
                    if not heatmap_data_video.empty:
                        heatmap_data_video['date'] = pd.to_datetime(heatmap_data_video['time_jst']).dt.date
                        # Use day of the month and month-year for columns
                        heatmap_data_video['day'] = pd.to_datetime(heatmap_data_video['date']).dt.day
                        heatmap_data_video['month_year'] = pd.to_datetime(heatmap_data_video['date']).dt.to_period('M').astype(str)

                        heatmap_data_video['weekday_num'] = pd.to_datetime(heatmap_data_video['date']).dt.dayofweek
                        weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
                        heatmap_data_video['weekday'] = heatmap_data_video['weekday_num'].map(weekday_map)

                        # Create pivot table for video heatmap
                        pivot_table_video = pd.pivot_table(
                            heatmap_data_video,
                            values='daily_watch_count',
                            index='weekday',
                            columns=['month_year', 'day'], # Use month-year and day for columns
                            fill_value=0
                        )

                         # Define weekday order for correct display
                        weekday_order = list(weekday_map.values())

                        # Reindex to ensure correct weekday order
                        pivot_table_video = pivot_table_video.reindex(weekday_order)

                        fig_heatmap_video, ax_heatmap_video = plt.subplots(figsize=(20, 8)) # Adjust figure size for daily view
                        sns.heatmap(
                            pivot_table_video,
                            ax=ax_heatmap_video,
                            cmap='YlGnBu',
                            linewidths=.5,
                            annot=False,
                            fmt='d',
                             cbar_kws={'label': '視聴回数'} # colorbarのラベルを追加
                        )
                        ax_heatmap_video.set_title(f'日ごとの視聴回数ヒートマップ: {video_info["title"]}')
                        ax_heatmap_video.set_xlabel('年月 - 日') # Update xlabel
                        ax_heatmap_video.set_ylabel('曜日')
                        plt.xticks(rotation=90) # Rotate x-axis labels for readability
                        st.pyplot(fig_heatmap_video)
                        plt.close(fig_heatmap_video)
                    else:
                        st.info(f"動画ID: {video_id_input} のヒートマップデータは見つかりませんでした。")


                    st.markdown("---") # Add a separator

                    st.subheader('日次視聴回数')
                    df_filtered['date'] = df_filtered['time'].dt.date # Convert time to date
                    fig1, ax1 = plt.subplots(figsize=(14, 7))
                    # Change background color to white
                    fig1.patch.set_facecolor('white')
                    ax1.set_facecolor('white')
                     # Change bar color to a shade of blue
                    sns.barplot(data=df_filtered, x='date', y='daily_watch_count', color='skyblue', ax=ax1)
                    ax1.set_title(f'日次視聴回数: {video_info["title"]}')
                    ax1.set_xlabel('日付')
                    ax1.set_ylabel('日次視聴回数')
                    plt.xticks(rotation=90)

                    # Add data labels
                    for container in ax1.containers:
                        ax1.bar_label(container)

                    st.pyplot(fig1)
                    plt.close(fig1)

                    st.markdown("---") # Add a separator

                    st.subheader('累積視聴回数')
                    fig2, ax2 = plt.subplots(figsize=(14, 7))
                    # Change background color to white
                    fig2.patch.set_facecolor('white')
                    ax2.set_facecolor('white')
                    # Change line color to red
                    sns.lineplot(data=df_filtered, x='date', y='cumulative_watch_count', ax=ax2, color='red')
                    ax2.set_title(f'累積視聴回数: {video_info["title"]}')
                    ax2.set_xlabel('日付')
                    ax2.set_ylabel('累積視聴回数')
                    plt.xticks(rotation=90)

                    # Add data labels (can be crowded for many data points)
                    # for i, row in df_filtered.iterrows():
                    #     ax2.text(row['date'], row['cumulative_watch_count'], round(row['cumulative_watch_count'], 1), color='black', ha="center")

                    st.pyplot(fig2)
                    plt.close(fig2)

                    st.markdown("---") # Add a separator

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
