# cell_tracking_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import uuid
import tempfile
import os
import torch        
import zipfile, tempfile
from pathlib import Path
from model import *               
from cells_visualization import *
from cell_track import *



dashboard_results_path = "./tmp/OutputData/"

NUMERIC_COLS = [
    'lifetime', 'mean_size', 'mean_speed', 'mean_acceleration',
    'mean_angle', 'total_distance'
]

# Title

st.set_page_config(page_title='Cell‑Tracking Dashboard', layout='wide')
st.title('🧬 Cell‑Tracking Dashboard')



# try:
#     device                               
# except NameError:              
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')

@st.cache_data(show_spinner="🔬 Running full pipeline …", ttl=0)
def process_video(zip_file, _model, thr=0.35, fps_out=5, database=dashboard_results_path):
    """Segment, track & summarise a microscopy image sequence packed as a .zip - returns (df, video_bytes)."""

    #unzip images into a temporary directory
    extract_dir = tempfile.TemporaryDirectory()
    image_folder = Path(extract_dir.name)           

    #save the uploaded zip to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        tmp_zip.write(zip_file.read())
        zip_path = tmp_zip.name

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(image_folder)

        print(f"Contents of extracted folder {image_folder}:")

    # video‑specific folder
    video_id = str(uuid.uuid4())

    # folders
    for folder in ["frames", "predicted_masks", "labeled_masks","raw_images"]:
        os.makedirs(os.path.join(database, folder, video_id), exist_ok=True)

    #extract frames
    # frame_folder = os.path.join(database, "frames", video_id)

    # predict masks
    predicted_masks = os.path.join(database, "predicted_masks", video_id)
    predict_masks(_model, predicted_masks, image_folder)

    # load masks
    mask_files = sorted([f for f in os.listdir(predicted_masks) if f.endswith('.npy')])
    masks = [np.load(os.path.join(predicted_masks, f)) for f in mask_files]

    # track cells
    tracks = track_cells(masks)
    print(f"Tracked {len(tracks)} unique cells.")

    # detect mitosis
    mitosis = detect_mitosis(tracks)
    print(f"Detected {len(mitosis)} mitosis events.")

    # overlay + save masks
    labeled_masks = os.path.join(database, "labeled_masks", video_id)
    save_labeled_masks(tracks, original_masks=masks, save_dir=labeled_masks)

    # video from labeled masks
    os.makedirs(os.path.join(database, "segmentated_videos"), exist_ok=True)      
    video_path = os.path.join(database, "segmentated_videos", f"{video_id}.mp4")

    # video_bytes = create_video_from_images(labeled_masks,
    #                                    output_file=video_path,
    #                                    return_bytes=True)
    video_bytes = create_video_from_images(labeled_masks,
                                       output_file=video_path,
                                       return_bytes=False)

    segmented_fixed = video_bytes.replace(".mp4", "_fixed.mp4")
    fix_mp4_for_browser(video_bytes, segmented_fixed)

    #save raw images
    raw_images_path = os.path.join(database,"raw_images",video_id)
    stable_image_dir = save_extracted_images(image_folder,raw_images_path)
    
    
    # cerate and save raw video
    os.makedirs(os.path.join(database, "raw_videos"), exist_ok=True)      
    raw_video_path = os.path.join(database,"raw_videos",f"{video_id}.mp4")
    raw_video_bytes = create_video_from_raw_images( stable_image_dir, output_file=raw_video_path, return_bytes=False)

    raw_fixed = raw_video_bytes.replace(".mp4", "_fixed.mp4")
    fix_mp4_for_browser(raw_video_bytes, raw_fixed)


    #dataframe
    os.makedirs(os.path.join(database, "datasets"), exist_ok=True)    
    dataframe_path = os.path.join(database, "datasets", f"{video_id}.csv")
    advanced_features = extract_advanced_features(tracks, mitosis)
    df = save_features_to_csv(advanced_features, dataframe_path)

    # return df, video_bytes, raw_video_bytes
    return df, segmented_fixed, raw_fixed



# Sidebar
with st.sidebar:
    st.header('🎥 Video')
    # video_file = st.file_uploader('Microscopy video', type=['mp4', 'avi', 'mov', 'mkv'])
    video_file = st.file_uploader("Upload a folder of microscopy images (.zip)", type=["zip"])

    st.header('🧩 Checkpoint')
    ckpt_path = st.text_input('Segmentation Model','./models/maskrcnn_hela_best_v1.pth')

    if st.button('Load model', disabled=(video_file is None)):
        with st.spinner('Loading model …'):
            mdl = get_instance_seg_model(num_classes=2).to(device)
            mdl.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
            mdl.eval()
            st.session_state['model'] = mdl
        st.success('Model ready!')

    contamination = st.slider('Anomaly contamination', 0.01, 0.20, 0.05, 0.01)
    run_click = st.button('🚀 Run pipeline', disabled=('model' not in st.session_state or video_file is None))
    if run_click:
        # os.makedirs(f'{dashboard_results_path}/videos', exist_ok=True)
        # os.makedirs(f'{dashboard_results_path}/datasets', exist_ok=True)
        st.session_state['run_pipeline'] = True

# Run full pipeline
if st.session_state.get('run_pipeline') and 'processed' not in st.session_state:
    df_raw, vid_bytes, raw_video_bytes = process_video(video_file, st.session_state['model'])
    st.session_state['df_raw'] = df_raw
    st.session_state['vid'] = vid_bytes
    st.session_state['raw_video_bytes'] = raw_video_bytes
    st.session_state['processed'] = True
    st.rerun()


if 'processed' not in st.session_state:
    st.info('Upload a video, load the model, then click “Run pipeline”.')
    st.stop()

#Generated Data
df_raw = st.session_state['df_raw']

# anomaly detection
df = IsolationForest(
    n_estimators=300,
    contamination=contamination,
    random_state=42
).fit_predict(StandardScaler().fit_transform(df_raw[NUMERIC_COLS]))
df_raw = df_raw.copy()
df_raw['anomaly'] = df
df_raw['anomaly_flag'] = df_raw['anomaly'].map({1: '✅ Normal', -1: '❗ Anomaly'})


with st.sidebar:
    st.markdown('---')
    st.header('🔎 Filters')

    min_life, max_life = int(df_raw['lifetime'].min()), int(df_raw['lifetime'].max())
    life_range = st.slider('Lifetime (frames)', min_life, max_life, (min_life, max_life))

    role_options = df_raw['role'].unique().tolist() if 'role' in df_raw.columns else []
    role_choice = st.selectbox('Role', ['All'] + sorted(role_options))

# apply filters
mask = df_raw['lifetime'].between(*life_range)
if role_choice != 'All':
    mask &= df_raw['role'] == role_choice
df_view = df_raw[mask]


st.subheader('📺 Tracked Video Comparison')

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🟤 Raw Video**")
    st.video(st.session_state['raw_video_bytes'], format="video/mp4")

    # raw_path = os.path.abspath(st.session_state['raw_video_bytes'])
    # with open(raw_path, "rb") as f:
    #     st.video(f.read(), format="video/mp4")



with col2:
    st.markdown("**🟢 Labeled Video**")
    st.video(st.session_state['vid'], format="video/mp4")

# KPI block--
def kpi_block(df):
    total = len(df)
    avg_life = df['lifetime'].mean()
    mitosis_events = df.get('mitosis_status', pd.Series(dtype='object')).astype(str).eq('True').sum()
    c1, c2, c3 = st.columns(3)
    c1.metric('Total Cells', f'{total}')
    c2.metric('Avg Lifetime', f'{avg_life:.1f} frames')
    c3.metric('Mitosis Events', f'{mitosis_events}')

kpi_block(df_view)

# Tabs
(overview_tab, dist_tab, move_tab, corr_tab, angle_tab,
 mitosis_tab, roles_tab, raw_tab) = st.tabs([
    'Overview', 'Distributions', 'Movement',
    'Correlation', 'Angle', 'Mitosis', 'Roles', 'Raw Data'
])

# Overview
with overview_tab:
    st.plotly_chart(px.scatter(
        df_view, x='birth_frame', y='death_frame',
        color='anomaly_flag',
        hover_data=['id', 'lifetime', 'total_distance'],
        labels={'birth_frame': 'Birth', 'death_frame': 'Death'},
        title='Cell Life‑Cycle Timeline'
    ), use_container_width=True)

# Distributions
with dist_tab:
    feat_x = st.selectbox('X‑axis', NUMERIC_COLS, key='dist_x')
    feat_y = st.selectbox('Y‑axis', NUMERIC_COLS, index=1, key='dist_y')
    st.plotly_chart(px.histogram(
        df_view, x=feat_x, nbins=40, opacity=0.75,
        color='anomaly_flag', barmode='overlay',
        title=f'Distribution of {feat_x}'
    ), use_container_width=True)
    st.plotly_chart(px.scatter(
        df_view, x=feat_x, y=feat_y,
        color='anomaly_flag', hover_data=['id'],
        title=f'{feat_x} vs {feat_y}'
    ), use_container_width=True)

# Movement
with move_tab:
    st.plotly_chart(px.scatter(
        df_view, x='total_distance', y='mean_speed',
        color='anomaly_flag',
        hover_data=['id', 'lifetime'],
        title='Mean Speed vs Total Distance'
    ), use_container_width=True)

# Correlation
with corr_tab:
    st.plotly_chart(px.imshow(
        df_view[NUMERIC_COLS].corr(),
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Matrix'
    ), use_container_width=True)

# Angle
with angle_tab:
    angles = (df_view['mean_angle'] + 360) % 360
    counts, bins = np.histogram(angles, bins=np.linspace(0, 360, 37))
    centers = (bins[:-1] + bins[1:]) / 2
    fig_angle = go.Figure(go.Barpolar(r=counts, theta=centers, width=10))
    fig_angle.update_layout(title='Mean Direction Distribution',
                            polar=dict(radialaxis=dict(showticklabels=False)))
    st.plotly_chart(fig_angle, use_container_width=True)

# Mitosis
with mitosis_tab:
    if 'mitosis_status' in df_view.columns:
        counts = df_view['mitosis_status'].astype(str).value_counts().reset_index()
        counts.columns = ['mitosis', 'count']
        st.plotly_chart(px.bar(counts, x='mitosis', y='count', text_auto=True,
                               title='Mitosis Events Count'),
                        use_container_width=True)
    else:
        st.info('No mitosis_status column found in this dataset.')

# Roles
with roles_tab:
    if 'role' in df_view.columns:
        mean_stats = df_view.groupby('role')[NUMERIC_COLS].mean().reset_index()
        fig_roles = go.Figure()
        for _, row in mean_stats.iterrows():
            fig_roles.add_trace(go.Scatterpolar(
                r=row[NUMERIC_COLS].values,
                theta=NUMERIC_COLS,
                fill='toself',
                name=row['role']
            ))
        fig_roles.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                title='Average Feature Profile per Role')
        st.plotly_chart(fig_roles, use_container_width=True)
    else:
        st.info("No 'role' column available in the dataset.")

# Raw data
with raw_tab:
    st.dataframe(df_view, use_container_width=True, height=500)
    st.download_button('💾 Download Filtered CSV',
                       df_view.to_csv(index=False).encode(),
                       file_name='filtered_cells.csv',
                       mime='text/csv')
    selected_ids = st.multiselect('Inspect cell id(s)', df_view['id'].unique())
    for cid in selected_ids:
        st.write(f'### Cell {cid}')
        st.table(df_view[df_view['id'] == cid].T)
