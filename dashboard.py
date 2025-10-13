import streamlit as st
import polars as pl
import plotly.express as px
from pathlib import Path
import sys

# Add utils to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.file_handling import get_child_subchilds_tuples
# from utils.polars import (
#     # load_participant_session_data,  # This will be defined within the dashboard for now
#     get_trial,
# )
# from utils.plots import plot_time_series, plot_psd_heatmap, plot_average_psd

st.set_page_config(layout="wide")

st.title("iEEG Data Analysis Dashboard")

# --- Data Loading and Selection ---

# NOTE TO USER: Please specify the correct path to your preprocessed data directory.
DATA_PATH = Path("preprocessed_data")

@st.cache_data
def get_available_sessions():
    """Gets a list of available participant and session tuples."""
    if not DATA_PATH.exists():
        st.error(f"Data directory not found at: {DATA_PATH}. Please create it.")
        return []
    # NOTE TO USER: This uses the function you provided.
    return get_child_subchilds_tuples(DATA_PATH)

# NOTE TO USER: This is a placeholder for your actual data loading function.
# You will need to replace this with your actual data loading logic from your notebook.
@st.cache_data
def load_participant_session_data(participant_id: str, session: str):
    """
    Loads the session data for a given participant and session.
    This is a placeholder and returns a dummy DataFrame.
    """
    st.info(f"Loading data for {participant_id}, {session}...")

    # NOTE TO USER: Here you should call your actual data loading function,
    # which likely takes participant_id and session as input and returns a Polars DataFrame.
    # For example:
    # from my_analysis_utils import load_data
    # return load_data(participant_id, session)

    # Creating a dummy dataframe for demonstration purposes.
    dummy_data = {
        'trial': [1, 1, 1, 2, 2, 2, 3, 3],
        'dbs_on': [True, True, True, False, False, False, True, True],
        'lfp_ch_1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'lfp_ch_2': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'ecog_ch_1': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        'ecog_ch_2': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        'x_coord': [10, 11, 12, 13, 14, 15, None, 17],
        'y_coord': [20, 21, 22, 23, 24, 25, None, 27],
        'z_coord': [30, 31, 32, 33, 34, 35, None, 37],
        'training_speed': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        'time': [0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1]
    }
    return pl.DataFrame(dummy_data)

# --- Sidebar for Selection ---
st.sidebar.header("Session Selection")

available_sessions = get_available_sessions()

if not available_sessions:
    st.sidebar.warning("No sessions found. Check the `preprocessed_data` directory.")
else:
    # Create a user-friendly dropdown from the tuples
    session_options = [f"{p.split('=')[1]} - {s.split('=')[1]}" for p, s in available_sessions]
    selected_option = st.sidebar.selectbox("Select Participant and Session", options=session_options)

    if selected_option:
        p_id_str, session_str = selected_option.split(" - ")

        # Find the corresponding tuple to get the full participant and session strings
        selected_tuple = next(((p, s) for p, s in available_sessions if p.endswith(p_id_str) and s.endswith(session_str)), None)

        if selected_tuple and st.sidebar.button("Load Data"):
            participant_id, session = selected_tuple
            session_data = load_participant_session_data(participant_id, session)

            if not session_data.is_empty():
                st.session_state['session_data'] = session_data
                st.session_state['participant_id'] = participant_id
                st.session_state['session'] = session
                st.sidebar.success(f"Data loaded for {p_id_str}, Session {session_str}")
            else:
                st.sidebar.error("Failed to load data.")

# --- Main Panel for Analysis ---
st.header("Trial Analysis")

if 'session_data' in st.session_state:
    session_data = st.session_state['session_data']

    # Get a list of available trials
    trials = sorted(session_data['trial'].unique().to_list())

    # Get a list of LFP and ECoG channels from the dataframe columns
    lfp_channels = sorted([col for col in session_data.columns if 'lfp' in col.lower()])
    ecog_channels = sorted([col for col in session_data.columns if 'ecog' in col.lower()])

    for trial_num in trials:
        trial_data = session_data.filter(pl.col('trial') == trial_num)
        dbs_on = trial_data['dbs_on'][0]

        with st.expander(f"Trial {trial_num} (DBS: {'ON' if dbs_on else 'OFF'})", expanded=trial_num == trials[0]):
            st.markdown(f"### Details for Trial {trial_num}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Channel Time Series")
                selected_lfp = st.selectbox(f"LFP Channel (Trial {trial_num})", lfp_channels, key=f"lfp_{trial_num}")
                selected_ecog = st.selectbox(f"ECoG Channel (Trial {trial_num})", ecog_channels, key=f"ecog_{trial_num}")

                # NOTE TO USER: Replace with your actual time series plotting function
                if selected_lfp:
                    fig_lfp = px.line(trial_data.to_pandas(), x='time', y=selected_lfp, title=f"LFP: {selected_lfp}")
                    st.plotly_chart(fig_lfp, use_container_width=True)

                if selected_ecog:
                    fig_ecog = px.line(trial_data.to_pandas(), x='time', y=selected_ecog, title=f"ECoG: {selected_ecog}")
                    st.plotly_chart(fig_ecog, use_container_width=True)

            with col2:
                st.subheader("Coordinates and Speed")

                # Plot coordinates vs time
                coords_over_time = trial_data.select(['time', 'x_coord', 'y_coord', 'z_coord']).drop_nulls()
                if not coords_over_time.is_empty():
                    fig_coords_time = px.line(coords_over_time.to_pandas(), x='time', y=['x_coord', 'y_coord', 'z_coord'], title="Coordinates over Time")
                    st.plotly_chart(fig_coords_time, use_container_width=True)
                else:
                    st.warning("No coordinate data available for this trial.")

                # Plot coordinates in 3D space (no time)
                coords_no_time = trial_data.select(['x_coord', 'y_coord', 'z_coord']).drop_nulls()
                if not coords_no_time.is_empty() and 'z_coord' in coords_no_time.columns:
                    fig_coords_3d = px.scatter_3d(coords_no_time.to_pandas(), x='x_coord', y='y_coord', z='z_coord', title="3D Trajectory")
                    st.plotly_chart(fig_coords_3d, use_container_width=True)
                elif not coords_no_time.is_empty():
                     fig_coords_2d = px.scatter(coords_no_time.to_pandas(), x='x_coord', y='y_coord', title="2D Trajectory")
                     st.plotly_chart(fig_coords_2d, use_container_width=True)

                # Plot raw training speed
                speed_data = trial_data.select(['time', 'training_speed']).drop_nulls()
                if not speed_data.is_empty():
                    fig_speed = px.line(speed_data.to_pandas(), x='time', y='training_speed', title="Raw Training Speed")
                    st.plotly_chart(fig_speed, use_container_width=True)

            st.markdown("---") # Horizontal line for separation

    # --- PSD Analysis Section ---
    st.header("Power Spectral Density (PSD) Analysis")

    # NOTE TO USER: The following sections for PSD require you to have pre-calculated
    # PSD data. The dummy data below is for layout purposes only. You will need to
    # load your PSD data and replace the placeholder plots with your actual
    # visualization functions (e.g., from utils.plots).

    # --- PSD Heatmaps with Channel Selection ---
    st.subheader("PSD Heatmaps per Channel")
    col1_psd, col2_psd = st.columns(2)
    with col1_psd:
        selected_lfp_psd = st.selectbox("Select LFP Channel for PSD Heatmap", lfp_channels, key="lfp_psd_heatmap")
        # NOTE TO USER: Replace with your plot_psd_heatmap function
        # Example: fig = plot_psd_heatmap(psd_data_lfp, channel=selected_lfp_psd)
        # st.plotly_chart(fig)
        st.info(f"Placeholder for LFP PSD Heatmap for channel: {selected_lfp_psd}")
        st.image("https://via.placeholder.com/400x300.png?text=LFP+PSD+Heatmap", use_column_width=True)

    with col2_psd:
        selected_ecog_psd = st.selectbox("Select ECoG Channel for PSD Heatmap", ecog_channels, key="ecog_psd_heatmap")
        # NOTE TO USER: Replace with your plot_psd_heatmap function
        # Example: fig = plot_psd_heatmap(psd_data_ecog, channel=selected_ecog_psd)
        # st.plotly_chart(fig)
        st.info(f"Placeholder for ECoG PSD Heatmap for channel: {selected_ecog_psd}")
        st.image("https://via.placeholder.com/400x300.png?text=ECoG+PSD+Heatmap", use_column_width=True)

    st.markdown("---")

    # --- Average PSD Plots ---
    st.subheader("Average PSD Analysis")

    # NOTE TO USER: You should have dataframes for average PSDs:
    # avg_psd_no_ica_lfp, avg_psd_no_ica_ecog, avg_psd_with_ica_lfp, etc.
    # The dataframes should have 'freq' and 'power' columns.

    # Dummy data for average PSD plots
    dummy_psd_df = pl.DataFrame({
        'freq': range(0, 101, 5),
        'power_lfp': [1 / (f + 1) for f in range(0, 101, 5)],
        'power_ecog': [1.5 / (f + 1) for f in range(0, 101, 5)]
    })

    col1_avg, col2_avg = st.columns(2)
    with col1_avg:
        st.markdown("#### Average PSD (Without ICA)")
        # LFP Average PSD (No ICA)
        fig_lfp_no_ica = px.line(dummy_psd_df.to_pandas(), x='freq', y='power_lfp', title="Average LFP PSD (No ICA)")
        st.plotly_chart(fig_lfp_no_ica, use_container_width=True)
        # ECoG Average PSD (No ICA)
        fig_ecog_no_ica = px.line(dummy_psd_df.to_pandas(), x='freq', y='power_ecog', title="Average ECoG PSD (No ICA)")
        st.plotly_chart(fig_ecog_no_ica, use_container_width=True)

    with col2_avg:
        st.markdown("#### Average PSD (With ICA)")
        # LFP Average PSD (With ICA)
        fig_lfp_with_ica = px.line(dummy_psd_df.to_pandas(), x='freq', y='power_lfp', title="Average LFP PSD (With ICA)")
        st.plotly_chart(fig_lfp_with_ica, use_container_width=True)
        # ECoG Average PSD (With ICA)
        fig_ecog_with_ica = px.line(dummy_psd_df.to_pandas(), x='freq', y='power_ecog', title="Average ECoG PSD (With ICA)")
        st.plotly_chart(fig_ecog_with_ica, use_container_width=True)

else:
    st.info("Select a participant and session from the sidebar and click 'Load Data' to begin.")