#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from typing import List, Tuple, Optional
import time

# Configurações
BASE_DIR = os.path.expanduser('~/imu_logs')
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 360


def find_latest_files(base_dir: str) -> Tuple[Optional[str], List[str]]:
    """Encontra o CSV merged mais recente e os vídeos correspondentes"""
    # Busca CSV merged mais recente
    csv_pattern = os.path.join(base_dir, 'merged_*.csv')
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        return None, []
    
    latest_csv = max(csv_files, key=os.path.getctime)
    
    # Busca vídeos em subdiretórios
    video_extensions = ['*.mp4', '*.mkv', '*.avi']
    video_files = []
    for ext in video_extensions:
        pattern = os.path.join(base_dir, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return latest_csv, sorted(video_files)


def load_data(csv_path: str) -> pd.DataFrame:
    """Carrega e processa o CSV merged"""
    df = pd.read_csv(csv_path)
    
    # Converte timestamp para datetime
    df['timestamp'] = pd.to_datetime(df['sec'] * 1e9 + df['nanosec'], unit='ns')
    df['timestamp_sec'] = df['sec'] + df['nanosec'] / 1e9
    
    # Remove linhas com dados IMU vazios
    df = df.dropna(subset=['orient_x', 'orient_y', 'orient_z', 'orient_w'])
    
    return df


def create_imu_plots(df: pd.DataFrame, current_time: float) -> go.Figure:
    """Cria gráficos da IMU"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Orientação (Quaternion)', 'Velocidade Angular (rad/s)', 'Aceleração Linear (m/s²)'],
        vertical_spacing=0.08
    )
    
    # Orientação
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['orient_x'], name='orient_x', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['orient_y'], name='orient_y', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['orient_z'], name='orient_z', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['orient_w'], name='orient_w', line=dict(color='orange')),
        row=1, col=1
    )
    
    # Velocidade angular
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['ang_vel_x'], name='ang_vel_x', line=dict(color='red')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['ang_vel_y'], name='ang_vel_y', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['ang_vel_z'], name='ang_vel_z', line=dict(color='blue')),
        row=2, col=1
    )
    
    # Aceleração linear
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['lin_acc_x'], name='lin_acc_x', line=dict(color='red')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['lin_acc_y'], name='lin_acc_y', line=dict(color='green')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp_sec'], y=df['lin_acc_z'], name='lin_acc_z', line=dict(color='blue')),
        row=3, col=1
    )
    
    # Adiciona linha vertical na posição atual
    fig.add_vline(
        x=current_time,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Tempo: {current_time:.3f}s"
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title="Dados da IMU Sincronizados"
    )
    
    return fig


def get_video_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """Extrai frame específico do vídeo"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Converte BGR para RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def main():
    st.set_page_config(page_title="IMU Dashboard", layout="wide")
    
    st.title("📊 Dashboard IMU + Vídeo Sincronizado")
    
    # Inicializa session state
    if 'current_time' not in st.session_state:
        st.session_state.current_time = None
    if 'auto_play' not in st.session_state:
        st.session_state.auto_play = False
    if 'play_speed' not in st.session_state:
        st.session_state.play_speed = 0.1
    
    # Sidebar para seleção de arquivos
    st.sidebar.header("📁 Arquivos")
    
    # Busca arquivos automaticamente
    csv_path, video_files = find_latest_files(BASE_DIR)
    
    if csv_path is None:
        st.error(f"Nenhum arquivo merged_*.csv encontrado em {BASE_DIR}")
        return
    
    st.sidebar.success(f"CSV: {os.path.basename(csv_path)}")
    
    # Seleção de vídeo
    if not video_files:
        st.error("Nenhum arquivo de vídeo encontrado")
        return
    
    selected_video = st.sidebar.selectbox(
        "Selecione o vídeo:",
        video_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Carrega dados
    with st.spinner("Carregando dados..."):
        df = load_data(csv_path)
    
    st.sidebar.info(f"Total de registros: {len(df)}")
    st.sidebar.info(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    # Filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtro por tópico
    topics = df['topic'].unique()
    selected_topic = st.sidebar.selectbox("Tópico:", topics)
    df_filtered = df[df['topic'] == selected_topic].copy()
    
    # Filtro por tempo
    time_range = st.sidebar.slider(
        "Período (segundos):",
        min_value=float(df_filtered['timestamp_sec'].min()),
        max_value=float(df_filtered['timestamp_sec'].max()),
        value=(float(df_filtered['timestamp_sec'].min()), float(df_filtered['timestamp_sec'].max())),
        step=0.1
    )
    
    df_filtered = df_filtered[
        (df_filtered['timestamp_sec'] >= time_range[0]) & 
        (df_filtered['timestamp_sec'] <= time_range[1])
    ]
    
    if len(df_filtered) == 0:
        st.error("Nenhum dado no período selecionado")
        return
    
    # Inicializa current_time se não estiver definido
    if st.session_state.current_time is None:
        st.session_state.current_time = float(df_filtered['timestamp_sec'].min())
    
    # Controles de reprodução
    st.sidebar.header("▶️ Controles")
    
    # Slider de tempo
    current_time = st.sidebar.slider(
        "Tempo (s):",
        min_value=float(df_filtered['timestamp_sec'].min()),
        max_value=float(df_filtered['timestamp_sec'].max()),
        value=st.session_state.current_time,
        step=0.1,
        key="time_slider"
    )
    
    # Atualiza session state quando slider muda
    if current_time != st.session_state.current_time:
        st.session_state.current_time = current_time
    
    # Botões de controle
    col1, col2, col3, col4 = st.sidebar.columns(4)
    
    with col1:
        if st.button("⏮️", help="Primeiro frame"):
            st.session_state.current_time = float(df_filtered['timestamp_sec'].min())
            st.rerun()
    
    with col2:
        if st.button("⏪", help="Frame anterior"):
            prev_times = df_filtered[df_filtered['timestamp_sec'] < st.session_state.current_time]['timestamp_sec']
            if len(prev_times) > 0:
                st.session_state.current_time = float(prev_times.max())
                st.rerun()
    
    with col3:
        if st.button("⏩", help="Próximo frame"):
            next_times = df_filtered[df_filtered['timestamp_sec'] > st.session_state.current_time]['timestamp_sec']
            if len(next_times) > 0:
                st.session_state.current_time = float(next_times.min())
                st.rerun()
    
    with col4:
        if st.button("⏭️", help="Último frame"):
            st.session_state.current_time = float(df_filtered['timestamp_sec'].max())
            st.rerun()
    
    # Auto-play
    auto_play = st.sidebar.checkbox("Auto-play", value=st.session_state.auto_play)
    st.session_state.auto_play = auto_play
    
    if auto_play:
        play_speed = st.sidebar.slider("Velocidade (s):", 0.01, 1.0, st.session_state.play_speed, 0.01)
        st.session_state.play_speed = play_speed
        
        # Auto-play usando placeholder
        placeholder = st.empty()
        
        # Encontra próximo tempo
        next_times = df_filtered[df_filtered['timestamp_sec'] > st.session_state.current_time]['timestamp_sec']
        if len(next_times) > 0:
            st.session_state.current_time = float(next_times.min())
        else:
            st.session_state.current_time = float(df_filtered['timestamp_sec'].min())
        
        # Pausa para simular velocidade
        time.sleep(play_speed)
        st.rerun()
    
    # Encontra frame mais próximo ao tempo atual
    closest_idx = (df_filtered['timestamp_sec'] - st.session_state.current_time).abs().idxmin()
    current_frame_data = df_filtered.loc[closest_idx]
    current_frame_index = int(current_frame_data['frame_index'])
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráficos da IMU
        fig = create_imu_plots(df_filtered, st.session_state.current_time)
        
        # Exibe gráfico
        st.plotly_chart(fig, use_container_width=True, key="imu_plot")
    
    with col2:
        # Vídeo
        st.subheader("🎥 Vídeo")
        
        # Extrai frame atual
        frame = get_video_frame(selected_video, current_frame_index)
        
        if frame is not None:
            # Redimensiona frame para exibição
            height, width = frame.shape[:2]
            if width > DEFAULT_VIDEO_WIDTH:
                scale = DEFAULT_VIDEO_WIDTH / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            st.image(frame, caption=f"Frame {current_frame_index}", use_column_width=True)
        else:
            st.error(f"Erro ao carregar frame {current_frame_index}")
        
        # Informações do frame atual
        st.subheader("📊 Dados do Frame")
        st.write(f"**Tempo:** {st.session_state.current_time:.3f} s")
        st.write(f"**Frame:** {current_frame_index}")
        st.write(f"**Arquivo:** {os.path.basename(current_frame_data['video_file'])}")
        
        # Dados da IMU
        st.write("**Orientação (Quaternion):**")
        st.write(f"X: {current_frame_data['orient_x']:.4f}")
        st.write(f"Y: {current_frame_data['orient_y']:.4f}")
        st.write(f"Z: {current_frame_data['orient_z']:.4f}")
        st.write(f"W: {current_frame_data['orient_w']:.4f}")
        
        st.write("**Velocidade Angular (rad/s):**")
        st.write(f"X: {current_frame_data['ang_vel_x']:.4f}")
        st.write(f"Y: {current_frame_data['ang_vel_y']:.4f}")
        st.write(f"Z: {current_frame_data['ang_vel_z']:.4f}")
        
        st.write("**Aceleração Linear (m/s²):**")
        st.write(f"X: {current_frame_data['lin_acc_x']:.4f}")
        st.write(f"Y: {current_frame_data['lin_acc_y']:.4f}")
        st.write(f"Z: {current_frame_data['lin_acc_z']:.4f}")


if __name__ == "__main__":
    main()
