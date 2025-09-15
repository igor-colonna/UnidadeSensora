#!/usr/bin/env python3
"""
Visualizador IMU + Vídeo Sincronizado
Exibe vídeo com overlay de dados IMU e gráfico de aceleração em tempo real
"""

import cv2
import pandas as pd
import numpy as np
import os
import glob
from typing import Optional, Tuple
import argparse


class IMUVideoViewer:
    ACTIVE_VIEWER = None
    def __init__(self, csv_path: str, video_path: str):
        """
        Inicializa o visualizador
        
        Args:
            csv_path: Caminho para o arquivo CSV merged
            video_path: Caminho para o arquivo de vídeo
        """
        self.csv_path = csv_path
        self.video_path = video_path
        self.df = None
        self.cap = None
        self.current_frame_idx = 0
        self.is_playing = False
        self.total_frames = 0
        
        # Configurações de exibição
        self.overlay_font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.55
        self.font_thickness = 2
        self.text_color = (255, 255, 255)  # Branco
        self.bg_color = (0, 0, 0)  # Preto
        self.line_thickness = 1

        # Lista de sinais a plotar (um gráfico por dado)
        self.signal_list = [
            ('lin_acc_x', (0, 0, 255)),
            ('lin_acc_y', (0, 255, 0)),
            ('lin_acc_z', (255, 0, 0)),
            ('ang_vel_x', (0, 170, 255)),
            ('ang_vel_y', (0, 255, 170)),
            ('ang_vel_z', (170, 0, 255)),
            ('orient_x', (200, 200, 50)),
            ('orient_y', (50, 200, 200)),
            ('orient_z', (200, 50, 200)),
            ('orient_w', (180, 180, 180)),
        ]

        # Configurações dos gráficos em tiras
        self.strip_height = 50
        self.strip_width = 600
        self.graph_margin = 12
        self.history_len = 200  # número de amostras no histórico
        
    def load_data(self) -> bool:
        """
        Carrega dados do CSV e vídeo
        
        Returns:
            bool: True se carregou com sucesso, False caso contrário
        """
        try:
            # Carrega CSV
            print(f"Carregando dados de: {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            
            # Remove linhas com dados IMU vazios
            self.df = self.df.dropna(subset=['orient_x', 'orient_y', 'orient_z', 'orient_w'])
            
            # Converte timestamp para segundos
            self.df['timestamp_sec'] = self.df['sec'] + self.df['nanosec'] / 1e9
            
            print(f"Dados carregados: {len(self.df)} registros")
            print(f"Período: {self.df['timestamp_sec'].min():.3f}s a {self.df['timestamp_sec'].max():.3f}s")
            
            # Carrega vídeo
            print(f"Carregando vídeo: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                print(f"Erro: Não foi possível abrir o vídeo {self.video_path}")
                return False
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Vídeo carregado: {self.total_frames} frames, {fps:.2f} FPS")
            
            return True
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False
    
    def get_current_imu_data(self) -> Optional[pd.Series]:
        """
        Obtém dados da IMU para o frame atual
        
        Returns:
            pd.Series: Dados da IMU ou None se não encontrado
        """
        if self.df is None or len(self.df) == 0:
            return None
        
        # Busca dados mais próximos ao frame atual
        current_frame_data = self.df[self.df['frame_index'] == self.current_frame_idx]
        
        if len(current_frame_data) > 0:
            return current_frame_data.iloc[0]
        
        # Se não encontrou frame exato, busca o mais próximo
        closest_idx = (self.df['frame_index'] - self.current_frame_idx).abs().idxmin()
        return self.df.iloc[closest_idx]
    
    def draw_text_overlay(self, frame: np.ndarray, imu_data: pd.Series) -> np.ndarray:
        """
        Desenha overlay de texto com dados da IMU
        
        Args:
            frame: Frame do vídeo
            imu_data: Dados da IMU para o frame atual
            
        Returns:
            np.ndarray: Frame com overlay de texto
        """
        if imu_data is None:
            return frame
        
        # Prepara textos
        timestamp = imu_data['timestamp_sec']
        frame_idx = imu_data['frame_index']
        
        texts = [
            f"Tempo: {timestamp:.3f}s",
            f"Frame: {int(frame_idx)}",
        ]

        # Fundo canto superior esquerdo
        overlay = frame.copy()
        box_w = 260
        box_h = 20 + len(texts) * 22
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y_offset = 28
        for text in texts:
            cv2.putText(frame, text, (20, y_offset), self.overlay_font, self.font_scale, self.text_color, self.font_thickness)
            y_offset += 22
        
        return frame
    
    def draw_strip_plots(self, frame: np.ndarray) -> np.ndarray:
        """
        Desenha gráficos em tiras empilhadas (um por dado) na parte inferior.
        """
        height, width = frame.shape[:2]
        num_plots = len(self.signal_list)
        total_h = num_plots * self.strip_height + self.graph_margin
        base_y = height - total_h
        base_x = width - self.strip_width - self.graph_margin

        start_idx = max(0, self.current_frame_idx - self.history_len)
        end_idx = min(len(self.df), self.current_frame_idx + 1)
        hist = self.df.iloc[start_idx:end_idx]
        if len(hist) < 2:
            return frame

        # Para cada sinal, desenha uma tira
        for i, (col, color) in enumerate(self.signal_list):
            y0 = base_y + i * self.strip_height
            # fundo
            cv2.rectangle(frame, (base_x, y0), (base_x + self.strip_width, y0 + self.strip_height), self.bg_color, -1)
            cv2.rectangle(frame, (base_x, y0), (base_x + self.strip_width, y0 + self.strip_height), (80, 80, 80), 1)

            vals = hist[col].values.astype(np.float32)
            # Seleciona escala por grupo
            if col.startswith('lin_acc_'):
                vmin, vmax = group_ranges.get('lin_acc', (float(np.min(vals)), float(np.max(vals))))
            elif col.startswith('ang_vel_'):
                vmin, vmax = group_ranges.get('ang_vel', (float(np.min(vals)), float(np.max(vals))))
            elif col.startswith('orient_'):
                vmin, vmax = group_ranges.get('orient', (float(np.min(vals)), float(np.max(vals))))
            else:
                vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax == vmin:
                vmax = vmin + 1.0
            def y_from_val(v):
                # margem interna de 6px
                h = self.strip_height - 12
                yy = int((v - vmin) / (vmax - vmin) * h)
                return y0 + (self.strip_height - 6) - yy

            # desenha linha
            n = len(vals)
            for k in range(1, n):
                x1 = base_x + int((k - 1) * (self.strip_width - 10) / (n - 1)) + 5
                x2 = base_x + int(k * (self.strip_width - 10) / (n - 1)) + 5
                y1 = y_from_val(vals[k - 1])
                y2 = y_from_val(vals[k])
                cv2.line(frame, (x1, y1), (x2, y2), color, 1)

            # labels
            cv2.putText(frame, col, (base_x + 8, y0 + 18), self.overlay_font, 0.5, (220, 220, 220), 1)
            cv2.putText(frame, f"{vals[-1]:.3f}", (base_x + self.strip_width - 80, y0 + 18), self.overlay_font, 0.5, color, 1)

            return frame

    def render_plots_canvas(self) -> np.ndarray:
        """
        Renderiza os gráficos em uma imagem separada para exibir em outra janela.
        """
        # Tamanho do canvas
        num_plots = len(self.signal_list)
        canvas_h = num_plots * self.strip_height + self.graph_margin * 2
        canvas_w = self.strip_width + self.graph_margin * 2
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        base_x = self.graph_margin
        base_y = self.graph_margin

        start_idx = max(0, self.current_frame_idx - self.history_len)
        end_idx = min(len(self.df), self.current_frame_idx + 1)
        hist = self.df.iloc[start_idx:end_idx]
        if len(hist) < 2:
            return canvas

        # Calcula faixa (min,max) por grupo para manter mesma escala por tipo
        group_cols = {
            'lin_acc': [c for c in hist.columns if c.startswith('lin_acc_')],
            'ang_vel': [c for c in hist.columns if c.startswith('ang_vel_')],
            'orient':  [c for c in hist.columns if c.startswith('orient_')],
        }
        group_ranges = {}
        for g, cols in group_cols.items():
            if cols:
                vals = hist[cols].values.astype(np.float32)
                gmin = float(np.nanmin(vals))
                gmax = float(np.nanmax(vals))
                if gmax == gmin:
                    gmax = gmin + 1.0
                group_ranges[g] = (gmin, gmax)

        for i, (col, color) in enumerate(self.signal_list):
            y0 = base_y + i * self.strip_height
            # fundo
            cv2.rectangle(canvas, (base_x, y0), (base_x + self.strip_width, y0 + self.strip_height), (20, 20, 20), -1)
            cv2.rectangle(canvas, (base_x, y0), (base_x + self.strip_width, y0 + self.strip_height), (80, 80, 80), 1)

            vals = hist[col].values.astype(np.float32)
            # Seleciona escala por grupo
            if col.startswith('lin_acc_'):
                vmin, vmax = group_ranges.get('lin_acc', (float(np.min(vals)), float(np.max(vals))))
            elif col.startswith('ang_vel_'):
                vmin, vmax = group_ranges.get('ang_vel', (float(np.min(vals)), float(np.max(vals))))
            elif col.startswith('orient_'):
                vmin, vmax = group_ranges.get('orient', (float(np.min(vals)), float(np.max(vals))))
            else:
                vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax == vmin:
                vmax = vmin + 1.0
            def y_from_val(v):
                h = self.strip_height - 12
                yy = int((v - vmin) / (vmax - vmin) * h)
                return y0 + (self.strip_height - 6) - yy

            n = len(vals)
            for k in range(1, n):
                x1 = base_x + int((k - 1) * (self.strip_width - 10) / (n - 1)) + 5
                x2 = base_x + int(k * (self.strip_width - 10) / (n - 1)) + 5
                y1 = y_from_val(vals[k - 1])
                y2 = y_from_val(vals[k])
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

            cv2.putText(canvas, col, (base_x + 8, y0 + 18), self.overlay_font, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas, f"{vals[-1]:.3f}", (base_x + self.strip_width - 80, y0 + 18), self.overlay_font, 0.5, color, 1)

        # marcador de posição atual (linha branca à direita)
        cv2.line(canvas, (base_x + self.strip_width - 12, base_y), (base_x + self.strip_width - 12, base_y + num_plots * self.strip_height), (200, 200, 200), 2)

        return canvas
 
    def draw_controls_info(self, frame: np.ndarray) -> np.ndarray:
        """
        Desenha informações de controle na tela
        
        Args:
            frame: Frame do vídeo
        
        Returns:
            np.ndarray: Frame com informações de controle
        """
        height, width = frame.shape[:2]
        
        # Informações de controle
        controls = [
            "Controles:",
            "ESPACO: Play/Pause",
            "F: Proximo frame",
            "B: Frame anterior",
            "Q: Sair"
        ]
        
        # Desenha fundo (canto superior direito)
        overlay = frame.copy()
        box_w = 260
        box_h = 20 + len(controls) * 18 + 8
        cv2.rectangle(overlay, (width - box_w - 10, 10), (width - 10, 10 + box_h), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Desenha textos
        y_offset = 28
        for control in controls:
            cv2.putText(frame, control, (width - box_w, y_offset), 
                        self.overlay_font, 0.4, self.text_color, 1)
            y_offset += 20
        
        # Status de reprodução
        status = "PLAYING" if self.is_playing else "PAUSED"
        color = (0, 255, 0) if self.is_playing else (0, 0, 255)
        cv2.putText(frame, status, (width - box_w, 10 + box_h - 8), 
                    self.overlay_font, 0.6, color, 2)
        
        return frame
    
    def main_loop(self):
        """
        Loop principal do visualizador
        """
        if not self.load_data():
            return
        
        print("\n=== Controles ===")
        print("ESPACO: Play/Pause")
        print("F: Proximo frame")
        print("B: Frame anterior")
        print("Q: Sair")
        print("S: +5s, A: -5s, D: +1s, W: -1s")
        print("Clique na barra inferior para buscar")
        print("================\n")

        # Callback de mouse para seek pelo progresso
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # barra de progresso: 12px de altura no rodapé
                h, w = frame.shape[:2] if 'frame' in locals() else (720, 1280)
                bar_h = 12
                if y >= h - bar_h - 10 and y <= h - 10:
                    ratio = max(0.0, min(1.0, (x - 10) / float(max(1, w - 20))))
                    self.current_frame_idx = int(ratio * (self.total_frames - 1))
        
        IMUVideoViewer.ACTIVE_VIEWER = self
        cv2.namedWindow('IMU Video Viewer', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('IMU Video Viewer', on_mouse)
        
        while True:
            # Obtém frame atual do vídeo
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"Erro ao ler frame {self.current_frame_idx}")
                break
            
            # Obtém dados da IMU para o frame atual
            imu_data = self.get_current_imu_data()
            
            # Desenha overlays
            frame = self.draw_text_overlay(frame, imu_data)
            frame = self.draw_controls_info(frame)
            #frame = self.draw_strip_plots(frame)
            
            # Desenha barra de progresso e tempo atual
            h, w = frame.shape[:2]
            bar_h = 12
            ratio = 0 if self.total_frames <= 1 else self.current_frame_idx / float(self.total_frames - 1)
            x0, y0 = 10, h - bar_h - 10
            x1 = int(x0 + ratio * (w - 20))
            # fundo barra
            cv2.rectangle(frame, (x0, y0), (w - 10, y0 + bar_h), (40, 40, 40), -1)
            # progresso
            cv2.rectangle(frame, (x0, y0), (x1, y0 + bar_h), (0, 180, 255), -1)
            # tempos
            cur_sec = self.current_frame_idx / max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
            total_sec = self.total_frames / max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
            cv2.putText(frame, f"{cur_sec:6.2f}s / {total_sec:6.2f}s", (x0, y0 - 8), self.overlay_font, 0.5, (220,220,220), 1)

            # Exibe janelas
            cv2.imshow('IMU Video Viewer', frame)
            plots = self.render_plots_canvas()
            cv2.imshow('IMU Signals', plots)
            
            # Captura teclas
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Espaço
                self.is_playing = not self.is_playing
                print(f"Status: {'PLAYING' if self.is_playing else 'PAUSED'}")
            elif key == ord('f'):  # Próximo frame
                if not self.is_playing and self.current_frame_idx < self.total_frames - 1:
                    self.current_frame_idx += 1
                    print(f"Frame: {self.current_frame_idx}")
            elif key == ord('b'):  # Frame anterior
                if not self.is_playing and self.current_frame_idx > 0:
                    self.current_frame_idx -= 1
                    print(f"Frame: {self.current_frame_idx}")
            # saltos por tempo
            elif key == ord('s'):  # +5s
                fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
                self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + int(5 * fps))
            elif key == ord('a'):  # -5s
                fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
                self.current_frame_idx = max(0, self.current_frame_idx - int(5 * fps))
            elif key == ord('d'):  # +1s
                fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
                self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + int(1 * fps))
            elif key == ord('w'):  # -1s
                fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
                self.current_frame_idx = max(0, self.current_frame_idx - int(1 * fps))
            
            # Auto-avanço quando playing
            if self.is_playing:
                self.current_frame_idx += 1
                if self.current_frame_idx >= self.total_frames:
                    self.current_frame_idx = 0  # Loop
        
        # Limpeza
        self.cap.release()
        cv2.destroyAllWindows()
        print("Visualizador encerrado.")


def find_latest_files(base_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Encontra o CSV merged mais recente e um vídeo correspondente
    
    Args:
        base_dir: Diretório base para busca
        
    Returns:
        Tuple: (caminho_csv, caminho_video) ou (None, None) se não encontrado
    """
    # Busca CSV merged mais recente
    csv_pattern = os.path.join(base_dir, 'merged_*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        return None, None
    
    latest_csv = max(csv_files, key=os.path.getctime)
    
    # Busca vídeos em subdiretórios
    video_extensions = ['*.mp4', '*.mkv', '*.avi']
    video_files = []
    for ext in video_extensions:
        pattern = os.path.join(base_dir, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    if not video_files:
        return latest_csv, None
    
    # Retorna o primeiro vídeo encontrado
    return latest_csv, sorted(video_files)[0]


def main():
    """
    Função principal
    """
    parser = argparse.ArgumentParser(description='Visualizador IMU + Vídeo Sincronizado')
    parser.add_argument('--csv', type=str, help='Caminho para o arquivo CSV merged')
    parser.add_argument('--video', type=str, help='Caminho para o arquivo de vídeo')
    parser.add_argument('--base-dir', type=str, default='~/imu_logs', 
                       help='Diretório base para busca automática de arquivos')
    
    args = parser.parse_args()
    
    # Determina caminhos dos arquivos
    if args.csv and args.video:
        csv_path = args.csv
        video_path = args.video
    else:
        # Busca automática
        base_dir = os.path.expanduser(args.base_dir)
        csv_path, video_path = find_latest_files(base_dir)
        
        if csv_path is None:
            print(f"Erro: Nenhum arquivo merged_*.csv encontrado em {base_dir}")
            return
        
        if video_path is None:
            print(f"Erro: Nenhum arquivo de vídeo encontrado em {base_dir}")
            return
    
    print(f"CSV: {csv_path}")
    print(f"Vídeo: {video_path}")
    
    # Cria e executa visualizador
    viewer = IMUVideoViewer(csv_path, video_path)
    viewer.main_loop()


if __name__ == "__main__":
    main()
