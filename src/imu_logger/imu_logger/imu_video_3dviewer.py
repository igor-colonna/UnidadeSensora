#!/usr/bin/env python3
"""
Visualizador IMU + Vídeo Sincronizado
Exibe vídeo com overlay de dados IMU, gráfico de sinais e visualização 3D da orientação.
"""

import cv2
import pandas as pd
import numpy as np
import os
import glob
from typing import Optional, Tuple
import argparse

# ADICIONADO: Importações para visualização 3D com PyQtGraph e Scipy para rotações
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication
from scipy.spatial.transform import Rotation as R


class IMUVideoViewer:
    ACTIVE_VIEWER = None
    def __init__(self, csv_path: str, video_path: str):
        """
        Inicializa o visualizador
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

        # ADICIONADO: Atributos para a visualização 3D
        self.pg_app = None
        self.pg_win = None
        self.imu_axes = None

    # NOVO: Método para criar a malha 3D de uma caixa retangular
    def create_imu_box_mesh(self):
        """
        Cria um GLMeshItem representando a IMU como uma caixa retangular,
        definindo seus 8 vértices e 12 faces triangulares manualmente.
        Este método é o correto e não depende de funções inexistentes.
        """
        # Dimensões da caixa (comprimento, largura, altura)
        l, w, h = 0.1, 0.05, 0.02

        # 8 vértices (cantos) da caixa centrada na origem
        verts = np.array([
            [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2], [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
            [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2], [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
        ])

        # 12 faces triangulares (2 por cada lado do cubo)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], # Face -Z (inferior)
            [4, 6, 5], [4, 7, 6], # Face +Z (superior)
            [0, 4, 5], [0, 5, 1], # Face -Y (traseira)
            [2, 6, 7], [2, 7, 3], # Face +Y (frontal)
            [0, 7, 4], [0, 3, 7], # Face -X (esquerda)
            [1, 5, 6], [1, 6, 2], # Face +X (direita)
        ])
        
        # Cores para cada uma das 12 faces triangulares
        colors = np.array([
            (0.5, 0.5, 0.5, 0.8), (0.5, 0.5, 0.5, 0.8), # Cinza
            (0,   0,   1,   0.8), (0,   0,   1,   0.8), # Azul (+Z)
            (0.5, 0.5, 0.5, 0.8), (0.5, 0.5, 0.5, 0.8), # Cinza
            (0,   1,   0,   0.8), (0,   1,   0,   0.8), # Verde (+Y)
            (0.5, 0.5, 0.5, 0.8), (0.5, 0.5, 0.5, 0.8), # Cinza
            (1,   0,   0,   0.8), (1,   0,   0,   0.8), # Vermelho (+X)
        ])
        
        # Cria o item de malha com os dados definidos manualmente
        mesh = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=colors,
            smooth=False,
            drawEdges=True,
            edgeColor=(1,1,1,0.5)
        )
        return mesh

        
    def load_data(self) -> bool:
        """
        Carrega dados do CSV e vídeo
        """
        try:
            print(f"Carregando dados de: {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            self.df = self.df.dropna(subset=['orient_x', 'orient_y', 'orient_z', 'orient_w'])
            self.df['timestamp_sec'] = self.df['sec'] + self.df['nanosec'] / 1e9
            print(f"Dados carregados: {len(self.df)} registros")
            print(f"Período: {self.df['timestamp_sec'].min():.3f}s a {self.df['timestamp_sec'].max():.3f}s")
            
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

    # MODIFICADO: setup_3d_viewer agora usa a malha da caixa
    def setup_3d_viewer(self):
        self.pg_app = QApplication.instance() or QApplication([])
        self.pg_win = gl.GLViewWidget()
        self.pg_win.setWindowTitle('IMU Orientation 3D')
        self.pg_win.setGeometry(100, 100, 600, 600)
        self.pg_win.setCameraPosition(distance=0.5) # Diminui a distância para ver a caixa melhor
        self.pg_win.addItem(gl.GLGridItem())

        # Cria a caixa retangular e a adiciona à cena
        self.imu_mesh_object = self.create_imu_box_mesh()
        self.pg_win.addItem(self.imu_mesh_object)

        # Adiciona também um sistema de eixos no canto para referência global
        axes = gl.GLAxisItem(size=pg.Vector(0.1,0.1,0.1))
        self.pg_win.addItem(axes)

        self.pg_win.show()
        print("Visualizador 3D com modelo de caixa inicializado.")

    # MODIFICADO: update_3d_view agora aplica a transformação na caixa
    def update_3d_view(self, imu_data: Optional[pd.Series]):
        if self.pg_win is None or imu_data is None:
            return
        
        quat = [imu_data['orient_x'], imu_data['orient_y'], imu_data['orient_z'], imu_data['orient_w']]
        try:
            rotation_matrix = R.from_quat(quat).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            # Aplica a transformação ao nosso novo objeto de malha
            self.imu_mesh_object.setTransform(pg.Transform3D(transform_matrix))
        except Exception:
            pass
        
        self.pg_app.processEvents()

    def get_current_imu_data(self) -> Optional[pd.Series]:
        if self.df is None or len(self.df) == 0:
            return None
        current_frame_data = self.df[self.df['frame_index'] == self.current_frame_idx]
        if not current_frame_data.empty:
            return current_frame_data.iloc[0]
        closest_idx = (self.df['frame_index'] - self.current_frame_idx).abs().idxmin()
        return self.df.loc[closest_idx]
    
    def draw_text_overlay(self, frame: np.ndarray, imu_data: pd.Series) -> np.ndarray:
        if imu_data is None:
            return frame
        timestamp = imu_data['timestamp_sec']
        frame_idx = imu_data['frame_index']
        texts = [f"Tempo: {timestamp:.3f}s", f"Frame: {int(frame_idx)}"]
        overlay = frame.copy()
        box_w, box_h = 260, 20 + len(texts) * 22
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (20, 28 + i * 22), self.overlay_font, self.font_scale, self.text_color, self.font_thickness)
        return frame
    
    # ADICIONADO: Função auxiliar para calcular escalas dos gráficos
    def _calculate_group_ranges(self, hist_df):
        group_cols = {
            'lin_acc': [c for c in hist_df.columns if c.startswith('lin_acc_')],
            'ang_vel': [c for c in hist_df.columns if c.startswith('ang_vel_')],
            'orient':  [c for c in hist_df.columns if c.startswith('orient_')],
        }
        group_ranges = {}
        for g, cols in group_cols.items():
            if cols and not hist_df[cols].empty:
                vals = hist_df[cols].values.astype(np.float32)
                gmin, gmax = np.nanmin(vals), np.nanmax(vals)
                if gmax == gmin: gmax = gmin + 1.0
                group_ranges[g] = (float(gmin), float(gmax))
        return group_ranges

    def draw_strip_plots(self, frame: np.ndarray) -> np.ndarray:
        # CORRIGIDO: Esta função tinha bugs no código original (NameError e return prematuro)
        height, width = frame.shape[:2]
        num_plots = len(self.signal_list)
        base_y = height - (num_plots * self.strip_height + self.graph_margin)
        base_x = width - self.strip_width - self.graph_margin

        start_idx = max(0, self.current_frame_idx - self.history_len)
        end_idx = min(len(self.df), self.current_frame_idx + 1)
        hist = self.df.iloc[start_idx:end_idx]
        if len(hist) < 2:
            return frame

        group_ranges = self._calculate_group_ranges(hist) # CORRIGIDO: Definição de group_ranges

        for i, (col, color) in enumerate(self.signal_list):
            y0 = base_y + i * self.strip_height
            cv2.rectangle(frame, (base_x, y0), (base_x + self.strip_width, y0 + self.strip_height), self.bg_color, -1)
            cv2.rectangle(frame, (base_x, y0), (base_x + self.strip_width, y0 + self.strip_height), (80, 80, 80), 1)

            vals = hist[col].values.astype(np.float32)
            group = col.split('_')[0]
            vmin, vmax = group_ranges.get(group, (np.min(vals), np.max(vals)))
            if vmax == vmin: vmax = vmin + 1.0

            def y_from_val(v, y_base, h, val_min, val_max):
                if val_max - val_min == 0: return y_base + h // 2
                scale = (v - val_min) / (val_max - val_min)
                return y_base + (h - 6) - int(scale * (h - 12))

            for k in range(1, len(vals)):
                x1 = base_x + 5 + int((k - 1) * (self.strip_width - 10) / (len(vals) - 1))
                x2 = base_x + 5 + int(k * (self.strip_width - 10) / (len(vals) - 1))
                y1 = y_from_val(vals[k-1], y0, self.strip_height, vmin, vmax)
                y2 = y_from_val(vals[k], y0, self.strip_height, vmin, vmax)
                cv2.line(frame, (x1, y1), (x2, y2), color, 1)

            cv2.putText(frame, col, (base_x + 8, y0 + 18), self.overlay_font, 0.5, (220, 220, 220), 1)
            cv2.putText(frame, f"{vals[-1]:.3f}", (base_x + self.strip_width - 80, y0 + 18), self.overlay_font, 0.5, color, 1)
        
        return frame # CORRIGIDO: return movido para o final da função

    def render_plots_canvas(self) -> np.ndarray:
        num_plots = len(self.signal_list)
        canvas_h = num_plots * self.strip_height + self.graph_margin * 2
        canvas_w = self.strip_width + self.graph_margin * 2
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        base_x, base_y = self.graph_margin, self.graph_margin
        start_idx = max(0, self.current_frame_idx - self.history_len)
        end_idx = min(len(self.df), self.current_frame_idx + 1)
        hist = self.df.iloc[start_idx:end_idx]
        if len(hist) < 2: return canvas

        group_ranges = self._calculate_group_ranges(hist)

        for i, (col, color) in enumerate(self.signal_list):
            y0 = base_y + i * self.strip_height
            cv2.rectangle(canvas, (base_x, y0), (base_x + self.strip_width, y0 + self.strip_height), (20, 20, 20), -1)
            cv2.rectangle(canvas, (base_x, y0), (base_x + self.strip_width, y0 + self.strip_height), (80, 80, 80), 1)

            if col not in hist.columns: continue
            vals = hist[col].values.astype(np.float32)
            group = col.split('_')[0]
            vmin, vmax = group_ranges.get(group, (np.min(vals), np.max(vals)))
            if vmax == vmin: vmax = vmin + 1.0

            def y_from_val(v, y_base, h, val_min, val_max):
                if val_max - val_min == 0: return y_base + h // 2
                scale = (v - val_min) / (val_max - val_min)
                return y_base + (h - 6) - int(scale * (h - 12))

            for k in range(1, len(vals)):
                x1 = base_x + 5 + int((k - 1) * (self.strip_width - 10) / (len(vals) - 1))
                x2 = base_x + 5 + int(k * (self.strip_width - 10) / (len(vals) - 1))
                y1 = y_from_val(vals[k-1], y0, self.strip_height, vmin, vmax)
                y2 = y_from_val(vals[k], y0, self.strip_height, vmin, vmax)
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

            cv2.putText(canvas, col, (base_x + 8, y0 + 18), self.overlay_font, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas, f"{vals[-1]:.3f}", (base_x + self.strip_width - 80, y0 + 18), self.overlay_font, 0.5, color, 1)

        cv2.line(canvas, (base_x + self.strip_width - 12, base_y), (base_x + self.strip_width - 12, base_y + num_plots * self.strip_height), (200, 200, 200), 2)
        return canvas
 
    def draw_controls_info(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        controls = ["Controles:", "ESPACO: Play/Pause", "F: Proximo frame", "B: Frame anterior", "Q: Sair"]
        overlay = frame.copy()
        box_w, box_h = 260, 20 + len(controls) * 18 + 8
        cv2.rectangle(overlay, (width - box_w - 10, 10), (width - 10, 10 + box_h), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (width - box_w, 28 + i * 20), self.overlay_font, 0.4, self.text_color, 1)
        status = "PLAYING" if self.is_playing else "PAUSED"
        color = (0, 255, 0) if self.is_playing else (0, 0, 255)
        cv2.putText(frame, status, (width - box_w, 10 + box_h - 8), self.overlay_font, 0.6, color, 2)
        return frame
    
    def main_loop(self):
        if not self.load_data(): return
        
        print("\n=== Controles ===", "ESPACO: Play/Pause", "F/B: Frame Fwd/Back", "Q: Sair", "S/A: +/-5s", "D/W: +/-1s", "================\n", sep='\n')

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

        self.setup_3d_viewer()
        
        while True:
            if not self.pg_win.isVisible():
                print("Janela 3D fechada. Encerrando.")
                break

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                self.current_frame_idx = 0
                continue
            
            imu_data = self.get_current_imu_data()
            
            frame = self.draw_text_overlay(frame, imu_data)
            frame = self.draw_controls_info(frame)
            # frame = self.draw_strip_plots(frame) # Descomente para desenhar plots no vídeo
            
            h, w = frame.shape[:2]
            bar_h, bar_y = 12, h - 22
            ratio = self.current_frame_idx / float(self.total_frames - 1) if self.total_frames > 1 else 0
            cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + bar_h), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, bar_y), (10 + int(ratio * (w - 20)), bar_y + bar_h), (0, 180, 255), -1)

            cv2.imshow('IMU Video Viewer', frame)
            cv2.imshow('IMU Signals', self.render_plots_canvas())
            self.update_3d_view(imu_data)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'): break
            elif key == ord(' '): self.is_playing = not self.is_playing
            elif key == ord('f') and not self.is_playing: self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
            elif key == ord('b') and not self.is_playing: self.current_frame_idx = max(0, self.current_frame_idx - 1)
            
            fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
            if key == ord('s'): self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + int(5 * fps))
            elif key == ord('a'): self.current_frame_idx = max(0, self.current_frame_idx - int(5 * fps))
            elif key == ord('d'): self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + int(1 * fps))
            elif key == ord('w'): self.current_frame_idx = max(0, self.current_frame_idx - int(1 * fps))
            
            if self.is_playing:
                self.current_frame_idx += 1
                if self.current_frame_idx >= self.total_frames: self.current_frame_idx = 0
        
        self.cap.release()
        cv2.destroyAllWindows()
        if self.pg_win: self.pg_win.close()
        if self.pg_app: self.pg_app.quit()
        print("Visualizador encerrado.")

def find_latest_files(base_dir: str) -> Tuple[Optional[str], Optional[str]]:
    csv_pattern = os.path.join(base_dir, 'merged_*.csv')
    csv_files = glob.glob(csv_pattern)
    if not csv_files: return None, None
    latest_csv = max(csv_files, key=os.path.getctime)
    
    video_files = []
    for ext in ['*.mp4', '*.mkv', '*.avi']:
        video_files.extend(glob.glob(os.path.join(base_dir, '**', ext), recursive=True))
    
    return latest_csv, (sorted(video_files)[0] if video_files else None)

def main():
    parser = argparse.ArgumentParser(description='Visualizador IMU + Vídeo Sincronizado')
    parser.add_argument('--csv', type=str, help='Caminho para o arquivo CSV merged')
    parser.add_argument('--video', type=str, help='Caminho para o arquivo de vídeo')
    parser.add_argument('--base-dir', type=str, default='~/imu_logs', help='Diretório base para busca automática')
    
    args = parser.parse_args()
    
    if args.csv and args.video:
        csv_path, video_path = args.csv, args.video
    else:
        base_dir = os.path.expanduser(args.base_dir)
        csv_path, video_path = find_latest_files(base_dir)
        if csv_path is None:
            print(f"Erro: Nenhum arquivo merged_*.csv encontrado em {base_dir}"); return
        if video_path is None:
            print(f"Erro: Nenhum arquivo de vídeo encontrado em {base_dir}"); return
    
    print(f"Usando CSV: {csv_path}\nUsando Vídeo: {video_path}")
    
    viewer = IMUVideoViewer(csv_path, video_path)
    viewer.main_loop()

if __name__ == "__main__":
    main()