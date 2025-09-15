#!/usr/bin/env python3
"""
Script para rodar o visualizador IMU + Vídeo
"""
import subprocess
import sys
import os

def main():
    # Caminho para o visualizador
    viewer_path = os.path.join(os.path.dirname(__file__), 'imu_logger', 'imu_video_3dviewer.py')
    
    # Verifica se o arquivo existe
    if not os.path.exists(viewer_path):
        print(f"Erro: Visualizador não encontrado em {viewer_path}")
        sys.exit(1)
    
    # Comando para rodar o visualizador
    cmd = [sys.executable, viewer_path] + sys.argv[1:]
    
    print("🚀 Iniciando Visualizador IMU + Vídeo...")
    print("📊 Controles:")
    print("  ESPAÇO: Play/Pause")
    print("  F: Próximo frame")
    print("  B: Frame anterior")
    print("  Q: Sair")
    print("⏹️  Pressione Ctrl+C para parar")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Visualizador encerrado!")

if __name__ == "__main__":
    main()
