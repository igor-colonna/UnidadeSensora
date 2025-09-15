#!/usr/bin/env python3
"""
Script para rodar o dashboard IMU + Vídeo (versão corrigida)
"""
import subprocess
import sys
import os

def main():
    # Caminho para o dashboard corrigido
    dashboard_path = os.path.join(os.path.dirname(__file__), 'imu_logger', 'imu_dashboard_fixed.py')
    
    # Verifica se o arquivo existe
    if not os.path.exists(dashboard_path):
        print(f"Erro: Dashboard não encontrado em {dashboard_path}")
        sys.exit(1)
    
    # Comando para rodar o Streamlit
    cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path, '--server.port=8501', '--server.address=0.0.0.0']
    
    print("🚀 Iniciando Dashboard IMU + Vídeo (versão corrigida)...")
    print("📊 Acesse: http://localhost:8501")
    print("⏹️  Pressione Ctrl+C para parar")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard encerrado!")

if __name__ == "__main__":
    main()
