#!/bin/bash

# Caminho do diretório onde o rosbag está gravando
BAG_DIR="imu_logs"

# Tempo entre verificações (em segundos)
INTERVAL=10

# Último tamanho conhecido
LAST_SIZE=0

echo "📡 Monitorando gravação do rosbag em '$BAG_DIR' a cada $INTERVAL segundos..."
echo "Pressione Ctrl+C para sair."
echo "-------------------------------------------------------------"

while true; do
    if ! pgrep -f "ros2 bag record" > /dev/null; then
        echo "❌ ros2 bag record NÃO está rodando!"
    else
        echo "✅ ros2 bag record está rodando."
    fi

    # Verifica o tamanho do diretório
    if [ -d "$BAG_DIR" ]; then
        SIZE=$(du -sb "$BAG_DIR" | awk '{print $1}')
        if [ "$SIZE" -eq "$LAST_SIZE" ]; then
            echo "⚠️  Tamanho do diretório não mudou. Verifique se ainda está gravando."
        else
            echo "📈 Tamanho atual: $((SIZE / 1024 / 1024)) MB"
            LAST_SIZE=$SIZE
        fi
    else
        echo "⚠️  Diretório '$BAG_DIR' não encontrado."
    fi

    echo "-------------------------------------------------------------"
    sleep $INTERVAL
done

