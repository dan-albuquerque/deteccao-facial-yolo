from ultralytics import YOLO

# Fazer detecção em uma imagem após o treinamento
# Carrega o modelo treinado
trained_model = YOLO("runs/detect/train/weights/best.pt")

imagens = ["/home/danilo/projetosMeus/transfer_yolo/kongsberg_friends.JPG"]
for imagem in imagens:
    print(f"Processando imagem: {imagem}")
    results = trained_model.predict(
        source=imagem,
        save=True,
        conf=0.5,
        device=0
    )

    # Mostra os resultados
    for result in results:
        print(f"Detecções encontradas: {len(result.boxes)}")
        result.show()  # Exibe a imagem com as detecções
