import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração do Roboflow usando variáveis de ambiente
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE")).project(os.getenv("ROBOFLOW_PROJECT"))
version = project.version(1)
dataset = version.download("yolov8")


from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Treinamento com o dataset baixado
model.train(
    data="Face-Detection-1/data.yaml", 
    imgsz=416,  # Tamanho da imagem reduzido
    epochs=5, 
    batch=8,
    workers=6,  # Menos workers
    device=0    # usa GPU 
)