# YOLO Face Detection - Transfer Learning com Roboflow e Ultralytics

Este projeto realiza **detecção facial** utilizando **YOLOv8** da Ultralytics e **transfer learning** em cima de um modelo pré-treinado. O dataset é obtido automaticamente da plataforma [Roboflow](https://roboflow.com/), já no formato compatível com YOLOv8.

## 🔍 O que o projeto faz

- Utiliza **transfer learning** com YOLOv8 (`yolov8n.pt`) para detectar rostos em imagens.
- Treina com um dataset facial obtido via API do Roboflow.
- Faz **predição** em imagens locais após o treinamento.
- Permite visualização e salvamento das imagens com as detecções feitas.

---

## 📁 Estrutura do Projeto

```

.
├── transfer\_yolo\_train.py      # Código para baixar dataset e treinar o modelo
├── transfer\_yolo\_predict.py    # Código para carregar modelo treinado e fazer predições
├── .env                        # Armazena as credenciais da API Roboflow (não subir no GitHub)
└── runs/                       # Pasta gerada com resultados de treino (auto-gerada)

````

---

## ⚙️ Pré-requisitos
| se preferir, é possivel rodar todo o codigo pelo google colab, mas lembre-se de rodar os comandos "!pip install roboflow ultranalytics" na primeira célula

- Python 3.8+
- GPU com CUDA (opcional, mas recomendado)
- Conta no [Roboflow](https://roboflow.com/) (grátis)

Instale as dependências:

```bash
pip install ultralytics roboflow python-dotenv
````

---

## 🔐 Configuração (.env)

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_WORKSPACE=your_workspace_name
ROBOFLOW_PROJECT=your_project_name
```

Você encontra esses dados no painel da Roboflow, após criar ou clonar um dataset.

---

## 🚀 Como treinar o modelo

Execute o script de treinamento:

```bash
python transfer_yolo_train.py
```

Isso irá:

* Baixar automaticamente o dataset do Roboflow
* Treinar o modelo com `yolov8n.pt` por 50 épocas
* Salvar o modelo treinado em `runs/detect/train/weights/best.pt`

---

## 🔎 Como fazer predições

copie o caminho da imagem desejada e coloque no arquivo `transfer_yolo_predict.py`:

```python
imagens = ["/caminho/para/imagem.jpg"]
```

E execute:

```bash
python transfer_yolo_predict.py
```

O script irá:

* Carregar o modelo treinado
* Detectar rostos na imagem
* Exibir as imagens com bounding boxes
* Salvar as imagens com as detecções automaticamente

---

## 📌 Observações

* Você pode ajustar parâmetros do treinamento no script `transfer_yolo_train.py`, como `epochs`, `imgsz`, `batch`, etc.
* O modelo usado é o `YOLOv8n` (versão leve). Pode trocar por `yolov8s.pt`, `yolov8m.pt` etc. para mais precisão.

---

## 🧠 Créditos

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Roboflow](https://roboflow.com/)
