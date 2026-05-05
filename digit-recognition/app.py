import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import os

# ====================== 配置 ======================
IMAGE_SIZE = 28
NUM_CLASSES = 10
MODEL_PATH = Path("model.pth")

# ====================== CNN 模型结构 ======================
class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

# ====================== 加载模型 ======================
def load_model():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        
        model = DigitCNN()
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        mean = checkpoint.get("mean", 0.1307) if isinstance(checkpoint, dict) else 0.1307
        std = checkpoint.get("std", 0.3081) if isinstance(checkpoint, dict) else 0.3081
        
        print("✅ 模型加载成功！")
        return model, float(mean), float(std)
    
    except Exception as e:
        print(f"⚠️  加载模型失败：{e}")
        model = DigitCNN()
        model.eval()
        return model, 0.1307, 0.3081

MODEL, MEAN, STD = load_model()

# ====================== 预处理 ======================
def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")
    image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    
    array = np.array(image, dtype=np.float32) / 255.0
    if array.mean() > 0.5:
        array = 1.0 - array
    
    array = (array - MEAN) / STD
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()

# ====================== 预测 ======================
@torch.no_grad()
def predict_digit(img_editor):
    try:
        if img_editor is None:
            return "请先绘制数字", {str(i): 0 for i in range(10)}
        
        if isinstance(img_editor, dict):
            image = img_editor.get("composite")
        else:
            image = img_editor

        tensor = preprocess(image)
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        pred = int(probs.argmax())
        result = f"预测：{pred}   置信度：{probs[pred]:.1%}"
        prob_dict = {str(i): float(probs[i]) for i in range(10)}
        return result, prob_dict

    except Exception as e:
        return f"错误：{str(e)}", {str(i):0 for i in range(10)}

# ====================== 界面 ======================
with gr.Blocks(title="手写数字识别") as demo:
    gr.Markdown("# 🖌️ 手写数字识别")

    draw_panel = gr.ImageEditor(
        label="绘制区",
        width=360,
        height=360,
        brush=gr.Brush(color="black", size=14),
        eraser=gr.Eraser(size=25),
        interactive=True
    )

    with gr.Row():
        result_text = gr.Label(label="识别结果")
        prob_label = gr.Label(label="概率分布", num_top_classes=3)

    btn = gr.Button("🔍 开始识别", variant="primary")
    btn.click(predict_digit, inputs=draw_panel, outputs=[result_text, prob_label])

# ====================== 启动（云平台适配） ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
