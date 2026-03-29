# 导入YOLOv8核心库
from ultralytics import YOLO
import torch
import torch.nn as nn
import os  # 用于校验文件是否存在
print(os.getcwd())

# ====================== 注意力机制模块：SE Attention ======================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ====================== 添加SE注意力的YOLOv8 C2f模块修改 ======================
def add_se_to_model(model):
    """将SE注意力模块注入到YOLOv8的C2f模块中"""
    for name, module in model.model.named_modules():
        if hasattr(module, 'cv2') and 'c2f' in name.lower():
            # 在C2f模块的Bottleneck后添加SE模块
            if hasattr(module, 'm'):
                # 为每个bottleneck添加SE注意力
                for i, bottleneck in enumerate(module.m):
                    # 在bottleneck的卷积层后添加SE
                    if hasattr(bottleneck, 'cv2'):
                        channels = bottleneck.cv2.out_channels
                        se = SEBlock(channels)
                        # 将SE模块作为属性附加到bottleneck
                        bottleneck.se = se
                        # 修改forward方法
                        original_forward = bottleneck.forward
                        def new_forward(x, orig=original_forward, se=bottleneck.se):
                            return se(orig(x))
                        bottleneck.forward = new_forward
    print("✓ 已为模型添加SE注意力机制")
    return model

# ====================== 关键修复：适配ultralytics 8.4.24的禁用下载方案 ======================
# 方案1：屏蔽requests库的下载请求（兜底，彻底阻止所有网络请求）
import requests
def blocked_request(*args, **kwargs):
    raise requests.exceptions.RequestException("Auto-download disabled")
# 临时替换requests.get，阻止YOLO的自动下载
original_get = requests.get
requests.get = blocked_request

# 方案2：设置环境变量（兼容旧版本）
os.environ["YOLO_VERBOSE"] = "False"  # 关闭冗余日志
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 避免无关的MPS报错

# ====================== 第一步：验证GPU环境（确认CUDA可用）======================
print("===== 检测运行环境 =====")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 校验GPU环境，无CUDA则提示
if not torch.cuda.is_available():
    raise RuntimeError("❌ 未检测到可用的CUDA设备！请确认：\n1. 安装了带CUDA的PyTorch\n2. 显卡驱动正常\n3. 显卡支持CUDA（NVIDIA显卡）")

# 打印GPU详细信息
gpu_count = torch.cuda.device_count()
current_gpu = torch.cuda.current_device()
gpu_name = torch.cuda.get_device_name(current_gpu)
print(f"GPU数量: {gpu_count}")
print(f"当前使用GPU: {current_gpu} ({gpu_name})")
print(f"当前运行设备: GPU (CUDA加速版本)\n")

# ====================== 第二步：核心配置（路径保持不变，适配GPU）======================
DATA_YAML_PATH = r"data_yms/test.yaml"  # 数据集配置
MODEL_PATH = r"data_yms/yolov8s.pt"     # 本地预训练模型路径
SAVE_DIR = r"data_yms/uav_campus_train"  # 训练结果保存路径
TEST_IMG_PATH = r"data_yms/images/val"  # 测试集路径

# 关键：校验本地模型/数据集路径是否存在，避免报错
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 本地模型文件不存在！请检查路径：{MODEL_PATH}\n建议重新下载模型并放到该路径")
if not os.path.exists(DATA_YAML_PATH):
    raise FileNotFoundError(f"❌ 数据集YAML文件不存在！请检查路径：{DATA_YAML_PATH}")
print(f"✅ 本地文件校验成功：\n   - 模型：{MODEL_PATH}\n   - 数据集：{DATA_YAML_PATH}\n")

# ====================== 第三步：GPU适配训练参数（核心优化，修复显存问题）======================
EPOCHS = 100         # GPU训练轮数（可适当增加，充分利用算力）
IMG_SIZE = 640       # 输入图像尺寸（GPU可支持更大，如800/1024）
BATCH_SIZE = 16      # Tesla T4 16G显存适配（32会OOM，16是安全值）
DEVICE = 0           # 使用第0块GPU（多GPU可设为[0,1]，CPU是"cpu"）

# ====================== 第四步：加载本地模型+添加注意力机制+开始GPU训练 ======================
print("===== 加载本地YOLOv8s模型，添加注意力机制 =====")
model = YOLO(MODEL_PATH)  # 加载本地模型，自动移到GPU

# 添加SE注意力机制
model = add_se_to_model(model)

# 先定义关键常量（建议根据300张样本调整）
EPOCHS = 5  # 快速测试：降低轮次快速走完流程
IMG_SIZE = 640  # 降低分辨率，加快训练速度
BATCH_SIZE = 8  # 小数据集batch减小，梯度更新更稳定
DEVICE = 0  # 你的GPU设备ID

# 优化后的训练参数（核心改动已标注）
train_results = model.train(
    data=DATA_YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    mosaic=0.5,        # 【改动】1.0→0.5，降低马赛克增强强度，避免特征失真
    copy_paste=0.2,    # 【改动】0.5→0.2，降低copy_paste强度，适配小样本
    mixup=0.0,         # 【改动】0.2→0.0，关闭mixup（小数据集易过拟合）
    project=os.path.dirname(SAVE_DIR),
    name=os.path.basename(SAVE_DIR),
    exist_ok=True,
    save=True,
    patience=30,       # 【改动】20→30，延长早停耐心值，避免提前终止
    lr0=0.005,         # 【改动】0.01→0.005，降低初始学习率，提升收敛稳定性
    lrf=0.01,          # 新增：学习率衰减因子（默认0.01，保持即可）
    weight_decay=0.0005,
    workers=4,
    pretrained=True,
    amp=False,         # 【改动】禁用AMP混合精度，避免下载检查
    cache="ram",       # 保留：内存缓存，不占用显存
    # 新增2个关键参数：提升框回归精度
    box=8.0,           # 【新增】框损失权重（默认7.5→8.0，强化框回归）
    iou=0.75,          # 【新增】NMS的IOU阈值（默认0.7→0.75，更严格的框筛选）
)

# 恢复requests.get（训练完成后不影响其他功能）
requests.get = original_get

# ====================== 第五步：GPU验证模型精度 ======================
print("\n===== 开始验证训练后模型精度 =====")
val_results = model.val(
    data=DATA_YAML_PATH,
    imgsz=IMG_SIZE,
    device=DEVICE,
    batch=BATCH_SIZE,
    workers=4,
    cache="ram",
)
# 打印核心精度指标
print(f"\n===== 模型训练精度汇总 =====")
print(f"mAP50（IOU=0.5）: {val_results.box.map50:.4f}")
print(f"mAP50-95（全IOU）: {val_results.box.map:.4f}\n")

# ====================== 第六步：GPU预测+保存结果 ======================
print("===== 开始对测试集进行GPU预测 =====")
predict_results = model.predict(
    source=TEST_IMG_PATH,
    imgsz=IMG_SIZE,
    device=DEVICE,
    save=True,         # 保存带检测框的图片
    save_txt=True,     # 保存检测框坐标（txt文件）
    conf=0.5,          # 置信度阈值
    iou=0.45,          # IOU阈值
    show_labels=True,
    show_conf=True,
    verbose=False,     # 关闭冗余日志
    workers=4,
)

# 打印所有结果保存路径
print(f"\n===== 所有任务完成，结果保存路径 =====")
print(f"📌 训练好的模型：{SAVE_DIR}/weights/best.pt（最佳精度模型）")
print(f"📌 检测结果图片：{predict_results[0].save_dir}（带检测框的测试图）")
print(f"📌 检测框坐标：{predict_results[0].save_dir}/labels（每张图的txt标注文件）")