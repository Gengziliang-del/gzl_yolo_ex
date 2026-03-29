import os
from PIL import Image

# ====================== 配置参数 ======================
SOURCE_DIR = "source_data"  # 源图片文件夹
OUTPUT_DIR = "out"  # 输出文件夹
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')  # 支持的图片格式

# ====================== 创建输出文件夹 ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ 输出文件夹已创建：{OUTPUT_DIR}")

# ====================== 遍历并处理图片 ======================
image_count = 0
success_count = 0

for filename in os.listdir(SOURCE_DIR):
    if not filename.lower().endswith(SUPPORTED_FORMATS):
        continue
    
    image_count += 1
    input_path = os.path.join(SOURCE_DIR, filename)
    
    try:
        # 打开图片
        img = Image.open(input_path)
        width, height = img.size
        
        # 计算中心点
        center_x, center_y = width // 2, height // 2
        
        # 计算四个部分的尺寸
        quarter_w = width // 2
        quarter_h = height // 2
        
        # 定义四个裁切区域的坐标 (left, top, right, bottom)
        # 左上
        crop1 = (0, 0, center_x, center_y)
        # 右上
        crop2 = (center_x, 0, width, center_y)
        # 左下
        crop3 = (0, center_y, center_x, height)
        # 右下
        crop4 = (center_x, center_y, width, height)
        
        # 生成输出文件名（去除扩展名）
        name_without_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        
        # 裁切并保存四个部分
        crops = [
            (crop1, f"{name_without_ext}_part1{ext}"),
            (crop2, f"{name_without_ext}_part2{ext}"),
            (crop3, f"{name_without_ext}_part3{ext}"),
            (crop4, f"{name_without_ext}_part4{ext}")
        ]
        
        for crop_coords, out_filename in crops:
            cropped_img = img.crop(crop_coords)
            output_path = os.path.join(OUTPUT_DIR, out_filename)
            cropped_img.save(output_path)
        
        success_count += 1
        print(f"✓ 已处理：{filename} → 4部分")
        
    except Exception as e:
        print(f"✗ 处理失败：{filename} - {str(e)}")

# ====================== 输出统计结果 ======================
print(f"\n===== 处理完成 =====")
print(f"找到图片：{image_count} 张")
print(f"成功处理：{success_count} 张")
print(f"裁切后总数：{success_count * 4} 张")
print(f"输出路径：{os.path.abspath(OUTPUT_DIR)}")
