import os
import json

# 假设这是包含 JSON 文件的目录路径（中文目录名）
json_dir = 'C:/Users/22132/Desktop/pic/outputs'

# 列出目录中所有 JSON 文件
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

for json_file in json_files:
    # 构建完整的文件路径
    json_path = os.path.join(json_dir, json_file)

    # 确保以 UTF-8 编码读取文件
    with open(json_path, 'r', encoding='utf-8') as f:
        item = json.load(f)
        print(f"读取文件 {json_file}:")
        print(item)  # 打印文件内容以验证读取是否正确
