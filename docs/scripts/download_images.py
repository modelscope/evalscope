import os
import re
import requests


# 定义下载图像的函数
def download_image(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 提取文件名
    filename = url.split('/')[-1]
    filepath = os.path.join(folder, filename)
    # 下载图片
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
    return filepath


# 处理Markdown文件
def process_markdown(input_path, output_path, image_folder):
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 匹配Markdown图片链接的正则表达式
    image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    matches = image_pattern.findall(content)

    for url in matches:
        try:
            local_path = download_image(url, image_folder)
            content = content.replace(url, local_path)
        except Exception as e:
            print(f'Error downloading {url}: {e}')

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)


# 主函数
if __name__ == '__main__':
    input_markdown_file = './docs/zh/best_practice/qwen3_vl.md'  # 输入的Markdown文件路径
    output_markdown_file = './docs/zh/best_practice/qwen3_vl.md'  # 输出的Markdown文件路径
    image_folder = './docs/zh/best_practice/images'  # 保存图片的文件夹

    process_markdown(input_markdown_file, output_markdown_file, image_folder)
