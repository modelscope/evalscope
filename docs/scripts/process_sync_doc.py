import os
import re
from tqdm import tqdm


def process_heading(content):
    """处理一级标题，替换为YAML front matter"""
    pattern = re.compile(r'^\s*#\s+(.+?)\s*$', re.MULTILINE)
    match = pattern.search(content)
    if match:
        title = match.group(1)
        # 只替换第一个匹配的一级标题
        content = pattern.sub('---\ntitle: ' + title + '\n---', content, count=1)
    return content

def process_code_blocks(content):
    """处理特殊代码块，转换为标准代码块"""
    pattern = re.compile(r'```\{code-block\}\s*(\w+)\s*([^\n]*)\n(.*?)```', re.DOTALL | re.MULTILINE)
    
    def replace_code_block(match):
        language = match.group(1)
        code_content = match.group(3)
        # 移除以冒号开头的参数行（如:caption:）
        lines = code_content.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith(':')]
        filtered_content = '\n'.join(filtered_lines).strip()
        return f'```{language}\n{filtered_content}\n```'
    
    return pattern.sub(replace_code_block, content)

def process_admonitions(content):
    """处理特殊提示块，转换为标准NOTE格式"""
    pattern = re.compile(r'(`{3,4})\{(\w+)\}\s*\n(.*?)\1', re.DOTALL)
    
    def replace_admonition(match):
        admonition_content = match.group(3).strip()
        # 将内容每行前加>，并添加> [!NOTE]
        lines = admonition_content.split('\n')
        formatted_lines = ['> [!NOTE]']
        for line in lines:
            if line.strip():
                formatted_lines.append('> ' + line)
            else:
                formatted_lines.append('>')
        return '\n'.join(formatted_lines)
    
    return pattern.sub(replace_admonition, content)

def process_tab_sets(content):
    """处理tab-set和tab-item，转换为小标题格式"""
    # 匹配整个tab-set块
    pattern = re.compile(r'::::\{tab-set\}(.*?)::::', re.DOTALL)
    
    def replace_tab_set(match):
        tab_set_content = match.group(1)
        
        # 匹配每个tab-item
        tab_item_pattern = re.compile(r':::\{tab-item\} (.*?)\n(.*?):::', re.DOTALL)
        
        def replace_tab_item(tab_match):
            tab_title = tab_match.group(1).strip()
            tab_content = tab_match.group(2).strip()
            
            # 将tab-item转换为小标题格式
            return f'**{tab_title}**\n\n{tab_content}\n'
        
        # 替换所有tab-item
        return tab_item_pattern.sub(replace_tab_item, tab_set_content)
    
    # 替换所有tab-set
    return pattern.sub(replace_tab_set, content)

def remove_identifier_lines(content):
    """删除Markdown文件开头的标识行，格式为(.*)=，只删除第一个匹配项"""
    pattern = re.compile(r'^\(.*\)=\s*$', re.MULTILINE)
    
    # 找到第一个匹配项的位置
    match = pattern.search(content)
    if match:
        # 计算要删除的文本范围（包括后面的换行符）
        start = match.start()
        end = match.end()
        
        # 检查后面是否有换行符，如果有则一起删除
        if end < len(content) and content[end] == '\n':
            end += 1
        
        # 删除匹配的文本
        content = content[:start] + content[end:]
    
    return content

def process_markdown_file(source_path, target_path):
    """处理单个Markdown文件"""
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 依次应用各种处理
        content = process_heading(content)
        content = process_code_blocks(content)
        content = process_admonitions(content)
        content = process_tab_sets(content)
        content = remove_identifier_lines(content)

        # 确保目标目录存在，并写入处理后的内容
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f'处理文件 {source_path} 时出错: {e}')
        return False


def get_markdown_files(source_dir, skip_dirs=None):
    """获取所有需要处理的Markdown文件列表"""
    markdown_files = []
    skip_dirs = skip_dirs or set()

    for root, dirs, files in os.walk(source_dir):
        # 检查当前目录是否包含需要跳过的目录名
        if any(skip_dir in root for skip_dir in skip_dirs):
            continue
            
        # 检查当前目录的直接子目录是否需要跳过
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.md') and file != 'index.md':
                source_path = os.path.join(root, file)
                markdown_files.append(source_path)
    
    return markdown_files

def main(source_dir, target_dir, skip_dirs=None):

    print(f'开始处理目录: {source_dir}')
    print(f'输出目录: {target_dir}')
    
    # 获取所有需要处理的Markdown文件
    markdown_files = get_markdown_files(source_dir, skip_dirs=skip_dirs)
    print(f'找到 {len(markdown_files)} 个需要处理的Markdown文件')
    
    # 使用tqdm显示进度条
    success_count = 0
    for source_path in tqdm(markdown_files, desc='处理Markdown文件'):
        # 计算目标路径
        relative_path = os.path.relpath(source_path, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        
        # 处理文件
        if process_markdown_file(source_path, target_path):
            success_count += 1
    
    print(f'处理完成！成功处理 {success_count}/{len(markdown_files)} 个文件')
    print(f'处理后的文件已保存到: {target_dir}')


def process_zh():
    source_directory = '/Users/yunlin/Code/eval-scope/docs/zh'
    target_directory = '/Users/yunlin/Code/documentation/tutorial/模型评测'
    skip_dirs = {'blog', 'experiments'}
    main(source_directory, target_directory, skip_dirs)

def process_en():
    source_directory = '/Users/yunlin/Code/eval-scope/docs/en'
    target_directory = '/Users/yunlin/Code/documentation/tutorial-en/Model Evaluation'
    skip_dirs = {'blog', 'experiments'}
    main(source_directory, target_directory, skip_dirs)

if __name__ == '__main__':
    process_zh()
    process_en()