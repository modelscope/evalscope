import os
import re
import shutil
from tqdm import tqdm


def process_heading(content):
    """处理一级标题，替换为YAML front matter"""
    pattern = re.compile(r'^\s*#\s+(.+?)\s*$', re.MULTILINE)
    match = pattern.search(content)
    if match:
        title = match.group(1)
        # Quote the title if it contains characters that would break YAML:
        # colon (:), hash (#), or starts with a YAML indicator character.
        yaml_unsafe = re.search(r'[:\[\]{}&*!|>\'"%@`]|^[?-]', title)
        if yaml_unsafe:
            title_yaml = '"' + title.replace('\\', '\\\\').replace('"', '\\"') + '"'
        else:
            title_yaml = title
        # 只替换第一个匹配的一级标题
        content = pattern.sub('---\ntitle: ' + title_yaml + '\n---', content, count=1)
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

def remove_toctree(content):
    """Remove MyST toctree directives (:::{toctree}...:::)."""
    pattern = re.compile(r':::\{toctree\}.*?:::', re.DOTALL)
    return pattern.sub('', content)


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



# ============================================================
# Next-format sync: sync to /documentation/next/model-evaluation
# ============================================================

def to_kebab_case(s):
    """Convert snake_case string to kebab-case."""
    return s.replace('_', '-')


def compute_target_path_next(source_path, source_dir, target_base, lang_suffix):
    """
    Compute the target file path for the 'next' format.

    Non-index files are placed inside a subdirectory named after themselves:
      docs/zh/user_guides/stress_test/quick_start.md
      → <target_base>/user-guides/stress-test/quick-start/quick-start_CN.md

    index.md files are placed directly in their kebab-cased directory:
      docs/zh/user_guides/stress_test/index.md
      → <target_base>/user-guides/stress-test/index.md
    """
    rel_path = os.path.relpath(source_path, source_dir)
    parts = rel_path.replace('\\', '/').split('/')

    filename = parts[-1]
    stem = filename[:-3]  # strip .md

    dir_parts = [to_kebab_case(p) for p in parts[:-1]]

    if filename == 'index.md':
        if dir_parts:
            return os.path.join(target_base, *dir_parts, 'index.md')
        return os.path.join(target_base, 'index.md')

    stem_kebab = to_kebab_case(stem)
    return os.path.join(target_base, *dir_parts, stem_kebab, f'{stem_kebab}{lang_suffix}.md')


def transform_doc_links(content, source_path, source_dir, target_base, lang_suffix, target_path):
    """
    Transform relative .md links in content so they point to the correct
    target paths in the 'next' format.

    Only relative links ending in .md (with optional #fragment) are transformed.
    Absolute URLs and non-.md links are left unchanged.
    """
    source_file_dir = os.path.dirname(source_path)
    target_file_dir = os.path.dirname(target_path)

    def replace_link(match):
        prefix = match.group(1)   # '' or '!'
        text = match.group(2)
        href = match.group(3)

        # Image links are handled separately
        if prefix == '!':
            return match.group(0)

        # Skip absolute URLs
        if href.startswith('http://') or href.startswith('https://') or href.startswith('/'):
            return match.group(0)

        # Split off fragment anchor
        fragment = ''
        path_part = href
        if '#' in href:
            idx = href.index('#')
            fragment = href[idx:]
            path_part = href[:idx]

        # Only transform links that end with .md
        if not path_part.endswith('.md'):
            return match.group(0)

        # Resolve the absolute source path for the linked file
        abs_link = os.path.normpath(os.path.join(source_file_dir, path_part))

        # Must reside within source_dir
        rel_from_source = os.path.relpath(abs_link, source_dir)
        if rel_from_source.startswith('..'):
            return match.group(0)

        # Compute the target path for the linked file (no language suffix in links)
        linked_target = compute_target_path_next(abs_link, source_dir, target_base, '')

        # Compute relative path from the current target file's directory
        rel_link = os.path.relpath(linked_target, target_file_dir)
        rel_link = rel_link.replace('\\', '/')
        if not rel_link.startswith('.'):
            rel_link = './' + rel_link

        return f'[{text}]({rel_link}{fragment})'

    # Match optional leading '!' followed by [text](href)
    pattern = re.compile(r'(!?)\[([^\]]*)\]\(([^)]*)\)')
    return pattern.sub(replace_link, content)


def process_images_next(content, source_path, target_path, lang_suffix):
    """
    Find local image references, copy the images to a sibling _resources/
    directory with a language suffix added to the filename, and update
    the references in the content.

    CN files get images suffixed with _cn, INTL_EN files with _en.
    """
    lang_tag = 'cn' if lang_suffix == '_CN' else 'en'
    target_dir = os.path.dirname(target_path)
    resources_dir = os.path.join(target_dir, '_resources')
    source_file_dir = os.path.dirname(source_path)

    def replace_image(match):
        alt = match.group(1)
        img_path = match.group(2)

        # Skip absolute URLs
        if img_path.startswith('http://') or img_path.startswith('https://') or img_path.startswith('/'):
            return match.group(0)

        # Resolve the absolute source image path
        abs_img_path = os.path.normpath(os.path.join(source_file_dir, img_path))

        if not os.path.exists(abs_img_path):
            return match.group(0)

        # Ensure _resources directory exists
        os.makedirs(resources_dir, exist_ok=True)

        # Build new filename: stem_cn.ext or stem_en.ext
        img_name = os.path.basename(abs_img_path)
        stem, ext = os.path.splitext(img_name)
        new_img_name = f'{stem}_{lang_tag}{ext}'
        new_img_path = os.path.join(resources_dir, new_img_name)

        # Copy image
        shutil.copy2(abs_img_path, new_img_path)

        return f'![{alt}](./_resources/{new_img_name})'

    pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    return pattern.sub(replace_image, content)


def process_markdown_file_next(source_path, target_path, source_dir, target_base, lang_suffix):
    """Process a single markdown file for the 'next' target format."""
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply the existing preprocessing pipeline
        content = process_heading(content)
        content = process_code_blocks(content)
        content = process_admonitions(content)
        content = process_tab_sets(content)
        content = remove_identifier_lines(content)
        content = remove_toctree(content)

        # Transform cross-document links to new target paths
        content = transform_doc_links(content, source_path, source_dir, target_base, lang_suffix, target_path)

        # Copy local images to _resources/ and update references
        content = process_images_next(content, source_path, target_path, lang_suffix)

        # Write output
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True
    except Exception as e:
        print(f'Error processing {source_path}: {e}')
        import traceback
        traceback.print_exc()
        return False


def sync_to_next(source_dir, target_base, lang_suffix, skip_dirs=None):
    """
    Sync markdown files from source_dir to target_base using the 'next' format.

    lang_suffix: '_CN' for Chinese docs, '_INTL_EN' for English docs.
    """
    print(f'\nSyncing: {source_dir}')
    print(f'Target:  {target_base}  (suffix={lang_suffix})')

    markdown_files = get_markdown_files(source_dir, skip_dirs=skip_dirs)
    print(f'Found {len(markdown_files)} markdown files')

    success_count = 0
    for source_path in tqdm(markdown_files, desc=f'Syncing ({lang_suffix})'):
        target_path = compute_target_path_next(source_path, source_dir, target_base, lang_suffix)
        if process_markdown_file_next(source_path, target_path, source_dir, target_base, lang_suffix):
            success_count += 1

    print(f'Done! {success_count}/{len(markdown_files)} files synced')
    return success_count


def process_next():
    """Sync both Chinese and English docs to the next format target directory."""
    target_dir = '/Users/yunlin/Code/documentation/next/model-evaluation'
    skip_dirs = {'blog', 'experiments'}

    # Sync Chinese docs → _CN.md
    sync_to_next(
        source_dir='/Users/yunlin/Code/eval-scope/docs/zh',
        target_base=target_dir,
        lang_suffix='_CN',
        skip_dirs=skip_dirs,
    )

    # Sync English docs → _INTL_EN.md
    sync_to_next(
        source_dir='/Users/yunlin/Code/eval-scope/docs/en',
        target_base=target_dir,
        lang_suffix='_INTL_EN',
        skip_dirs=skip_dirs,
    )


if __name__ == '__main__':
    process_next()
