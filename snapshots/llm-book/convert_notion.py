#!/usr/bin/env python3
"""把 Notion HTML 导出转为带 LaTeX 公式的英文底稿 Markdown（两步）：
步骤一：块级公式 figure -> 占位符；图片目录改名 assets；pandoc -> gfm。
步骤二（清理）：块公式还原 $$..$$；行内公式 span -> $..$；图片 figure -> Markdown 图片；
              剥离 Notion 版式 wrapper（div/span/icon/目录块）。"""
import glob, html, re, subprocess, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SRC = glob.glob('Alisa*book*.html')[0]
raw = open(SRC, encoding='utf-8').read()

# ---- 步骤一 ----
old_dir, new_dir = 'Alisa\u2019s book of LLMs', 'assets'
if not os.path.isdir(new_dir):
    os.rename(old_dir, new_dir)
raw = raw.replace(old_dir + '/', new_dir + '/')
raw = raw.replace('Alisa%E2%80%99s%20book%20of%20LLMs/', 'assets/')

eqs = []
def eq_repl(m):
    attrs = m.group(1)
    latex = re.search(r'data-notion-equation="([^"]*)"', attrs)
    latex = html.unescape(latex.group(1)) if latex else '??missing??'
    i = len(eqs); eqs.append(latex)
    return f'<p>EQPH{i:04d}</p>'
raw, n_eq = re.subn(r'<figure([^>]*class="equation"[^>]*)>(.*?)</figure>', eq_repl, raw, flags=re.S)
raw = re.sub(r'<style.*?</style>', '', raw, flags=re.S)

open('/tmp/notion_pre.html', 'w', encoding='utf-8').write(raw)
subprocess.run(['pandoc', '/tmp/notion_pre.html', '-f', 'html', '-t', 'gfm',
                '--wrap=none', '-o', '/tmp/notion_mid.md'], check=True)

# ---- 步骤二：清理 ----
out = open('/tmp/notion_mid.md', encoding='utf-8').read()

out = re.sub(r'\\?EQPH(\d{4})', lambda m: f'\n$$\n{eqs[int(m.group(1))]}\n$$\n', out)
out = out.replace('\\$', '$')

def inline_eq(m):
    return '$' + html.unescape(m.group(1)) + '$'
out, n_inline = re.subn(
    r'<span class="notion-text-equation-token" notion-inline-equation="([^"]*)"[^>]*>'
    r'(?:<span class="katex">[^<]*</span>)?</span>', inline_eq, out)

def img_repl(m):
    body = m.group(1)
    src = re.search(r'src="([^"]+)"', body)
    if not src:
        return ''
    cap = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', body, flags=re.S)
    cap = re.sub(r'<[^>]+>', '', cap.group(1)).strip() if cap else ''
    alt = cap or os.path.basename(html.unescape(src.group(1)))
    md = f'![{alt}]({src.group(1)})'
    return md
out = re.sub(r'<figure[^>]*class="image"[^>]*>(.*?)</figure>', img_repl, out, flags=re.S)
out = re.sub(r'<figure[^>]*>(.*?)</figure>', lambda m: m.group(1), out, flags=re.S)  # 其他残留 figure 只留内容

out = re.sub(r'<img src="https://app\.notion\.com/icons/[^"]*"[^>]*/?>', '', out)  # 页面图标
out = re.sub(r'<a href="#[0-9a-f-]{36}" class="table_of_contents-link">([^<]*)</a>', r'[TOC:\1]', out)
out = re.sub(r'<span[^>]*>|</span>', '', out)                       # 残留 span
out = re.sub(r'<div[^>]*class="[^"]*(row|column|page-body|page-header-icon|table_of_contents-item)[^"]*"[^>]*>', '', out)
out = re.sub(r'</?div[^>]*>', '', out)
out = re.sub(r'^\[TOC:([^\]]*)\]$', r'- \1', out, flags=re.M)       # 原目录块 -> 纯文本列表
out = re.sub(r'<!--\s*-->', '', out)
out = re.sub(r'\n{3,}', '\n\n', out)

header = ('<!-- 英文底稿：由 Notion HTML 导出（2026-11-15 下载，zip 见同目录）转换。'
          '管线：块公式/行内公式属性->占位符 -> pandoc 3.7 -> LaTeX 回填；图片指向 assets/（源自导出包，目录已改名）。'
          '转换脚本 convert_notion.py 可复跑。 -->\n\n')
open('alisa-book-of-llms-en.md', 'w', encoding='utf-8').write(header + out)
print(f'block equations: {n_eq}; inline equations: {n_inline}')
print('md images:', len(re.findall(r'^!\[', out, flags=re.M)))
print('headings:', len(re.findall(r'^#{1,4} ', out, flags=re.M)), '; words:', len(out.split()))
open('/tmp/toc.txt', 'w').write('\n'.join(re.findall(r'^#{1,4} .*', out, flags=re.M)))
