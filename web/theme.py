# -*- coding: utf-8 -*-
"""Gradio 主题资源：从 web/static 加载 CSS 与 head（字体、脚本），供 gr.Blocks 注入。"""

import os

_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_WEB_DIR, "static")


def get_theme_css() -> str:
    """返回 PreenCut 主题 CSS 全文，供 gr.Blocks(css=...) 使用。"""
    path = os.path.join(_STATIC_DIR, "preencut_theme.css")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_theme_head() -> str:
    """返回注入到页面 <head> 的 HTML（Tech Assistant + 字体与 PreenCut 脚本），供 gr.Blocks(head=...) 使用。"""
    tech_path = os.path.join(_STATIC_DIR, "tech_assistant_head.html")
    preencut_path = os.path.join(_STATIC_DIR, "preencut_theme_head.html")
    with open(tech_path, "r", encoding="utf-8") as f:
        tech = f.read()
    with open(preencut_path, "r", encoding="utf-8") as f:
        preencut = f.read()
    return tech + "\n" + preencut
