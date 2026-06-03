# -*- coding: utf-8 -*-
"""Gradio 界面用常量：表格复选框 HTML、空占位数据、默认选项等。"""

from typing import List

from config import ENABLE_ALIGNMENT

# 表格内复选框：preencut-cell-checkbox 避免被全局透明覆盖；preencut-checkbox-checked 用于选中高亮
CHECKBOX_CHECKED = '<span class="preencut-cell-checkbox preencut-checkbox-checked" style="display:flex;width:16px;height:16px;border:2px solid #1C1917;background:#C2410C;font-weight:bold;color:#fff;align-items:center;justify-content:center">✓</span>'
CHECKBOX_UNCHECKED = '<span class="preencut-cell-checkbox" style="display:flex;width:16px;height:16px;border:2px solid #1C1917;background:#f5f5f5;font-weight:bold;color:#1C1917;align-items:center;justify-content:center"></span>'

# 任务详情表「选择」列
TASK_SELECT_CHECKED = '<span class="preencut-cell-checkbox preencut-checkbox-checked" style="display:inline-block;width:16px;height:16px;border:2px solid #1C1917;background:#C2410C;color:#fff;text-align:center;line-height:14px;font-size:11px;vertical-align:middle">✓</span>'
TASK_SELECT_UNCHECKED = '<span class="preencut-cell-checkbox" style="display:inline-block;width:16px;height:16px;border:2px solid #1C1917;background:#f5f5f5;color:#1C1917;vertical-align:middle"></span>'

# 空 Dataframe 占位，避免 Gradio 将 [] 序列化为 '' 导致 DataframeData 校验报错
EMPTY_RESULT_TABLE: List[List] = [["", "", "", "", "", "", "", ""]]
EMPTY_SEGMENT_SELECTION: List[List] = [[CHECKBOX_UNCHECKED, "", "", "", "", "", "", "", ""]]

DEFAULT_ENABLE_ALIGNMENT = "开启" if ENABLE_ALIGNMENT else "关闭"
