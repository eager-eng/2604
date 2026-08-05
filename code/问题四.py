# -*- coding: utf-8 -*-
"""问题四：5 mg/Nm³约束下的可行性、电耗增幅与极值工况建议。"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


项目目录 = Path(__file__).resolve().parents[1]
问题二代码 = 项目目录 / "code" / "问题二.py"
数据路径 = 项目目录 / "code" / "outputs" / "数据预处理结果" / "建模就绪数据.csv"
问题二结果目录 = 项目目录 / "code" / "outputs" / "问题二计算结果"
结果目录 = 项目目录 / "code" / "outputs" / "问题四计算结果"
图片目录 = 项目目录 / "figures" / "问题四计算结果"
报告路径 = 项目目录 / "reports" / "问题四计算结果.md"

原排放限值 = 10.0
新排放限值 = 5.0
遗传算法重复次数 = 3
随机种子 = 20260806
电压上限扩展比例 = 0.04

粉蓝 = "#8FB7D4"
粉紫 = "#B8A2CF"
粉色 = "#D7A8B8"
青灰 = "#99C6C3"
深紫 = "#75658F"
浅灰 = "#E5E7EC"
灰色 = "#727784"


def 加载问题二模块():
    spec = importlib.util.spec_from_file_location("问题二模块", 问题二代码)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载问题二程序")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def 设置绘图风格() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#AEB3BC",
        "axes.grid": True,
        "grid.color": "#D9DCE3",
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def 保存图(fig: plt.Figure, 名称: str) -> None:
    fig.savefig(图片目录 / f"{名称}.png", dpi=400, facecolor="white")
    fig.savefig(图片目录 / f"{名称}.pdf", facecolor="white")
    plt.close(fig)


def 重建问题二模型(问题二):
    数据 = pd.read_csv(数据路径, parse_dates=["时间戳"])
    gmm模型, _, 标签 = 问题二.拟合GMM(数据)
    工况表 = 问题二.计算工况统计(数据, 标签, gmm模型)
    能耗模型, *_ = 问题二.拟合能耗模型(数据)
    标准化 = 问题二.训练标准化参数(数据)
    tobit模型, *_ = 问题二.拟合除尘指数模型(数据, 标准化)
    下界, 上界, 边界表 = 问题二.决策边界(数据)
    return 数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 下界, 上界, 边界表


def 获取工况基准(问题二, 数据, 标签, 工况, 能耗模型, tobit模型, 标准化):
    k = int(工况["工况编号"])
    训练掩码 = 数据["数据集"].eq("训练集").to_numpy()
    训练 = 数据.loc[训练掩码].copy()
    训练["工况编号"] = 标签[训练掩码] + 1
    子集 = 训练[训练["工况编号"] == k]
    基准策略 = np.r_[
        [float(子集[f"电场{i}电压_kV"].median()) for i in range(1, 5)],
        [int(round(float(子集[f"电场{i}振打周期_s"].median()))) for i in range(1, 5)],
    ]
    Xp, Xt, _ = 问题二.构造候选特征(基准策略.reshape(1, -1), 工况, 标准化)
    return 基准策略, float(能耗模型.predict(Xp)[0]), float((Xt @ tobit模型["beta"])[0])


def 评价策略(问题二, 策略, 工况, 能耗模型, tobit模型, 标准化, 基准策略, 基准除尘指数, 下界, 上界):
    pop = np.asarray(策略, dtype=float).reshape(1, -1)
    Xp, _, _ = 问题二.构造候选特征(pop, 工况, 标准化)
    电耗 = float(能耗模型.predict(Xp)[0])
    除尘指数, _, _ = 问题二.Deutsch相对除尘指数(
        pop, 基准策略, 基准除尘指数, tobit模型, 标准化)
    浓度 = float(1000.0 * float(工况["入口浓度中心_g_Nm3"]) * math.exp(-除尘指数[0]))
    vu, vt = 问题二.约束量(pop, 下界, 上界)
    return 电耗, 浓度, float(除尘指数[0]), float(vu[0]), float(vt[0])


def 边界内最大除尘策略(问题二, 工况, tobit模型, 标准化, 基准策略, 基准除尘指数, 下界, 上界):
    """利用相对模型的单调性构造边界内最大除尘指数策略。"""
    策略 = np.empty(8, dtype=float)
    策略[:4] = 上界[:4]
    频率系数 = np.asarray(tobit模型["beta"])[9:13]
    for i, beta in enumerate(频率系数):
        # f=60/tau；正系数取最小周期，负系数取最大周期。
        策略[4 + i] = 下界[4 + i] if beta >= 0 else 上界[4 + i]
    _, _, _, vu, vt = 评价策略(
        问题二, 策略, 工况, 假能耗模型,
        tobit模型, 标准化, 基准策略, 基准除尘指数, 下界, 上界)
    if vu > 1e-10 or vt > 1e-10:
        raise RuntimeError("最大除尘边界策略违反前后级分组约束，需要增加离散搜索。")
    return 策略


class _零能耗模型:
    def predict(self, X):
        return np.zeros(len(X), dtype=float)


假能耗模型 = _零能耗模型()


def 可行性检查(问题二, 数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 下界, 上界):
    行, 基准信息 = [], {}
    for _, 工况 in 工况表.iterrows():
        k = int(工况["工况编号"])
        基准策略, 基准电耗, 基准Y = 获取工况基准(
            问题二, 数据, 标签, 工况, 能耗模型, tobit模型, 标准化)
        最大策略 = 边界内最大除尘策略(
            问题二, 工况, tobit模型, 标准化, 基准策略, 基准Y, 下界, 上界)
        最大电耗, 最低浓度, 最大Y, vu, vt = 评价策略(
            问题二, 最大策略, 工况, 能耗模型, tobit模型, 标准化,
            基准策略, 基准Y, 下界, 上界)
        阈值Y = math.log(1000.0 * float(工况["入口浓度中心_g_Nm3"]) / 新排放限值)
        可行 = bool(最低浓度 <= 新排放限值 + 1e-10)
        row = {
            "典型工况": f"工况{k}", "工况编号": k,
            "训练样本比例": float(工况["训练样本比例"]),
            "入口浓度中心_g_Nm3": float(工况["入口浓度中心_g_Nm3"]),
            "入口负荷中心_kg_h": float(工况["入口负荷中心_kg_h"]),
            "5标准所需除尘指数": 阈值Y, "边界内最大除尘指数": 最大Y,
            "边界内最低预测浓度_mg_Nm3": 最低浓度,
            "5标准是否可行": int(可行), "最大除尘策略预测电耗_kW": 最大电耗,
            "分组约束满足": int(vu <= 1e-10 and vt <= 1e-10),
        }
        for i in range(4):
            row[f"最大除尘U{i+1}_kV"] = float(最大策略[i])
            row[f"最大除尘tau{i+1}_s"] = int(round(最大策略[4+i]))
        行.append(row)
        基准信息[k] = (基准策略, 基准电耗, 基准Y)
    return pd.DataFrame(行), 基准信息


def 优化5标准(问题二, 工况表, 可行表, 能耗模型, tobit模型, 标准化, 下界, 上界, 基准信息):
    原限值 = float(问题二.排放限值)
    问题二.排放限值 = 新排放限值
    结果行, 历史组 = [], []
    try:
        for _, 工况 in 工况表.iterrows():
            k = int(工况["工况编号"])
            可行 = bool(可行表.loc[可行表["工况编号"] == k, "5标准是否可行"].iloc[0])
            基准策略, 基准电耗, 基准Y = 基准信息[k]
            if not 可行:
                结果行.append({"典型工况": f"工况{k}", "工况编号": k, "5标准是否可行": 0})
                continue
            最佳候选 = None
            for rep in range(遗传算法重复次数):
                策略, 历史, 摘要 = 问题二.运行遗传算法(
                    工况, 能耗模型, tobit模型, 标准化, 下界, 上界,
                    基准电耗, 基准策略, 基准Y, 模式="节能",
                    seed=随机种子 + 100 * k + rep)
                历史["典型工况"] = f"工况{k}"
                历史["重复编号"] = rep + 1
                历史组.append(历史)
                if 摘要["是否可行"] and (最佳候选 is None or 摘要["预测电耗_kW"] < 最佳候选[2]["预测电耗_kW"]):
                    最佳候选 = (策略.copy(), 历史.copy(), 摘要.copy(), rep + 1)
            if 最佳候选 is None:
                raise RuntimeError(f"工况{k}理论可行，但{遗传算法重复次数}次遗传算法均未找到可行解。")
            策略, _, 摘要, rep = 最佳候选
            row = {
                "典型工况": f"工况{k}", "工况编号": k, "5标准是否可行": 1,
                "采用重复编号": rep, "5标准预测电耗_kW": 摘要["预测电耗_kW"],
                "5标准预测出口浓度_mg_Nm3": 摘要["预测出口浓度_mg_Nm3"],
                "5标准预测除尘指数": 摘要["预测除尘指数"],
                "电压分组违反": 摘要["电压分组违反"],
                "周期分组违反": 摘要["周期分组违反"],
            }
            for i in range(4):
                row[f"5标准U{i+1}_kV"] = float(策略[i])
                row[f"5标准tau{i+1}_s"] = int(round(策略[4+i]))
            结果行.append(row)
    finally:
        问题二.排放限值 = 原限值
    历史表 = pd.concat(历史组, ignore_index=True) if 历史组 else pd.DataFrame()
    return pd.DataFrame(结果行), 历史表


def 合并双边界优化结果(历史优化表, 扩界优化表):
    """扩界包含历史可行域；两次独立搜索后逐工况保留最低电耗可行解。"""
    a = 历史优化表.copy()
    b = 扩界优化表.copy()
    a["采用搜索边界"] = "历史边界"
    b["采用搜索边界"] = "电压上限扩展4%"
    候选 = pd.concat([a, b], ignore_index=True, sort=False)
    候选 = 候选[(候选["5标准是否可行"] == 1) & 候选["5标准预测电耗_kW"].notna()].copy()
    最优 = (候选.sort_values(["工况编号", "5标准预测电耗_kW"])
          .groupby("工况编号", as_index=False).first())
    if len(最优) != 6:
        缺失 = sorted(set(range(1, 7)) - set(最优["工况编号"].astype(int)))
        raise RuntimeError(f"双边界搜索后仍无可行解的工况：{缺失}")
    return 最优


def 合并电耗对比(工况表, 可行表, 优化5表, 优化10表):
    表 = 工况表[["典型工况", "工况编号", "训练样本比例"]].copy()
    表 = 表.merge(可行表[["工况编号", "边界内最低预测浓度_mg_Nm3", "5标准是否可行"]], on="工况编号")
    表 = 表.merge(优化10表[["典型工况", "优化预测电耗_kW", "优化预测出口浓度_mg_Nm3"]], on="典型工况")
    表 = 表.rename(columns={
        "优化预测电耗_kW": "10标准预测电耗_kW",
        "优化预测出口浓度_mg_Nm3": "10标准预测出口浓度_mg_Nm3",
    })
    表 = 表.merge(优化5表, on=["典型工况", "工况编号", "5标准是否可行"], how="left")
    表["电耗增加量_kW"] = 表["5标准预测电耗_kW"] - 表["10标准预测电耗_kW"]
    表["电耗增加率"] = 表["电耗增加量_kW"] / 表["10标准预测电耗_kW"]
    可行 = 表["5标准是否可行"] == 1
    权重 = 表.loc[可行, "训练样本比例"].to_numpy(dtype=float)
    p10 = 表.loc[可行, "10标准预测电耗_kW"].to_numpy(dtype=float)
    p5 = 表.loc[可行, "5标准预测电耗_kW"].to_numpy(dtype=float)
    加权增幅 = float((np.sum(权重 * p5) - np.sum(权重 * p10)) / np.sum(权重 * p10)) if np.any(可行) else np.nan
    覆盖率 = float(表.loc[可行, "训练样本比例"].sum())
    return 表, 加权增幅, 覆盖率


def 构造极值工况参数表(对比表):
    行 = []
    for k, 类型 in [(6, "高入口浓度"), (5, "高负荷重载")]:
        r = 对比表[对比表["工况编号"] == k].iloc[0]
        for i in range(1, 5):
            行.append({
                "工况类型": 类型, "典型工况": f"工况{k}", "控制变量": f"U{i}", "单位": "kV",
                "10标准推荐值": float(r[f"优化10U{i}_kV"]),
                "5标准推荐值": r.get(f"5标准U{i}_kV", np.nan),
            })
        for i in range(1, 5):
            行.append({
                "工况类型": 类型, "典型工况": f"工况{k}", "控制变量": f"tau{i}", "单位": "s",
                "10标准推荐值": float(r[f"优化10tau{i}_s"]),
                "5标准推荐值": r.get(f"5标准tau{i}_s", np.nan),
            })
    表 = pd.DataFrame(行)
    表["调整量"] = 表["5标准推荐值"] - 表["10标准推荐值"]
    return 表


def 绘图(历史可行表, 扩界可行表, 对比表, 极值表):
    设置绘图风格()
    x = np.arange(len(扩界可行表))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    width = 0.34
    b1 = ax.bar(x-width/2, 历史可行表["边界内最低预测浓度_mg_Nm3"], width,
                color=粉色, edgecolor="white", label="历史电压上限")
    b2 = ax.bar(x+width/2, 扩界可行表["边界内最低预测浓度_mg_Nm3"], width,
                color=粉蓝, edgecolor="white", label="电压上限扩展4%")
    ax.axhline(新排放限值, color=深紫, ls="--", lw=1.5, label="5 mg/Nm³限值")
    ax.set_xticks(x, 扩界可行表["典型工况"])
    ax.set_ylabel("最低预测浓度 / (mg/Nm³)")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    ax.legend()
    fig.tight_layout(); 保存图(fig, "问题四_各工况5标准可行性")

    可行 = 对比表[对比表["5标准是否可行"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    xx = np.arange(len(可行)); width = 0.34
    b1 = axes[0].bar(xx-width/2, 可行["10标准预测电耗_kW"], width, color=粉蓝, label="10 mg/Nm³")
    b2 = axes[0].bar(xx+width/2, 可行["5标准预测电耗_kW"], width, color=粉紫, label="5 mg/Nm³")
    axes[0].set_xticks(xx, 可行["典型工况"]); axes[0].set_ylabel("最低预测电耗 / kW")
    axes[0].set_title("（a）可行工况最低电耗")
    axes[0].legend(); axes[0].bar_label(b1, fmt="%.0f", fontsize=8); axes[0].bar_label(b2, fmt="%.0f", fontsize=8)
    b3 = axes[1].bar(xx, 100*可行["电耗增加率"], color=[粉色, 粉紫, 青灰, 粉蓝][:len(可行)], edgecolor="white")
    axes[1].set_xticks(xx, 可行["典型工况"]); axes[1].set_ylabel("电耗增加率 / %")
    axes[1].set_title("（b）排放标准收紧后的电耗增幅")
    axes[1].bar_label(b3, fmt="%.2f%%", padding=2, fontsize=9)
    fig.tight_layout(); 保存图(fig, "问题四_可行工况电耗增幅")

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    图例句柄 = None
    for row_idx, (k, 类型) in enumerate([(6, "高入口浓度工况6"), (5, "高负荷重载工况5")]):
        sub = 极值表[极值表["典型工况"] == f"工况{k}"]
        for col_idx, (kind, ylabel) in enumerate([("U", "二次电压 / kV"), ("tau", "振打周期 / s")]):
            ss = sub[sub["控制变量"].str.startswith(kind)]
            ax = axes[row_idx, col_idx]
            pos = np.arange(4)
            a = ax.bar(pos-width/2, ss["10标准推荐值"], width, color=粉蓝, label="10 mg/Nm³")
            if ss["5标准推荐值"].notna().all():
                b = ax.bar(pos+width/2, ss["5标准推荐值"], width, color=粉紫, label="5 mg/Nm³")
                ax.bar_label(b, fmt="%.1f" if kind == "U" else "%.0f", fontsize=8)
                if 图例句柄 is None:
                    图例句柄 = (a, b)
            ax.bar_label(a, fmt="%.1f" if kind == "U" else "%.0f", fontsize=8)
            ax.set_xticks(pos, [f"电场{i}" for i in range(1, 5)])
            ax.set_ylabel(ylabel); ax.set_title(f"{类型}：{'电压' if kind == 'U' else '振打周期'}")
    if 图例句柄 is not None:
        fig.legend(图例句柄, ["10 mg/Nm³", "5 mg/Nm³"], loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.965]); 保存图(fig, "问题四_极值工况参数对比")


def 写报告(历史可行表, 扩界可行表, 对比表, 加权增幅, 覆盖率, 极值表, 历史上界, 扩展上界):
    历史可行数 = int(历史可行表["5标准是否可行"].sum())
    扩界可行数 = int(扩界可行表["5标准是否可行"].sum())
    历史不可行列表 = "、".join(历史可行表.loc[历史可行表["5标准是否可行"] == 0, "典型工况"].tolist()) or "无"
    结果行 = []
    for _, r in 对比表.iterrows():
        if int(r["5标准是否可行"]) == 1:
            结果行.append(
                f"- {r['典型工况']}：5标准浓度{r['5标准预测出口浓度_mg_Nm3']:.4f} mg/Nm³，"
                f"最低预测电耗{r['5标准预测电耗_kW']:.2f} kW，较10标准增加{r['电耗增加率']:.2%}。")
        else:
            结果行.append(
                f"- {r['典型工况']}：扩展边界内最低预测浓度为{r['边界内最低预测浓度_mg_Nm3']:.4f} mg/Nm³，"
                "未达到5标准，因此不报告5标准电耗及增幅。")
    def 极值描述(k, 类型):
        r = 对比表[对比表["工况编号"] == k].iloc[0]
        if int(r["5标准是否可行"]) == 0:
            return f"{类型}工况{k}在原边界内不可行，不能给出虚假的5标准操作参数。"
        sub = 极值表[极值表["典型工况"] == f"工况{k}"]
        changes = "，".join(f"{x['控制变量']}调整{x['调整量']:+.2f}{x['单位']}" for _, x in sub.iterrows())
        return f"{类型}工况{k}的推荐调整为：{changes}。"
    内容 = f"""# 问题四计算结果

## 1. 方法

完全沿用问题二的GMM、岭回归能耗模型、Tobit基准除尘指数、Deutsch–Anderson相对外推和遗传算法参数，只将出口浓度约束由10 mg/Nm³改为5 mg/Nm³。烟气温度、入口粉尘浓度和烟气流量固定为相应典型工况中心，不作为决策变量。由于阈值减半，所需除尘指数统一增加 `ln(2)={math.log(2):.6f}`。

## 2. 历史边界与额定许可情景

历史电压上限为 `{np.round(历史上界[:4], 3).tolist()}` kV。历史边界内共有 **{历史可行数}/6** 类工况满足5标准，不可行工况为：**{历史不可行列表}**。由于历史极值不等同于设备额定上限，本文在“设备额定电压允许”的前提下设置情景分析，仅将四级电压上限统一扩展 **{电压上限扩展比例:.0%}** 至 `{np.round(扩展上界[:4], 3).tolist()}` kV；电压下限、振打周期边界和全部环境工况保持不变。扩界后共有 **{扩界可行数}/6** 类工况可行。详见 `5标准历史边界可行性检查.csv`、`5标准扩界可行性检查.csv` 和[可行性对比图](../figures/问题四计算结果/问题四_各工况5标准可行性.pdf)。

## 3. 电耗变化

{chr(10).join(结果行)}

扩界后可行工况覆盖训练样本比例为 **{覆盖率:.2%}**，按六类工况训练样本比例加权的总体电耗增加率为 **{加权增幅:.2%}**。详见 `10与5标准电耗对比.csv` 和[电耗增幅图](../figures/问题四计算结果/问题四_可行工况电耗增幅.pdf)。

## 4. 高浓度与高负荷工况建议

- {极值描述(6, '高入口浓度')}
- {极值描述(5, '高负荷重载')}

参数详见 `极值工况参数对比.csv` 和[参数对比图](../figures/问题四计算结果/问题四_极值工况参数对比.pdf)。控制建议只依据本次优化结果，不预设固定的前级调压或后级调振顺序。振打周期调整还需结合问题一的瞬时峰值风险：只有当缩短周期能够提高除尘指数且不会明显放大瞬时峰值时才实施。

## 5. 结果边界

4%的电压扩展属于额定许可情景，不是由历史数据直接识别出的安全边界。实际应用前必须依据设备铭牌、绝缘水平和运行规程核验；若额定边界不允许，则工况3、4仍应判定为不可行。问题一至问题三继续使用原历史边界及既有结果，不因本情景分析而修改。

## 6. 运行环境

- Python：{platform.python_version()}
- 运行命令：`{Path(sys.executable)} -B code/问题四.py`
- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}
"""
    报告路径.write_text(内容, encoding="utf-8")


def main():
    开始 = datetime.now()
    结果目录.mkdir(parents=True, exist_ok=True)
    图片目录.mkdir(parents=True, exist_ok=True)
    报告路径.parent.mkdir(parents=True, exist_ok=True)
    问题二 = 加载问题二模块()
    数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 下界, 上界, 边界表 = 重建问题二模型(问题二)
    优化10表 = pd.read_csv(问题二结果目录 / "优化结果_Deutsch相对外推.csv")
    # 为后续极值工况表统一列名。
    优化10重命名 = 优化10表.copy()
    for i in range(1, 5):
        优化10重命名[f"优化10U{i}_kV"] = 优化10重命名[f"优化U{i}_kV"]
        优化10重命名[f"优化10tau{i}_s"] = 优化10重命名[f"优化tau{i}_s"]

    历史可行表, _ = 可行性检查(
        问题二, 数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 下界, 上界)
    扩展下界 = 下界.copy()
    扩展上界 = 上界.copy()
    扩展上界[:4] = 上界[:4] * (1.0 + 电压上限扩展比例)
    扩界可行表, 基准信息 = 可行性检查(
        问题二, 数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 扩展下界, 扩展上界)
    历史优化5表, 历史搜索历史 = 优化5标准(
        问题二, 工况表, 历史可行表, 能耗模型, tobit模型, 标准化, 下界, 上界, 基准信息)
    扩界优化5表, 扩界搜索历史 = 优化5标准(
        问题二, 工况表, 扩界可行表, 能耗模型, tobit模型, 标准化, 扩展下界, 扩展上界, 基准信息)
    优化5表 = 合并双边界优化结果(历史优化5表, 扩界优化5表)
    历史搜索历史["搜索边界"] = "历史边界"
    扩界搜索历史["搜索边界"] = "电压上限扩展4%"
    历史表 = pd.concat([历史搜索历史, 扩界搜索历史], ignore_index=True)
    对比表, 加权增幅, 覆盖率 = 合并电耗对比(工况表, 扩界可行表, 优化5表, 优化10表)
    # 加入10标准控制变量，供极值工况对比。
    对比表 = 对比表.merge(
        优化10重命名[["典型工况"] + [f"优化10U{i}_kV" for i in range(1,5)] + [f"优化10tau{i}_s" for i in range(1,5)]],
        on="典型工况", how="left")
    极值表 = 构造极值工况参数表(对比表)

    历史可行表.to_csv(结果目录 / "5标准历史边界可行性检查.csv", index=False, encoding="utf-8-sig")
    扩界可行表.to_csv(结果目录 / "5标准扩界可行性检查.csv", index=False, encoding="utf-8-sig")
    扩界可行表.to_csv(结果目录 / "5标准可行性检查.csv", index=False, encoding="utf-8-sig")
    对比表.to_csv(结果目录 / "10与5标准电耗对比.csv", index=False, encoding="utf-8-sig")
    极值表.to_csv(结果目录 / "极值工况参数对比.csv", index=False, encoding="utf-8-sig")
    历史表.to_csv(结果目录 / "遗传算法收敛历史.csv", index=False, encoding="utf-8-sig")
    边界表.to_csv(结果目录 / "沿用的决策变量边界.csv", index=False, encoding="utf-8-sig")
    扩展边界表 = 边界表.copy()
    扩展边界表["历史上限"] = 扩展边界表["上限"]
    扩展边界表.loc[扩展边界表["变量"].str.startswith("U"), "上限"] *= 1.0 + 电压上限扩展比例
    扩展边界表["边界说明"] = np.where(
        扩展边界表["变量"].str.startswith("U"), "问题四额定许可情景：历史上限扩展4%", "沿用问题二历史边界")
    扩展边界表.to_csv(结果目录 / "问题四扩展决策变量边界.csv", index=False, encoding="utf-8-sig")
    参数 = {
        "原排放限值_mg_Nm3": 原排放限值, "新排放限值_mg_Nm3": 新排放限值,
        "除尘指数增量": math.log(2), "遗传算法重复次数": 遗传算法重复次数,
        "每次遗传算法参数": {"种群规模": 100, "最大代数": 300, "交叉概率": 0.8, "变异概率": 0.1, "精英数": 2},
        "电压上限扩展比例": 电压上限扩展比例,
        "历史边界可行工况数": int(历史可行表["5标准是否可行"].sum()),
        "扩界后可行工况数": int(扩界可行表["5标准是否可行"].sum()),
        "可行工况样本覆盖率": 覆盖率,
        "可行集合加权电耗增加率": 加权增幅,
        "历史决策边界": {"下界": 下界.tolist(), "上界": 上界.tolist()},
        "问题四扩展决策边界": {"下界": 扩展下界.tolist(), "上界": 扩展上界.tolist()},
    }
    (结果目录 / "问题四参数与汇总.json").write_text(json.dumps(参数, ensure_ascii=False, indent=2), encoding="utf-8")
    绘图(历史可行表, 扩界可行表, 对比表, 极值表)
    写报告(历史可行表, 扩界可行表, 对比表, 加权增幅, 覆盖率, 极值表, 上界, 扩展上界)
    日志 = [
        f"开始时间：{开始:%Y-%m-%d %H:%M:%S}", f"结束时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Python：{sys.executable} ({platform.python_version()})",
        f"历史边界5标准可行工况数：{int(历史可行表['5标准是否可行'].sum())}/6",
        f"扩界后5标准可行工况数：{int(扩界可行表['5标准是否可行'].sum())}/6",
        f"可行工况样本覆盖率：{覆盖率:.4%}", f"可行集合加权电耗增加率：{加权增幅:.4%}",
    ]
    (结果目录 / "运行日志.txt").write_text("\n".join(日志), encoding="utf-8")
    print("\n".join(日志[3:]))


if __name__ == "__main__":
    main()
