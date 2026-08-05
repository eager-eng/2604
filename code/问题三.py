# -*- coding: utf-8 -*-
"""问题三：典型工况最优策略对比与局部边际效益分析。"""

from __future__ import annotations

import importlib.util
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
结果目录 = 项目目录 / "code" / "outputs" / "问题三计算结果"
图片目录 = 项目目录 / "figures" / "问题三计算结果"
报告路径 = 项目目录 / "reports" / "问题三计算结果.md"

对比工况编号 = [4, 5]
工况名称 = {4: "工况A（低温轻载）", 5: "工况B（中温重载）"}
排放限值 = 10.0
电压扰动步长 = 1.0
周期扰动步长 = 5.0

粉蓝 = "#8FB7D4"
粉紫 = "#B8A2CF"
粉色 = "#D7A8B8"
深紫 = "#75658F"
浅灰 = "#E8EAF0"
灰色 = "#6F7480"


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
        "axes.labelsize": 10,
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
    下界, 上界, _ = 问题二.决策边界(数据)
    return 数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 下界, 上界


def 获取工况基准(问题二, 数据, 标签, 工况: pd.Series, 能耗模型, tobit模型, 标准化):
    k = int(工况["工况编号"])
    训练 = 数据[数据["数据集"] == "训练集"].copy()
    训练标签 = 标签[数据["数据集"].eq("训练集").to_numpy()] + 1
    子集 = 训练[训练标签 == k]
    基准策略 = np.r_[
        [float(子集[f"电场{i}电压_kV"].median()) for i in range(1, 5)],
        [int(round(float(子集[f"电场{i}振打周期_s"].median()))) for i in range(1, 5)],
    ]
    Xp, Xt, _ = 问题二.构造候选特征(基准策略.reshape(1, -1), 工况, 标准化)
    基准电耗 = float(能耗模型.predict(Xp)[0])
    基准除尘指数 = float((Xt @ tobit模型["beta"])[0])
    return 基准策略, 基准电耗, 基准除尘指数


def 评价策略(问题二, 策略, 工况, 能耗模型, tobit模型, 标准化, 基准策略, 基准除尘指数, 下界, 上界):
    pop = np.asarray(策略, dtype=float).reshape(1, -1)
    Xp, _, _ = 问题二.构造候选特征(pop, 工况, 标准化)
    电耗 = float(能耗模型.predict(Xp)[0])
    除尘指数, _, _ = 问题二.Deutsch相对除尘指数(
        pop, 基准策略, 基准除尘指数, tobit模型, 标准化)
    浓度 = float(1000.0 * float(工况["入口浓度中心_g_Nm3"]) * np.exp(-除尘指数[0]))
    vu, vt = 问题二.约束量(pop, 下界, 上界)
    return 电耗, 浓度, float(vu[0]), float(vt[0])


def 构造对比表(优化表: pd.DataFrame, 工况表: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    选中 = 工况表[工况表["工况编号"].isin(对比工况编号)].copy()
    选中["论文工况"] = 选中["工况编号"].map(工况名称)
    选中["选择依据"] = 选中["工况编号"].map({4: "入口粉尘负荷最低", 5: "入口粉尘负荷最高"})
    选中 = 选中[[
        "论文工况", "典型工况", "工况编号", "选择依据", "训练样本数", "烟气温度中心_℃",
        "入口浓度中心_g_Nm3", "烟气流量中心_Nm3_h", "入口负荷中心_kg_h",
    ]]

    行 = []
    for k in 对比工况编号:
        r = 优化表[优化表["典型工况"] == f"工况{k}"].iloc[0]
        row = {"论文工况": 工况名称[k], "原始工况": f"工况{k}"}
        for i in range(1, 5):
            row[f"U{i}_kV"] = float(r[f"优化U{i}_kV"])
            row[f"tau{i}_s"] = int(r[f"优化tau{i}_s"])
        row["平均电压_kV"] = float(np.mean([row[f"U{i}_kV"] for i in range(1, 5)]))
        row["平均振打频率_次_分钟"] = float(np.mean([60.0 / row[f"tau{i}_s"] for i in range(1, 5)]))
        row["预测出口浓度_mg_Nm3"] = float(r["优化预测出口浓度_mg_Nm3"])
        row["预测总电耗_kW"] = float(r["优化预测电耗_kW"])
        行.append(row)
    操作表 = pd.DataFrame(行)
    return 选中, 操作表


def 计算局部扰动(问题二, 数据, 标签, 工况表, 操作表, 能耗模型, tobit模型, 标准化, 下界, 上界):
    行 = []
    变量 = [f"U{i}" for i in range(1, 5)] + [f"tau{i}" for i in range(1, 5)]
    for k in 对比工况编号:
        工况 = 工况表[工况表["工况编号"] == k].iloc[0]
        基准策略, _, 基准除尘指数 = 获取工况基准(
            问题二, 数据, 标签, 工况, 能耗模型, tobit模型, 标准化)
        op = 操作表[操作表["原始工况"] == f"工况{k}"].iloc[0]
        最优策略 = np.r_[
            [float(op[f"U{i}_kV"]) for i in range(1, 5)],
            [int(op[f"tau{i}_s"]) for i in range(1, 5)],
        ]
        P0, C0, _, _ = 评价策略(
            问题二, 最优策略, 工况, 能耗模型, tobit模型, 标准化,
            基准策略, 基准除尘指数, 下界, 上界)
        for j, 名称 in enumerate(变量):
            步长 = 电压扰动步长 if j < 4 else 周期扰动步长
            for 方向 in (-1, 1):
                候选 = 最优策略.copy()
                候选[j] += 方向 * 步长
                if j >= 4:
                    候选[j] = round(候选[j])
                边界满足 = bool(下界[j] - 1e-10 <= 候选[j] <= 上界[j] + 1e-10)
                if 边界满足:
                    P1, C1, vu, vt = 评价策略(
                        问题二, 候选, 工况, 能耗模型, tobit模型, 标准化,
                        基准策略, 基准除尘指数, 下界, 上界)
                    分组满足 = bool(vu <= 1e-10 and vt <= 1e-10)
                    排放满足 = bool(C1 <= 排放限值 + 1e-10)
                    dP, dC = P1 - P0, C1 - C0
                    减排效益 = -dC / dP if dP > 0 and dC < 0 else np.nan
                    节能效益 = -dP / dC if dP < 0 and dC > 0 and 排放满足 else np.nan
                    双赢 = bool(dP < 0 and dC <= 0 and 分组满足)
                    达标节能量 = -dP if dP < 0 and 排放满足 and 分组满足 else np.nan
                    if 双赢 and 排放满足:
                        节能类型 = "节能减排双赢"
                    elif np.isfinite(节能效益) and 分组满足:
                        节能类型 = "有排放代价的达标节能"
                    else:
                        节能类型 = "无达标节能方向"
                else:
                    P1 = C1 = dP = dC = vu = vt = np.nan
                    分组满足 = 排放满足 = 双赢 = False
                    减排效益 = 节能效益 = np.nan
                    达标节能量 = np.nan
                    节能类型 = "不可用"
                行.append({
                    "论文工况": 工况名称[k], "原始工况": f"工况{k}", "变量": 名称,
                    "变量类型": "电压" if j < 4 else "振打周期",
                    "扰动方向": "+" if 方向 > 0 else "-", "扰动步长": 步长,
                    "原值": 最优策略[j], "扰动后值": 候选[j],
                    "边界满足": int(边界满足), "分组约束满足": int(分组满足),
                    "排放约束满足": int(排放满足), "基准预测电耗_kW": P0,
                    "扰动后预测电耗_kW": P1, "电耗变化_kW": dP,
                    "基准预测浓度_mg_Nm3": C0, "扰动后预测浓度_mg_Nm3": C1,
                    "浓度变化_mg_Nm3": dC,
                    "减排边际效益_mg_Nm3每kW": 减排效益,
                    "节能边际效益_kW每mg_Nm3": 节能效益,
                    "达标局部节能量_kW": 达标节能量,
                    "节能方向类型": 节能类型,
                    "是否节能减排双赢": int(双赢),
                })
    return pd.DataFrame(行)


def 汇总边际效益(明细: pd.DataFrame) -> pd.DataFrame:
    行 = []
    for (工况, 变量), 子集 in 明细.groupby(["论文工况", "变量"], sort=False):
        可用 = 子集[(子集["边界满足"] == 1) & (子集["分组约束满足"] == 1)]
        减排 = 可用.dropna(subset=["减排边际效益_mg_Nm3每kW"])
        节能 = 可用.dropna(subset=["节能边际效益_kW每mg_Nm3"])
        达标节能 = 可用.dropna(subset=["达标局部节能量_kW"])
        减排最佳 = 减排.loc[减排["减排边际效益_mg_Nm3每kW"].idxmax()] if len(减排) else None
        节能最佳 = 节能.loc[节能["节能边际效益_kW每mg_Nm3"].idxmax()] if len(节能) else None
        达标节能最佳 = 达标节能.loc[达标节能["达标局部节能量_kW"].idxmax()] if len(达标节能) else None
        行.append({
            "论文工况": 工况, "变量": 变量,
            "最佳减排方向": "无" if 减排最佳 is None else str(减排最佳["扰动方向"]),
            "减排边际效益_mg_Nm3每kW": np.nan if 减排最佳 is None else float(减排最佳["减排边际效益_mg_Nm3每kW"]),
            "最佳节能方向": "无" if 节能最佳 is None else str(节能最佳["扰动方向"]),
            "节能边际效益_kW每mg_Nm3": np.nan if 节能最佳 is None else float(节能最佳["节能边际效益_kW每mg_Nm3"]),
            "存在达标节能方向": int(节能最佳 is not None),
            "最佳达标节能方向": "无" if 达标节能最佳 is None else str(达标节能最佳["扰动方向"]),
            "达标局部节能量_kW": np.nan if 达标节能最佳 is None else float(达标节能最佳["达标局部节能量_kW"]),
            "达标节能类型": "无" if 达标节能最佳 is None else str(达标节能最佳["节能方向类型"]),
        })
    汇总 = pd.DataFrame(行)
    for 指标, 排名列 in [
        ("减排边际效益_mg_Nm3每kW", "减排优先级"),
        ("节能边际效益_kW每mg_Nm3", "节能优先级"),
    ]:
        汇总[排名列] = 汇总.groupby("论文工况")[指标].rank(method="min", ascending=False)
    return 汇总


def 绘制最优参数图(操作表: pd.DataFrame) -> None:
    设置绘图风格()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    x = np.arange(4)
    width = 0.34
    for idx, (_, row) in enumerate(操作表.iterrows()):
        offset = (idx - 0.5) * width
        color = 粉蓝 if idx == 0 else 粉紫
        label = str(row["论文工况"])
        U = [row[f"U{i}_kV"] for i in range(1, 5)]
        tau = [row[f"tau{i}_s"] for i in range(1, 5)]
        bars_u = axes[0].bar(x + offset, U, width, color=color, edgecolor="white", label=label)
        bars_t = axes[1].bar(x + offset, tau, width, color=color, edgecolor="white", label=label)
        axes[0].bar_label(bars_u, fmt="%.1f", padding=2, fontsize=8)
        axes[1].bar_label(bars_t, fmt="%.0f", padding=2, fontsize=8)
    axes[0].set_xticks(x, [f"电场{i}" for i in range(1, 5)])
    axes[1].set_xticks(x, [f"电场{i}" for i in range(1, 5)])
    axes[0].set_ylabel("最优二次电压 / kV")
    axes[1].set_ylabel("最优振打周期 / s")
    axes[0].set_title("（a）四级电场最优电压")
    axes[1].set_title("（b）四级电场最优振打周期")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[0].set_ylim(0, max(75, axes[0].get_ylim()[1] * 1.08))
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.10)
    fig.tight_layout()
    保存图(fig, "问题三_最优电压与振打周期对比")


def 绘制边际效益图(汇总: pd.DataFrame) -> None:
    设置绘图风格()
    variables = [f"U{i}" for i in range(1, 5)] + [f"tau{i}" for i in range(1, 5)]
    conditions = [工况名称[k] for k in 对比工况编号]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    specs = [
        ("减排边际效益_mg_Nm3每kW", "（a）减排阶段边际效益", "mg/Nm³/kW", "PuBu"),
        ("达标局部节能量_kW", "（b）达标局部节能量", "kW", "BuPu"),
    ]
    for ax, (col, title, cbar_label, cmap) in zip(axes, specs):
        matrix = np.full((len(variables), len(conditions)), np.nan)
        for i, var in enumerate(variables):
            for j, cond in enumerate(conditions):
                s = 汇总[(汇总["变量"] == var) & (汇总["论文工况"] == cond)]
                if len(s):
                    matrix[i, j] = float(s.iloc[0][col])
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(conditions)), ["工况A\n低温轻载", "工况B\n中温重载"])
        ax.set_yticks(np.arange(len(variables)), [r"$U_1$", r"$U_2$", r"$U_3$", r"$U_4$", r"$\tau_1$", r"$\tau_2$", r"$\tau_3$", r"$\tau_4$"])
        ax.set_title(title)
        ax.grid(False)
        finite = matrix[np.isfinite(matrix)]
        threshold = np.nanmedian(finite) if len(finite) else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isfinite(val):
                    label = "—"
                elif col == "达标局部节能量_kW":
                    s = 汇总[(汇总["变量"] == variables[i]) & (汇总["论文工况"] == conditions[j])]
                    win = len(s) and s.iloc[0]["达标节能类型"] == "节能减排双赢"
                    label = f"{val:.3g}" + ("\n双赢" if win else "")
                else:
                    label = f"{val:.3g}"
                color = "white" if np.isfinite(val) and val > threshold else "#343742"
                ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color=color)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    保存图(fig, "问题三_局部扰动边际效益热力图")


def 写报告(工况对比, 操作表, 明细, 汇总):
    减排排序 = 汇总.dropna(subset=["减排边际效益_mg_Nm3每kW"]).sort_values(
        ["论文工况", "减排边际效益_mg_Nm3每kW"], ascending=[True, False])
    节能排序 = 汇总.dropna(subset=["达标局部节能量_kW"]).sort_values(
        ["论文工况", "达标局部节能量_kW"], ascending=[True, False])
    lines = []
    for cond in 操作表["论文工况"]:
        a = 操作表[操作表["论文工况"] == cond].iloc[0]
        red = 减排排序[减排排序["论文工况"] == cond]
        save = 节能排序[节能排序["论文工况"] == cond]
        red_text = "无满足定义的减排方向" if red.empty else f"{red.iloc[0]['变量']}（方向{red.iloc[0]['最佳减排方向']}，{red.iloc[0]['减排边际效益_mg_Nm3每kW']:.6g} mg/Nm³/kW）"
        save_text = "当前最优点附近没有保持达标的节能方向" if save.empty else f"{save.iloc[0]['变量']}（方向{save.iloc[0]['最佳达标节能方向']}，局部节电{save.iloc[0]['达标局部节能量_kW']:.6g} kW，{save.iloc[0]['达标节能类型']}）"
        lines.append(f"- {cond}：浓度{a['预测出口浓度_mg_Nm3']:.6f} mg/Nm³，电耗{a['预测总电耗_kW']:.3f} kW；减排首选为{red_text}；达标节能首选为{save_text}。")
    边界可用 = int(((明细["边界满足"] == 1) & (明细["分组约束满足"] == 1)).sum())
    报告 = f"""# 问题三计算结果

## 1. 分析方法

沿用问题二的GMM、岭回归能耗模型、左删失Tobit基准除尘指数和Deutsch–Anderson相对外推模型，不建立新的独立预测模型，也不重新运行遗传算法。选取入口粉尘负荷最低的工况4与最高的工况5，在两类最优解附近对8个控制变量分别实施正、负方向局部扰动。

## 2. 对比工况与最优策略

- 工况A（原始工况4）：低温轻载，入口负荷{工况对比.iloc[0]['入口负荷中心_kg_h']:.2f} kg/h；
- 工况B（原始工况5）：中温重载，入口负荷{工况对比.iloc[1]['入口负荷中心_kg_h']:.2f} kg/h。

精确操作参数见 `最优操作参数对比.csv`，图表见[最优电压与振打周期对比](../figures/问题三计算结果/问题三_最优电压与振打周期对比.pdf)。

## 3. 局部扰动与边际效益

两类工况各包含8组双向扰动，共16组、32个候选扰动点；若扰动越过变量边界则保留记录并标为不可用。共有{边界可用}/32个候选点同时满足变量边界与前后级分组约束。减排边际效益仅在电耗增加且浓度降低时定义，节能边际效益仅在电耗降低、浓度上升且扰动后仍满足10 mg/Nm³时定义。

{chr(10).join(lines)}

完整扰动数值见 `局部双向扰动明细.csv` 和 `边际效益汇总.csv`，图表见[局部扰动边际效益热力图](../figures/问题三计算结果/问题三_局部扰动边际效益热力图.pdf)。右图采用“保持达标时的局部节电量”，以便同时容纳浓度上升的常规节能方向和浓度、电耗同步下降的双赢方向；破折号表示没有满足约束的对应方向。

## 4. 结论边界

本结果反映问题二响应模型和给定扰动步长下的局部关系。优先级应按“当前工况—控制阶段—局部边际效益”判断，不能由两组最优参数的静态差异直接推断反电晕、电晕闭塞或固定的前后级控制顺序。若某节能扰动导致浓度超过10 mg/Nm³，则不能直接采用；若出现节能减排双赢方向，可优先作小幅局部调整，但其幅度仅代表当前步长下的局部改进，不等同于全局规律。

## 5. 运行环境

- Python：{platform.python_version()}
- 运行命令：`{Path(sys.executable)} -B code/问题三.py`
- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}
"""
    报告路径.write_text(报告, encoding="utf-8")


def main():
    开始 = datetime.now()
    结果目录.mkdir(parents=True, exist_ok=True)
    图片目录.mkdir(parents=True, exist_ok=True)
    报告路径.parent.mkdir(parents=True, exist_ok=True)
    问题二 = 加载问题二模块()
    数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 下界, 上界 = 重建问题二模型(问题二)
    优化表 = pd.read_csv(问题二结果目录 / "优化结果_Deutsch相对外推.csv")
    工况对比, 操作表 = 构造对比表(优化表, 工况表)
    明细 = 计算局部扰动(
        问题二, 数据, 标签, 工况表, 操作表, 能耗模型, tobit模型,
        标准化, 下界, 上界)
    汇总 = 汇总边际效益(明细)

    工况对比.to_csv(结果目录 / "对比工况特征.csv", index=False, encoding="utf-8-sig")
    操作表.to_csv(结果目录 / "最优操作参数对比.csv", index=False, encoding="utf-8-sig")
    明细.to_csv(结果目录 / "局部双向扰动明细.csv", index=False, encoding="utf-8-sig")
    汇总.to_csv(结果目录 / "边际效益汇总.csv", index=False, encoding="utf-8-sig")
    绘制最优参数图(操作表)
    绘制边际效益图(汇总)
    写报告(工况对比, 操作表, 明细, 汇总)

    日志 = [
        f"开始时间：{开始:%Y-%m-%d %H:%M:%S}",
        f"结束时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Python：{sys.executable} ({platform.python_version()})",
        f"对比工况：工况4、工况5",
        f"双向扰动组数：16，候选扰动点数：{len(明细)}",
        f"排放约束内的节能候选数：{int(明细['达标局部节能量_kW'].notna().sum())}",
    ]
    (结果目录 / "运行日志.txt").write_text("\n".join(日志), encoding="utf-8")
    print("\n".join(日志[3:]))


if __name__ == "__main__":
    main()
