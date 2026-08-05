# -*- coding: utf-8 -*-
"""问题一：基于右删失 Tobit 回归的潜在出口浓度重构与峰值分析。"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


项目目录 = Path(__file__).resolve().parents[1]
数据路径 = 项目目录 / "code" / "outputs" / "数据预处理结果" / "建模就绪数据.csv"
结果目录 = 项目目录 / "code" / "outputs" / "问题一计算结果"
图片目录 = 项目目录 / "figures" / "问题一计算结果"
报告路径 = 项目目录 / "reports" / "问题一计算结果.md"

删失上限 = 50.0
候选滞后 = [0, 1, 3, 5]
基础变量 = [
    "入口温度_℃", "入口粉尘浓度_g_Nm3", "烟气流量_Nm3_h", "入口粉尘负荷_kg_h",
    "电场1电压_kV", "电场2电压_kV", "电场3电压_kV", "电场4电压_kV",
]

颜色 = {
    "粉": "#D9A5B3", "蓝": "#83A9C9", "紫": "#A99AC7", "深紫": "#75658F",
    "浅蓝": "#B9D7E8", "浅粉": "#E9C8D0", "灰": "#6F7480", "浅灰": "#D9DCE3",
}


def 设置绘图风格() -> None:
    字体候选 = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams.update({
        "font.sans-serif": 字体候选,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#AEB3BC",
        "axes.grid": True,
        "grid.color": "#D9DCE3",
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def 构建设计矩阵(数据: pd.DataFrame, 滞后: int) -> tuple[np.ndarray, list[str]]:
    """使用预处理阶段的训练集标准化结果，构造指定统一滞后的解释变量。"""
    if 滞后 == 0:
        基础列 = [f"{变量}_标准化" for 变量 in 基础变量]
    else:
        基础列 = [f"{变量}_滞后{滞后}分钟_标准化" for 变量 in 基础变量]

    频率列 = [f"电场{i}_5分钟平均振打频率_次_分钟_标准化" for i in range(1, 5)]
    矩阵部分 = [数据[基础列].to_numpy(dtype=float), 数据[频率列].to_numpy(dtype=float)]

    # 同一电场的滞后电压与当前5分钟滚动频率相乘，描述两者的协同影响。
    交互矩阵 = np.column_stack([
        数据[基础列[4 + i]].to_numpy(dtype=float) * 数据[频率列[i]].to_numpy(dtype=float)
        for i in range(4)
    ])
    矩阵部分.append(交互矩阵)
    X = np.column_stack(矩阵部分)

    名称 = ["温度", "入口浓度", "烟气流量", "入口负荷"]
    名称 += [f"电场{i}电压" for i in range(1, 5)]
    名称 += [f"电场{i}振打频率" for i in range(1, 5)]
    名称 += [f"电场{i}电压×振打频率" for i in range(1, 5)]
    return np.column_stack([np.ones(len(X)), X]), ["常数项"] + 名称


def 负对数似然与梯度(参数: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    beta, eta = 参数[:-1], float(参数[-1])
    sigma = math.exp(eta)
    mu = X @ beta
    未删失 = y < 删失上限 - 1e-10
    z = (y[未删失] - mu[未删失]) / sigma
    a = (删失上限 - mu[~未删失]) / sigma

    nll = np.sum(eta - norm.logpdf(z)) - np.sum(norm.logsf(a))
    对mu梯度 = np.empty_like(y, dtype=float)
    对mu梯度[未删失] = (mu[未删失] - y[未删失]) / sigma**2
    if np.any(~未删失):
        mills = np.exp(np.clip(norm.logpdf(a) - norm.logsf(a), -40, 40))
        对mu梯度[~未删失] = -mills / sigma
        对eta梯度_删失 = np.sum(-mills * a)
    else:
        对eta梯度_删失 = 0.0
    对eta梯度 = np.sum(1.0 - z**2) + 对eta梯度_删失
    梯度 = np.r_[X.T @ 对mu梯度, 对eta梯度]
    return float(nll), 梯度


def 拟合Tobit(X: np.ndarray, y: np.ndarray, 初值: np.ndarray | None = None) -> dict:
    if 初值 is None:
        未删失 = y < 删失上限 - 1e-10
        beta0 = np.linalg.lstsq(X[未删失], y[未删失], rcond=None)[0]
        残差 = y[未删失] - X[未删失] @ beta0
        sigma0 = max(float(np.std(残差, ddof=min(1, len(残差) - 1))), 0.5)
        初值 = np.r_[beta0, math.log(sigma0)]

    def 目标(p: np.ndarray) -> float:
        return 负对数似然与梯度(p, X, y)[0]

    def 梯度(p: np.ndarray) -> np.ndarray:
        return 负对数似然与梯度(p, X, y)[1]

    边界 = [(None, None)] * (X.shape[1]) + [(math.log(0.05), math.log(30.0))]
    结果 = minimize(目标, 初值, jac=梯度, method="L-BFGS-B", bounds=边界,
                  options={"maxiter": 2500, "ftol": 1e-11, "gtol": 1e-6, "maxls": 60})
    return {
        "参数": np.asarray(结果.x, dtype=float),
        "beta": np.asarray(结果.x[:-1], dtype=float),
        "sigma": float(math.exp(结果.x[-1])),
        "负对数似然": float(结果.fun),
        "收敛": bool(结果.success),
        "迭代次数": int(结果.nit),
        "消息": str(结果.message),
    }


def 样本负对数似然(beta: np.ndarray, sigma: float, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu = X @ beta
    未删失 = y < 删失上限 - 1e-10
    结果 = np.empty(len(y), dtype=float)
    z = (y[未删失] - mu[未删失]) / sigma
    结果[未删失] = math.log(sigma) - norm.logpdf(z)
    a = (删失上限 - mu[~未删失]) / sigma
    结果[~未删失] = -norm.logsf(a)
    return 结果


def 数值Hessian(参数: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """对解析梯度作中心差分，得到观测信息矩阵。"""
    p = len(参数)
    H = np.empty((p, p), dtype=float)
    for j in range(p):
        步长 = 1e-4 * max(1.0, abs(float(参数[j])))
        上 = 参数.copy(); 上[j] += 步长
        下 = 参数.copy(); 下[j] -= 步长
        H[:, j] = (负对数似然与梯度(上, X, y)[1] - 负对数似然与梯度(下, X, y)[1]) / (2 * 步长)
    return (H + H.T) / 2


def 保存图(fig: plt.Figure, 文件名: str) -> None:
    fig.savefig(图片目录 / f"{文件名}.png", dpi=400, facecolor="white")
    fig.savefig(图片目录 / f"{文件名}.pdf", facecolor="white")
    plt.close(fig)


def 分析峰值(数据: pd.DataFrame, mu: np.ndarray, sigma: float) -> tuple[pd.DataFrame, float]:
    结果 = 数据.copy()
    结果["线性预测浓度_mg_Nm3"] = mu
    a = (删失上限 - mu) / sigma
    mills = np.exp(np.clip(norm.logpdf(a) - norm.logsf(a), -40, 40))
    结果["删失概率"] = norm.sf(a)
    结果["潜在浓度估计_mg_Nm3"] = np.where(
        结果["右删失标记"].to_numpy() == 1,
        mu + sigma * mills,
        结果["出口粉尘浓度_mg_Nm3"].to_numpy(),
    )
    结果["局部背景浓度_mg_Nm3"] = 结果["潜在浓度估计_mg_Nm3"].shift(1).rolling(10, min_periods=10).median()
    结果["相对峰高_mg_Nm3"] = 结果["潜在浓度估计_mg_Nm3"] - 结果["局部背景浓度_mg_Nm3"]
    训练可用 = (
        (结果["数据集"] == "训练集") &
        (结果["出口浓度原始缺失标记"] == 0) &
        结果["相对峰高_mg_Nm3"].notna()
    )
    阈值 = float(结果.loc[训练可用, "相对峰高_mg_Nm3"].quantile(0.95))
    结果["峰值标记"] = (
        (结果["相对峰高_mg_Nm3"] >= 阈值) &
        (结果["出口浓度原始缺失标记"] == 0)
    ).astype(int)
    return 结果, 阈值


def 四电场统计(峰值数据: pd.DataFrame) -> pd.DataFrame:
    行 = []
    有效 = 峰值数据[峰值数据["出口浓度原始缺失标记"] == 0].copy()
    组名 = ["低", "中低", "中高", "高"]
    for i in range(1, 5):
        频率列 = f"电场{i}_5分钟平均振打频率_次_分钟"
        分组列 = pd.qcut(有效[频率列], q=4, labels=组名, duplicates="drop")
        for 组 in 组名:
            子集 = 有效[分组列 == 组]
            峰 = 子集[子集["峰值标记"] == 1]
            中位频率 = float(子集[频率列].median())
            行.append({
                "电场": f"电场{i}", "振打频率组": 组, "样本数": int(len(子集)),
                "中位振打频率_次_分钟": 中位频率,
                "等效振打周期_s": float(60 / 中位频率),
                "峰值数量": int(len(峰)),
                "峰值发生概率": float(子集["峰值标记"].mean()),
                "平均相对峰高_mg_Nm3": float(峰["相对峰高_mg_Nm3"].mean()) if len(峰) else np.nan,
                "最大潜在峰值_mg_Nm3": float(峰["潜在浓度估计_mg_Nm3"].max()) if len(峰) else np.nan,
            })
    return pd.DataFrame(行)


def 绘制结果(滞后表: pd.DataFrame, 参数表: pd.DataFrame, 峰值数据: pd.DataFrame,
           电场统计: pd.DataFrame, 测试指标: dict) -> None:
    设置绘图风格()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(滞后表["滞后分钟"], 滞后表["验证集平均负对数似然"], color=颜色["深紫"],
            marker="o", lw=2, ms=6)
    最优 = 滞后表.loc[滞后表["是否最优"] == 1].iloc[0]
    ax.scatter([最优["滞后分钟"]], [最优["验证集平均负对数似然"]], s=110,
               color=颜色["粉"], edgecolor="white", zorder=5, label="最优滞后")
    ax.set(xlabel="统一滞后 / min", ylabel="验证集平均负对数似然", title="（a）候选滞后比较")
    ax.set_xticks(候选滞后); ax.legend()

    测试 = 峰值数据[峰值数据["数据集"] == "测试集"]
    未删失 = 测试[测试["右删失标记"] == 0]
    ax = axes[1]
    ax.scatter(未删失["出口粉尘浓度_mg_Nm3"], 未删失["线性预测浓度_mg_Nm3"],
               s=11, alpha=0.35, color=颜色["蓝"], edgecolors="none")
    边界 = [min(未删失["出口粉尘浓度_mg_Nm3"].min(), 未删失["线性预测浓度_mg_Nm3"].min()), 50]
    ax.plot(边界, 边界, color=颜色["灰"], ls="--", lw=1.3, label="理想线")
    ax.text(0.03, 0.95, f"未删失 MAE = {测试指标['未删失样本MAE_mg_Nm3']:.3f}",
            transform=ax.transAxes, va="top", color=颜色["深紫"])
    ax.set(xlabel="实测出口浓度 / (mg/Nm³)", ylabel="线性预测浓度 / (mg/Nm³)",
           title="（b）测试集未删失样本拟合")
    ax.legend()
    fig.suptitle("问题一：滞后选择与模型检验", fontsize=14, fontweight="bold")
    fig.tight_layout()
    保存图(fig, "问题一_滞后选择与模型检验")

    系数 = 参数表[(参数表["参数"] != "常数项") & (参数表["参数"] != "误差标准差σ")].copy()
    系数 = 系数.sort_values("估计值")
    fig, ax = plt.subplots(figsize=(9, 6.8))
    y = np.arange(len(系数))
    显著颜色 = np.where(系数["p值"] < 0.05, 颜色["紫"], 颜色["浅灰"])
    ax.hlines(y, 系数["置信区间下限"], 系数["置信区间上限"], color=显著颜色, lw=2)
    ax.scatter(系数["估计值"], y, c=显著颜色, s=42, edgecolor="white", zorder=3)
    ax.axvline(0, color=颜色["灰"], lw=1, ls="--")
    ax.set_yticks(y, 系数["参数"])
    ax.set(xlabel="标准化回归系数及95%置信区间", title="各工况变量对潜在出口浓度的影响")
    ax.text(0.99, 0.02, "紫色：p < 0.05；灰色：p ≥ 0.05", transform=ax.transAxes,
            ha="right", color=颜色["灰"])
    fig.tight_layout()
    保存图(fig, "问题一_参数影响与显著性")

    最大位置 = int(峰值数据["潜在浓度估计_mg_Nm3"].idxmax())
    起 = max(0, 最大位置 - 180); 止 = min(len(峰值数据), 最大位置 + 181)
    局部 = 峰值数据.iloc[起:止]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    ax = axes[0]
    ax.plot(峰值数据["时间戳"], 峰值数据["出口粉尘浓度_mg_Nm3"], color=颜色["浅蓝"], lw=0.8, label="传感器观测")
    ax.plot(峰值数据["时间戳"], 峰值数据["潜在浓度估计_mg_Nm3"], color=颜色["深紫"], lw=0.85, label="潜在浓度估计")
    峰 = 峰值数据[峰值数据["峰值标记"] == 1]
    ax.scatter(峰["时间戳"], 峰["潜在浓度估计_mg_Nm3"], s=9, color=颜色["粉"], label="识别峰值", zorder=4)
    ax.axhline(50, color=颜色["灰"], ls="--", lw=1, label="测量上限")
    ax.set(ylabel="浓度 / (mg/Nm³)", title="（a）七天浓度重构结果")
    ax.legend(ncol=4, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax = axes[1]
    ax.plot(局部["时间戳"], 局部["潜在浓度估计_mg_Nm3"], color=颜色["深紫"], lw=1.5, label="潜在浓度估计")
    ax.plot(局部["时间戳"], 局部["局部背景浓度_mg_Nm3"], color=颜色["蓝"], lw=1.2, ls="--", label="10分钟局部背景")
    局部峰 = 局部[局部["峰值标记"] == 1]
    ax.scatter(局部峰["时间戳"], 局部峰["潜在浓度估计_mg_Nm3"], s=28, color=颜色["粉"], zorder=4, label="识别峰值")
    ax.set(xlabel="时间", ylabel="浓度 / (mg/Nm³)", title="（b）最高潜在峰值附近6小时细节")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M")); ax.legend(ncol=3)
    fig.suptitle("问题一：潜在浓度重构与瞬时峰值识别", fontsize=14, fontweight="bold")
    fig.tight_layout()
    保存图(fig, "问题一_潜在浓度重构与峰值识别")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    顺序 = ["低", "中低", "中高", "高"]
    for i, ax in enumerate(axes.flat, start=1):
        子 = 电场统计[电场统计["电场"] == f"电场{i}"].set_index("振打频率组").loc[顺序].reset_index()
        x = np.arange(4)
        ax.bar(x, 子["峰值发生概率"] * 100, width=0.62, color=颜色["浅蓝"], edgecolor="white", label="峰值发生概率")
        ax2 = ax.twinx()
        ax2.plot(x, 子["平均相对峰高_mg_Nm3"], color=颜色["深紫"], marker="o", lw=2, label="平均相对峰高")
        ax.set_xticks(x, [f"{g}\n{t:.0f}s" for g, t in zip(顺序, 子["等效振打周期_s"])])
        ax.set(ylabel="峰值发生概率 / %", title=f"电场{i}")
        ax2.set_ylabel("平均相对峰高 / (mg/Nm³)")
        ax.grid(axis="x", visible=False); ax2.grid(False)
        if i == 1:
            线1, 标1 = ax.get_legend_handles_labels(); 线2, 标2 = ax2.get_legend_handles_labels()
            ax.legend(线1 + 线2, 标1 + 标2, loc="upper left")
    fig.suptitle("问题一：四电场振打强度分组与排放峰值\n横轴括号内为等效振打周期", fontsize=14, fontweight="bold")
    fig.tight_layout()
    保存图(fig, "问题一_四电场振打影响")


def 写报告(最优滞后: int, 滞后表: pd.DataFrame, 参数表: pd.DataFrame, 指标表: pd.DataFrame,
        峰值数据: pd.DataFrame, 电场统计: pd.DataFrame, 阈值: float, 最终模型: dict) -> None:
    指标 = dict(zip(指标表["指标"], 指标表["数值"]))
    显著 = 参数表[(参数表["参数"] != "常数项") & (参数表["参数"] != "误差标准差σ") & (参数表["p值"] < 0.05)]
    峰数 = int(峰值数据["峰值标记"].sum())
    最大峰 = float(峰值数据["潜在浓度估计_mg_Nm3"].max())
    最优验证 = float(滞后表.loc[滞后表["是否最优"] == 1, "验证集平均负对数似然"].iloc[0])
    电场摘要 = []
    for i in range(1, 5):
        子 = 电场统计[电场统计["电场"] == f"电场{i}"]
        高 = 子.loc[子["峰值发生概率"].idxmax()]
        电场摘要.append(f"- 电场{i}峰值发生概率最高的分组为“{高['振打频率组']}”，概率为{高['峰值发生概率']:.2%}，对应中位等效周期约{高['等效振打周期_s']:.1f} s。")

    内容 = f"""# 问题一计算结果

## 1. 求解方法

以出口粉尘浓度的潜在值为因变量，建立测量上限为50 mg/Nm³的右删失Tobit模型。输入包括温度、入口浓度、烟气流量、入口负荷、四级电场电压、四级电场5分钟平均振打频率，以及同一电场的标准化电压—振打频率交互项。候选统一滞后为0、1、3和5 min，先使用训练集拟合，再根据验证集平均负对数似然选择滞后；最终模型在训练集与验证集上重新估计，并只在测试集上评价。

最优统一滞后为 **{最优滞后} min**，验证集平均负对数似然为 **{最优验证:.4f}**。最终优化状态为“{'成功收敛' if 最终模型['收敛'] else '未正常收敛'}”，误差标准差估计为 **{最终模型['sigma']:.4f} mg/Nm³**。

## 2. 模型检验与参数解释

- 测试集平均负对数似然：{指标['测试集平均负对数似然']:.4f}；
- 测试集未删失样本MAE：{指标['未删失样本MAE_mg_Nm3']:.4f} mg/Nm³；
- 测试集样本数：{int(指标['测试集样本数'])}，其中未删失样本数为{int(指标['测试集未删失样本数'])}；
- Wald检验达到5%显著水平的解释变量共有{len(显著)}个。系数正负表示在其他变量不变时，对潜在出口浓度的影响方向；由于变量已经标准化，系数绝对值可用于比较同量纲变化下的相对影响强弱。

参数估计详见 `code/outputs/问题一计算结果/Tobit参数估计.csv`。需要注意，交互项系数反映电压与振打频率的联合变化效应，不宜脱离两个主效应单独解释。

## 3. 潜在浓度恢复与峰值识别

对观测值等于50 mg/Nm³的右删失样本，采用删失条件期望恢复潜在浓度估计；对未删失样本保留实测浓度。以过去10 min潜在浓度中位数为局部背景，训练集相对增量的95%分位数 **{阈值:.4f} mg/Nm³** 作为固定峰值阈值，并排除原始缺失后插值的时刻。全时段共识别{峰数}个瞬时峰值，最大潜在浓度估计为 **{最大峰:.3f} mg/Nm³**。该结果是模型条件下的潜在浓度估计，不等同于传感器未记录到的真实值。

## 4. 四电场振打影响

各电场按5分钟平均振打频率四分位分为低、中低、中高和高四组，并将组内中位频率换算为等效振打周期。

{chr(10).join(电场摘要)}

完整分组统计见 `code/outputs/问题一计算结果/四电场振打影响统计.csv`。组间差异用于描述统计关联，不能单独证明因果关系。

## 5. 图表与输出文件

- [滞后选择与模型检验](../figures/问题一计算结果/问题一_滞后选择与模型检验.pdf)
- [参数影响与显著性](../figures/问题一计算结果/问题一_参数影响与显著性.pdf)
- [潜在浓度重构与峰值识别](../figures/问题一计算结果/问题一_潜在浓度重构与峰值识别.pdf)
- [四电场振打影响](../figures/问题一计算结果/问题一_四电场振打影响.pdf)

主要数据文件包括滞后选择结果、Tobit参数估计、模型评价指标、潜在浓度与峰值序列、四电场振打影响统计和模型参数。出口浓度保持原始单位，所有输入变量沿用预处理阶段基于训练集计算的标准化参数。

## 6. 运行环境

- Python：{platform.python_version()}
- 运行命令：`{Path(sys.executable)} -B code/问题一.py`
- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    报告路径.write_text(内容, encoding="utf-8")


def main() -> None:
    开始 = datetime.now()
    结果目录.mkdir(parents=True, exist_ok=True)
    图片目录.mkdir(parents=True, exist_ok=True)
    报告路径.parent.mkdir(parents=True, exist_ok=True)
    if not 数据路径.exists():
        raise FileNotFoundError(f"未找到预处理数据：{数据路径}")

    数据 = pd.read_csv(数据路径, parse_dates=["时间戳"])
    y = 数据["出口粉尘浓度_mg_Nm3"].to_numpy(dtype=float)
    训练 = 数据["数据集"] == "训练集"
    验证 = 数据["数据集"] == "验证集"
    测试 = 数据["数据集"] == "测试集"

    滞后结果 = []
    模型缓存 = {}
    名称 = None
    for 滞后 in 候选滞后:
        X, 名称 = 构建设计矩阵(数据, 滞后)
        模型 = 拟合Tobit(X[训练], y[训练])
        模型缓存[滞后] = 模型
        验证nll = 样本负对数似然(模型["beta"], 模型["sigma"], X[验证], y[验证])
        滞后结果.append({
            "滞后分钟": 滞后,
            "训练集负对数似然": 模型["负对数似然"],
            "训练集平均负对数似然": 模型["负对数似然"] / int(训练.sum()),
            "验证集负对数似然": float(验证nll.sum()),
            "验证集平均负对数似然": float(验证nll.mean()),
            "误差标准差σ": 模型["sigma"],
            "是否收敛": int(模型["收敛"]),
            "迭代次数": 模型["迭代次数"],
        })
    滞后表 = pd.DataFrame(滞后结果)
    最优滞后 = int(滞后表.loc[滞后表["验证集平均负对数似然"].idxmin(), "滞后分钟"])
    滞后表["是否最优"] = (滞后表["滞后分钟"] == 最优滞后).astype(int)

    X, 名称 = 构建设计矩阵(数据, 最优滞后)
    拟合集 = 训练 | 验证
    最终模型 = 拟合Tobit(X[拟合集], y[拟合集], 初值=模型缓存[最优滞后]["参数"])
    if not 最终模型["收敛"]:
        raise RuntimeError(f"最终Tobit模型未收敛：{最终模型['消息']}")

    H = 数值Hessian(最终模型["参数"], X[拟合集], y[拟合集])
    协方差 = np.linalg.pinv(H, rcond=1e-10)
    标准误全 = np.sqrt(np.maximum(np.diag(协方差), 0))
    beta = 最终模型["beta"]
    beta_se = 标准误全[:-1]
    z值 = np.divide(beta, beta_se, out=np.full_like(beta, np.nan), where=beta_se > 0)
    p值 = 2 * norm.sf(np.abs(z值))
    参数表 = pd.DataFrame({
        "参数": 名称,
        "估计值": beta,
        "标准误": beta_se,
        "Wald_z值": z值,
        "p值": p值,
        "置信区间下限": beta - 1.96 * beta_se,
        "置信区间上限": beta + 1.96 * beta_se,
    })
    参数表 = pd.concat([参数表, pd.DataFrame([{
        "参数": "误差标准差σ", "估计值": 最终模型["sigma"], "标准误": np.nan,
        "Wald_z值": np.nan, "p值": np.nan, "置信区间下限": np.nan, "置信区间上限": np.nan,
    }])], ignore_index=True)

    mu = X @ beta
    测试nll = 样本负对数似然(beta, 最终模型["sigma"], X[测试], y[测试])
    测试未删失 = 测试.to_numpy() & (y < 删失上限 - 1e-10)
    mae = float(np.mean(np.abs(y[测试未删失] - mu[测试未删失])))
    指标字典 = {
        "测试集负对数似然": float(测试nll.sum()),
        "测试集平均负对数似然": float(测试nll.mean()),
        "未删失样本MAE_mg_Nm3": mae,
        "测试集样本数": int(测试.sum()),
        "测试集未删失样本数": int(测试未删失.sum()),
        "测试集删失比例": float(np.mean(y[测试] >= 删失上限 - 1e-10)),
    }
    指标表 = pd.DataFrame({"指标": list(指标字典.keys()), "数值": list(指标字典.values())})

    峰值数据, 峰值阈值 = 分析峰值(数据, mu, 最终模型["sigma"])
    电场统计 = 四电场统计(峰值数据)

    滞后表.to_csv(结果目录 / "滞后选择结果.csv", index=False, encoding="utf-8-sig")
    参数表.to_csv(结果目录 / "Tobit参数估计.csv", index=False, encoding="utf-8-sig")
    指标表.to_csv(结果目录 / "模型评价指标.csv", index=False, encoding="utf-8-sig")
    峰值数据.to_csv(结果目录 / "潜在浓度与峰值序列.csv", index=False, encoding="utf-8-sig")
    电场统计.to_csv(结果目录 / "四电场振打影响统计.csv", index=False, encoding="utf-8-sig")

    参数json = {
        "模型": "右删失Tobit回归",
        "删失上限_mg_Nm3": 删失上限,
        "候选滞后_分钟": 候选滞后,
        "最优滞后_分钟": 最优滞后,
        "解释变量": 名称,
        "回归系数": {k: float(v) for k, v in zip(名称, beta)},
        "误差标准差": 最终模型["sigma"],
        "峰值阈值_mg_Nm3": 峰值阈值,
        "峰值数量": int(峰值数据["峰值标记"].sum()),
        "拟合样本数": int(拟合集.sum()),
        "测试样本数": int(测试.sum()),
        "优化器收敛": 最终模型["收敛"],
        "优化器信息": 最终模型["消息"],
    }
    (结果目录 / "模型参数.json").write_text(json.dumps(参数json, ensure_ascii=False, indent=2), encoding="utf-8")

    绘制结果(滞后表, 参数表, 峰值数据, 电场统计, 指标字典)
    写报告(最优滞后, 滞后表, 参数表, 指标表, 峰值数据, 电场统计, 峰值阈值, 最终模型)

    日志 = [
        f"开始时间：{开始:%Y-%m-%d %H:%M:%S}",
        f"结束时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Python：{sys.executable} ({platform.python_version()})",
        f"输入数据：{数据路径}",
        f"数据规模：{len(数据)} 行，训练/验证/测试={int(训练.sum())}/{int(验证.sum())}/{int(测试.sum())}",
        f"最优滞后：{最优滞后} min",
        f"最终模型收敛：{最终模型['收敛']}，迭代次数：{最终模型['迭代次数']}",
        f"测试集平均负对数似然：{指标字典['测试集平均负对数似然']:.6f}",
        f"测试集未删失MAE：{mae:.6f} mg/Nm³",
        f"峰值阈值：{峰值阈值:.6f} mg/Nm³，峰值数量：{int(峰值数据['峰值标记'].sum())}",
    ]
    (结果目录 / "运行日志.txt").write_text("\n".join(日志), encoding="utf-8")
    # Windows 旧版终端可能使用 GBK，控制台摘要避免使用上标字符。
    print("\n".join(日志[4:]).replace("Nm³", "Nm3"))


if __name__ == "__main__":
    main()
