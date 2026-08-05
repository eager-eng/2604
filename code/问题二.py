# -*- coding: utf-8 -*-
"""问题二：典型工况下电压—振打周期协同节能优化。"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture


项目目录 = Path(__file__).resolve().parents[1]
数据路径 = 项目目录 / "code" / "outputs" / "数据预处理结果" / "建模就绪数据.csv"
结果目录 = 项目目录 / "code" / "outputs" / "问题二计算结果"
图片目录 = 项目目录 / "figures" / "问题二计算结果"
报告路径 = 项目目录 / "reports" / "问题二计算结果.md"

删失上限 = 50.0
排放限值 = 10.0
候选聚类数 = [2, 3, 4, 5, 6]
岭参数候选 = [0.01, 0.1, 1.0, 10.0, 100.0]
随机种子 = 20260805
Deutsch电压指数 = 2.0
Deutsch电场权重 = np.full(4, 0.25, dtype=float)

颜色 = ["#83A9C9", "#D9A5B3", "#A99AC7", "#9CC8C5", "#C6B7D8", "#E4BCC8"]
深紫 = "#75658F"
灰色 = "#717784"
浅灰 = "#D9DCE3"


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
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
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


def 拟合GMM(数据: pd.DataFrame) -> tuple[GaussianMixture, pd.DataFrame, np.ndarray]:
    聚类列 = ["入口温度_℃_标准化", "入口粉尘浓度_g_Nm3_标准化", "烟气流量_Nm3_h_标准化"]
    训练掩码 = 数据["数据集"].eq("训练集").to_numpy()
    X训练 = 数据.loc[训练掩码, 聚类列].to_numpy(dtype=float)
    行 = []
    模型组 = {}
    for k in 候选聚类数:
        模型 = GaussianMixture(
            n_components=k, covariance_type="full", n_init=5,
            max_iter=500, random_state=随机种子,
        ).fit(X训练)
        模型组[k] = 模型
        行.append({
            "聚类数K": k,
            "BIC": float(模型.bic(X训练)),
            "AIC": float(模型.aic(X训练)),
            "是否收敛": int(模型.converged_),
            "迭代次数": int(模型.n_iter_),
        })
    选择表 = pd.DataFrame(行)
    最优K = int(选择表.loc[选择表["BIC"].idxmin(), "聚类数K"])
    选择表["是否最优"] = (选择表["聚类数K"] == 最优K).astype(int)
    最优模型 = 模型组[最优K]
    X全部 = 数据[聚类列].to_numpy(dtype=float)
    标签 = 最优模型.predict(X全部)
    return 最优模型, 选择表, 标签


def 计算工况统计(数据: pd.DataFrame, 标签: np.ndarray, 模型: GaussianMixture) -> pd.DataFrame:
    临时 = 数据.copy()
    临时["典型工况编号"] = 标签 + 1
    训练 = 临时[临时["数据集"] == "训练集"]
    行 = []
    for k in range(1, 模型.n_components + 1):
        子集 = 训练[训练["典型工况编号"] == k]
        行.append({
            "典型工况": f"工况{k}",
            "工况编号": k,
            "训练样本数": int(len(子集)),
            "训练样本比例": float(len(子集) / len(训练)),
            "烟气温度中心_℃": float(子集["入口温度_℃"].mean()),
            "入口浓度中心_g_Nm3": float(子集["入口粉尘浓度_g_Nm3"].mean()),
            "烟气流量中心_Nm3_h": float(子集["烟气流量_Nm3_h"].mean()),
            "入口负荷中心_kg_h": float(子集["入口粉尘负荷_kg_h"].mean()),
            "平均最大后验概率": float(model_probability_mean(模型, 子集)),
        })
    return pd.DataFrame(行)


def model_probability_mean(模型: GaussianMixture, 子集: pd.DataFrame) -> float:
    列 = ["入口温度_℃_标准化", "入口粉尘浓度_g_Nm3_标准化", "烟气流量_Nm3_h_标准化"]
    if len(子集) == 0:
        return float("nan")
    return float(模型.predict_proba(子集[列].to_numpy(dtype=float)).max(axis=1).mean())


def 构造能耗特征(数据: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    环境列 = ["入口温度_℃_标准化", "烟气流量_Nm3_h_标准化", "入口粉尘负荷_kg_h_标准化"]
    电压列 = [f"电场{i}电压_kV_标准化" for i in range(1, 5)]
    频率列 = [f"电场{i}_5分钟平均振打频率_次_分钟_标准化" for i in range(1, 5)]
    U = 数据[电压列].to_numpy(dtype=float)
    X = np.column_stack([数据[环境列].to_numpy(dtype=float), U, U**2, 数据[频率列].to_numpy(dtype=float)])
    名称 = ["烟气温度", "烟气流量", "入口负荷"]
    名称 += [f"电场{i}电压" for i in range(1, 5)]
    名称 += [f"电场{i}电压平方" for i in range(1, 5)]
    名称 += [f"电场{i}振打频率" for i in range(1, 5)]
    return X, 名称


def 拟合能耗模型(数据: pd.DataFrame) -> tuple[Ridge, pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    X, 名称 = 构造能耗特征(数据)
    y = 数据["总电耗_kW"].to_numpy(dtype=float)
    训练 = 数据["数据集"].eq("训练集").to_numpy()
    验证 = 数据["数据集"].eq("验证集").to_numpy()
    测试 = 数据["数据集"].eq("测试集").to_numpy()
    选择行 = []
    for alpha in 岭参数候选:
        模型 = Ridge(alpha=alpha).fit(X[训练], y[训练])
        预测 = 模型.predict(X[验证])
        选择行.append({
            "岭参数lambda": alpha,
            "验证集MAE_kW": float(mean_absolute_error(y[验证], 预测)),
            "验证集RMSE_kW": float(math.sqrt(mean_squared_error(y[验证], 预测))),
            "验证集R2": float(r2_score(y[验证], 预测)),
        })
    选择表 = pd.DataFrame(选择行)
    最优alpha = float(选择表.loc[选择表["验证集RMSE_kW"].idxmin(), "岭参数lambda"])
    选择表["是否最优"] = np.isclose(选择表["岭参数lambda"], 最优alpha).astype(int)
    最终模型 = Ridge(alpha=最优alpha).fit(X[训练 | 验证], y[训练 | 验证])
    全部预测 = 最终模型.predict(X)
    评价行 = []
    for 名, 掩码 in [("训练集", 训练), ("验证集", 验证), ("测试集", 测试)]:
        评价行.append({
            "数据集": 名,
            "样本数": int(掩码.sum()),
            "MAE_kW": float(mean_absolute_error(y[掩码], 全部预测[掩码])),
            "RMSE_kW": float(math.sqrt(mean_squared_error(y[掩码], 全部预测[掩码]))),
            "R2": float(r2_score(y[掩码], 全部预测[掩码])),
        })
    评价表 = pd.DataFrame(评价行)
    系数表 = pd.DataFrame({"特征": ["常数项"] + 名称, "系数": np.r_[最终模型.intercept_, 最终模型.coef_]})
    return 最终模型, 选择表, 评价表, 名称, 全部预测, 系数表


def 构造Tobit矩阵(数据: pd.DataFrame, 标准化: dict) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    基础列 = [
        "入口温度_℃_滞后3分钟_标准化", "入口粉尘浓度_g_Nm3_滞后3分钟_标准化",
        "烟气流量_Nm3_h_滞后3分钟_标准化", "入口粉尘负荷_kg_h_滞后3分钟_标准化",
    ]
    频率列 = [f"电场{i}_5分钟平均振打频率_次_分钟_标准化" for i in range(1, 5)]
    F = 数据[频率列].to_numpy(dtype=float)
    Q = 数据["烟气流量_Nm3_h_滞后3分钟"].to_numpy(dtype=float)
    机理项 = np.column_stack([
        z值(数据[f"电场{i}电压_kV_滞后3分钟"].to_numpy(dtype=float) ** 2 / Q,
           标准化[f"电场{i}电压平方除流量"])
        for i in range(1, 5)
    ])
    X = np.column_stack([np.ones(len(数据)), 数据[基础列].to_numpy(dtype=float), 机理项, F])
    名称 = ["常数项", "烟气温度", "入口浓度", "烟气流量", "入口负荷"]
    名称 += [f"电场{i}电压平方/烟气流量" for i in range(1, 5)]
    名称 += [f"电场{i}振打频率" for i in range(1, 5)]
    cin = 数据["入口粉尘浓度_g_Nm3"].to_numpy(dtype=float)
    cout = 数据["出口粉尘浓度_mg_Nm3"].to_numpy(dtype=float)
    c = np.log(1000.0 * cin / 删失上限)
    未删失 = cout < 删失上限 - 1e-10
    y = np.where(未删失, np.log(1000.0 * cin / cout), c)
    return X, 名称, y, c, 未删失


def 左删失负对数似然梯度(参数: np.ndarray, X: np.ndarray, y: np.ndarray,
                     c: np.ndarray, 未删失: np.ndarray) -> tuple[float, np.ndarray]:
    beta, eta = 参数[:-1], float(参数[-1])
    sigma = math.exp(eta)
    mu = X @ beta
    z = (y[未删失] - mu[未删失]) / sigma
    a = (c[~未删失] - mu[~未删失]) / sigma
    nll = np.sum(eta - norm.logpdf(z)) - np.sum(norm.logcdf(a))
    对mu = np.empty(len(y), dtype=float)
    对mu[未删失] = (mu[未删失] - y[未删失]) / sigma**2
    if np.any(~未删失):
        mills = np.exp(np.clip(norm.logpdf(a) - norm.logcdf(a), -40, 40))
        对mu[~未删失] = mills / sigma
        对eta删失 = np.sum(mills * a)
    else:
        对eta删失 = 0.0
    对eta = np.sum(1.0 - z**2) + 对eta删失
    梯度 = np.r_[X.T @ 对mu, 对eta]
    return float(nll), 梯度


def 拟合左删失Tobit(X: np.ndarray, y: np.ndarray, c: np.ndarray, 未删失: np.ndarray,
              初值: np.ndarray | None = None, 非负系数索引: tuple[int, ...] = ()) -> dict:
    if 初值 is None:
        beta0 = np.linalg.lstsq(X[未删失], y[未删失], rcond=None)[0]
        残差 = y[未删失] - X[未删失] @ beta0
        sigma0 = max(float(np.std(残差)), 0.01)
        初值 = np.r_[beta0, math.log(sigma0)]
    初值 = np.asarray(初值, dtype=float).copy()
    for j in 非负系数索引:
        初值[j] = max(float(初值[j]), 1e-6)

    def fun(p):
        return 左删失负对数似然梯度(p, X, y, c, 未删失)[0]

    def jac(p):
        return 左删失负对数似然梯度(p, X, y, c, 未删失)[1]

    边界 = [(None, None)] * X.shape[1] + [(math.log(1e-4), math.log(5.0))]
    for j in 非负系数索引:
        边界[j] = (0.0, None)
    结果 = minimize(fun, 初值, jac=jac, method="L-BFGS-B", bounds=边界,
                  options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 80})
    return {
        "参数": np.asarray(结果.x), "beta": np.asarray(结果.x[:-1]),
        "sigma": float(math.exp(结果.x[-1])), "负对数似然": float(结果.fun),
        "收敛": bool(结果.success), "迭代次数": int(结果.nit), "消息": str(结果.message),
    }


def 样本左删失NLL(beta, sigma, X, y, c, 未删失):
    mu = X @ beta
    结果 = np.empty(len(y), dtype=float)
    z = (y[未删失] - mu[未删失]) / sigma
    结果[未删失] = math.log(sigma) - norm.logpdf(z)
    a = (c[~未删失] - mu[~未删失]) / sigma
    结果[~未删失] = -norm.logcdf(a)
    return 结果


def 拟合除尘指数模型(数据: pd.DataFrame, 标准化: dict):
    X, 名称, y, c, 未删失 = 构造Tobit矩阵(数据, 标准化)
    训练 = 数据["数据集"].eq("训练集").to_numpy()
    验证 = 数据["数据集"].eq("验证集").to_numpy()
    测试 = 数据["数据集"].eq("测试集").to_numpy()
    非负系数索引 = (5, 6, 7, 8)
    训练模型 = 拟合左删失Tobit(X[训练], y[训练], c[训练], 未删失[训练],
                       非负系数索引=非负系数索引)
    最终模型 = 拟合左删失Tobit(X[训练 | 验证], y[训练 | 验证], c[训练 | 验证],
                       未删失[训练 | 验证], 初值=训练模型["参数"],
                       非负系数索引=非负系数索引)
    if not 最终模型["收敛"]:
        raise RuntimeError(f"左删失Tobit模型未收敛：{最终模型['消息']}")
    mu = X @ 最终模型["beta"]
    cout预测 = 1000.0 * 数据["入口粉尘浓度_g_Nm3"].to_numpy(dtype=float) * np.exp(-mu)
    评价行 = []
    for 名, 掩码 in [("训练集", 训练), ("验证集", 验证), ("测试集", 测试)]:
        nll = 样本左删失NLL(最终模型["beta"], 最终模型["sigma"], X[掩码], y[掩码], c[掩码], 未删失[掩码])
        可比 = 掩码 & 未删失
        评价行.append({
            "数据集": 名, "样本数": int(掩码.sum()),
            "平均负对数似然": float(nll.mean()),
            "未删失样本数": int(可比.sum()),
            "未删失除尘指数MAE": float(mean_absolute_error(y[可比], mu[可比])),
            "未删失出口浓度MAE_mg_Nm3": float(mean_absolute_error(
                数据.loc[可比, "出口粉尘浓度_mg_Nm3"], cout预测[可比])),
        })
    参数表 = pd.DataFrame({"参数": 名称, "估计值": 最终模型["beta"]})
    参数表["约束"] = ["非负" if i in 非负系数索引 else "无" for i in range(len(名称))]
    参数表 = pd.concat([参数表, pd.DataFrame([{"参数": "误差标准差sigma", "估计值": 最终模型["sigma"]}])], ignore_index=True)
    return 最终模型, pd.DataFrame(评价行), 参数表, mu, cout预测, 名称


def 训练标准化参数(数据: pd.DataFrame) -> dict[str, tuple[float, float]]:
    训练 = 数据[数据["数据集"] == "训练集"]
    变量 = ["入口温度_℃", "入口粉尘浓度_g_Nm3", "烟气流量_Nm3_h", "入口粉尘负荷_kg_h"]
    变量 += [f"电场{i}电压_kV" for i in range(1, 5)]
    变量 += [f"电场{i}_5分钟平均振打频率_次_分钟" for i in range(1, 5)]
    参数 = {v: (float(训练[v].mean()), float(训练[v].std(ddof=0))) for v in 变量}
    for v in ["入口温度_℃", "入口粉尘浓度_g_Nm3", "烟气流量_Nm3_h", "入口粉尘负荷_kg_h"] + [f"电场{i}电压_kV" for i in range(1, 5)]:
        列 = f"{v}_滞后3分钟"
        参数[列] = (float(训练[列].mean()), float(训练[列].std(ddof=0)))
    Q = 训练["烟气流量_Nm3_h_滞后3分钟"].to_numpy(dtype=float)
    for i in range(1, 5):
        值 = 训练[f"电场{i}电压_kV_滞后3分钟"].to_numpy(dtype=float) ** 2 / Q
        参数[f"电场{i}电压平方除流量"] = (float(值.mean()), float(值.std(ddof=0)))
    return 参数


def z值(x: np.ndarray | float, 参数: tuple[float, float]):
    return (np.asarray(x) - 参数[0]) / 参数[1]


def 决策边界(数据: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    下界, 上界, 行 = [], [], []
    for i in range(1, 5):
        列 = f"电场{i}电压_kV"
        lo, hi = float(数据[列].min()), float(数据[列].max())
        下界.append(lo); 上界.append(hi)
        行.append({"变量": f"U{i}", "类型": "连续", "下限": lo, "上限": hi, "单位": "kV"})
    for i in range(1, 5):
        列 = f"电场{i}振打周期_s"
        lo = int(math.floor(float(数据[列].min())))
        hi = int(math.ceil(float(数据[列].max())))
        下界.append(lo); 上界.append(hi)
        行.append({"变量": f"tau{i}", "类型": "整数", "下限": lo, "上限": hi, "单位": "s"})
    return np.asarray(下界, dtype=float), np.asarray(上界, dtype=float), pd.DataFrame(行)


def 构造候选特征(pop: np.ndarray, 工况: pd.Series, 标准化: dict):
    n = len(pop)
    U = pop[:, :4]
    tau = np.rint(pop[:, 4:]).astype(int)
    f = 60.0 / tau
    env_power = np.column_stack([
        np.full(n, z值(工况["烟气温度中心_℃"], 标准化["入口温度_℃"])),
        np.full(n, z值(工况["烟气流量中心_Nm3_h"], 标准化["烟气流量_Nm3_h"])),
        np.full(n, z值(工况["入口负荷中心_kg_h"], 标准化["入口粉尘负荷_kg_h"])),
    ])
    Uz_power = np.column_stack([z值(U[:, i], 标准化[f"电场{i+1}电压_kV"]) for i in range(4)])
    Fz = np.column_stack([z值(f[:, i], 标准化[f"电场{i+1}_5分钟平均振打频率_次_分钟"]) for i in range(4)])
    Xpower = np.column_stack([env_power, Uz_power, Uz_power**2, Fz])

    env_tobit = np.column_stack([
        np.full(n, z值(工况["烟气温度中心_℃"], 标准化["入口温度_℃_滞后3分钟"])),
        np.full(n, z值(工况["入口浓度中心_g_Nm3"], 标准化["入口粉尘浓度_g_Nm3_滞后3分钟"])),
        np.full(n, z值(工况["烟气流量中心_Nm3_h"], 标准化["烟气流量_Nm3_h_滞后3分钟"])),
        np.full(n, z值(工况["入口负荷中心_kg_h"], 标准化["入口粉尘负荷_kg_h_滞后3分钟"])),
    ])
    Q中心 = float(工况["烟气流量中心_Nm3_h"])
    机理项 = np.column_stack([
        z值(U[:, i] ** 2 / Q中心, 标准化[f"电场{i+1}电压平方除流量"])
        for i in range(4)
    ])
    Xtobit = np.column_stack([np.ones(n), env_tobit, 机理项, Fz])
    return Xpower, Xtobit, tau


def 约束量(pop: np.ndarray, 下界: np.ndarray, 上界: np.ndarray):
    U, tau = pop[:, :4], pop[:, 4:]
    vu = np.maximum(0.0, (U[:, 2] + U[:, 3] - U[:, 0] - U[:, 1]) / 2.0)
    vt = np.maximum(0.0, (tau[:, 0] + tau[:, 1] - tau[:, 2] - tau[:, 3]) / 2.0)
    vu /= max(float(np.ptp(np.r_[下界[:4], 上界[:4]])), 1e-9)
    vt /= max(float(np.ptp(np.r_[下界[4:], 上界[4:]])), 1e-9)
    return vu, vt


def Deutsch相对除尘指数(pop: np.ndarray, 基准策略: np.ndarray, 基准除尘指数: float,
                 tobit模型: dict, 标准化: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """以Tobit基准工况为锚点，按Deutsch–Anderson电压平方关系进行相对外推。"""
    U = pop[:, :4]
    tau = np.rint(pop[:, 4:]).astype(int)
    U参考 = np.asarray(基准策略[:4], dtype=float)
    tau参考 = np.rint(np.asarray(基准策略[4:], dtype=float)).astype(int)
    电压倍率 = np.sum(Deutsch电场权重 * (U / U参考) ** Deutsch电压指数, axis=1)
    f = 60.0 / tau
    f参考 = 60.0 / tau参考
    Fz = np.column_stack([
        z值(f[:, i], 标准化[f"电场{i+1}_5分钟平均振打频率_次_分钟"])
        for i in range(4)
    ])
    Fz参考 = np.asarray([
        float(z值(f参考[i], 标准化[f"电场{i+1}_5分钟平均振打频率_次_分钟"]))
        for i in range(4)
    ])
    # 机理约束Tobit中频率系数位于beta[9:13]，仅用于相对基准的振打修正。
    振打修正 = (Fz - Fz参考) @ np.asarray(tobit模型["beta"])[9:13]
    除尘指数 = 基准除尘指数 * 电压倍率 + 振打修正
    return 除尘指数, 电压倍率, 振打修正


def 运行遗传算法(工况: pd.Series, 能耗模型: Ridge, tobit模型: dict, 标准化: dict,
           下界: np.ndarray, 上界: np.ndarray, 基准电耗: float,
           基准策略: np.ndarray, 基准除尘指数: float,
           模式: str = "节能", seed: int = 随机种子) -> tuple[np.ndarray, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    种群数, 代数 = 100, 300 if 模式 == "节能" else 200
    pop = rng.uniform(下界, 上界, size=(种群数, 8))
    pop[:, 4:] = np.rint(pop[:, 4:])
    阈值 = math.log(1000.0 * float(工况["入口浓度中心_g_Nm3"]) / 排放限值)

    def 评价(p):
        Xp, _, tau = 构造候选特征(p, 工况, 标准化)
        power = 能耗模型.predict(Xp)
        yhat, 电压倍率, 振打修正 = Deutsch相对除尘指数(
            p, 基准策略, 基准除尘指数, tobit模型, 标准化)
        cout = 1000.0 * float(工况["入口浓度中心_g_Nm3"]) * np.exp(-yhat)
        vu, vt = 约束量(p, 下界, 上界)
        if 模式 == "节能":
            vc = np.maximum(0.0, (阈值 - yhat) / 阈值)
            fitness = power / 基准电耗 + 1e5 * vc**2 + 100.0 * vu**2 + 100.0 * vt**2
        else:
            vc = np.zeros(len(p))
            fitness = -yhat + 100.0 * vu**2 + 100.0 * vt**2
        feasible = (yhat >= 阈值 - 1e-10) & (vu <= 1e-10) & (vt <= 1e-10)
        return fitness, power, yhat, cout, vu, vt, feasible, vc

    历史 = []
    全局最佳 = None
    全局组约束最佳 = None
    for gen in range(代数 + 1):
        指标 = 评价(pop)
        fitness, power, yhat, cout, vu, vt, feasible, vc = 指标
        idx = int(np.argmin(fitness))
        可行idx = np.flatnonzero(feasible)
        if len(可行idx):
            fidx = int(可行idx[np.argmin(power[可行idx])])
            if 全局最佳 is None or power[fidx] < 全局最佳[1]:
                全局最佳 = (pop[fidx].copy(), float(power[fidx]), float(yhat[fidx]), float(cout[fidx]))
        组约束idx = np.flatnonzero((vu <= 1e-10) & (vt <= 1e-10))
        if len(组约束idx):
            if 模式 == "节能":
                gidx = int(组约束idx[np.argmin(fitness[组约束idx])])
                比较值 = float(fitness[gidx])
            else:
                gidx = int(组约束idx[np.argmax(yhat[组约束idx])])
                比较值 = float(-yhat[gidx])
            if 全局组约束最佳 is None or 比较值 < 全局组约束最佳[1]:
                全局组约束最佳 = (pop[gidx].copy(), 比较值)
        历史.append({
            "代数": gen, "模式": 模式,
            "当代最优适应度": float(fitness[idx]),
            "当代最小预测电耗_kW": float(power[idx]),
            "当代最大除尘指数": float(yhat.max()),
            "当代最小预测浓度_mg_Nm3": float(cout.min()),
            "当代可行个体数": int(feasible.sum()),
        })
        if gen == 代数:
            break
        精英数 = 2
        elite_idx = np.argsort(fitness)[:精英数]
        new_pop = [pop[i].copy() for i in elite_idx]
        while len(new_pop) < 种群数:
            候选1 = rng.integers(0, 种群数, 3); 候选2 = rng.integers(0, 种群数, 3)
            p1 = pop[候选1[np.argmin(fitness[候选1])]].copy()
            p2 = pop[候选2[np.argmin(fitness[候选2])]].copy()
            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < 0.8:
                alpha = rng.random(4)
                c1[:4] = alpha * p1[:4] + (1 - alpha) * p2[:4]
                c2[:4] = alpha * p2[:4] + (1 - alpha) * p1[:4]
                mask = rng.random(4) < 0.5
                c1[4:][mask], c2[4:][mask] = p2[4:][mask], p1[4:][mask]
            for child in (c1, c2):
                if rng.random() < 0.1:
                    j = int(rng.integers(0, 4))
                    child[j] += rng.normal(0, 0.05 * (上界[j] - 下界[j]))
                    q = int(rng.integers(4, 8))
                    child[q] += int(rng.integers(-5, 6))
                child[:] = np.clip(child, 下界, 上界)
                child[4:] = np.rint(child[4:])
                new_pop.append(child)
                if len(new_pop) >= 种群数:
                    break
        pop = np.asarray(new_pop)

    指标 = 评价(pop)
    idx = int(np.argmin(指标[0]))
    if 模式 == "节能" and 全局最佳 is not None:
        最佳 = 全局最佳[0]
    elif 全局组约束最佳 is not None:
        最佳 = 全局组约束最佳[0]
    else:
        最佳 = pop[idx].copy()
    final = 评价(最佳.reshape(1, -1))
    _, 最佳电压倍率, 最佳振打修正 = Deutsch相对除尘指数(
        最佳.reshape(1, -1), 基准策略, 基准除尘指数, tobit模型, 标准化)
    摘要 = {
        "预测电耗_kW": float(final[1][0]), "预测除尘指数": float(final[2][0]),
        "预测出口浓度_mg_Nm3": float(final[3][0]), "电压分组违反": float(final[4][0]),
        "周期分组违反": float(final[5][0]), "是否可行": bool(final[6][0]),
        "除尘指数阈值": 阈值,
        "相对电压平方倍率": float(最佳电压倍率[0]),
        "振打除尘指数修正": float(最佳振打修正[0]),
    }
    return 最佳, pd.DataFrame(历史), 摘要


def 优化所有工况(数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 下界, 上界):
    训练 = 数据[数据["数据集"] == "训练集"].copy()
    训练["工况编号"] = 标签[数据["数据集"].eq("训练集").to_numpy()] + 1
    结果行, 历史组, 检查行 = [], [], []
    for _, 工况 in 工况表.iterrows():
        k = int(工况["工况编号"])
        子集 = 训练[训练["工况编号"] == k]
        基准 = np.r_[
            [float(子集[f"电场{i}电压_kV"].median()) for i in range(1, 5)],
            [int(round(float(子集[f"电场{i}振打周期_s"].median()))) for i in range(1, 5)],
        ]
        Xp, Xt, _ = 构造候选特征(基准.reshape(1, -1), 工况, 标准化)
        基准电耗 = float(能耗模型.predict(Xp)[0])
        基准Y = float((Xt @ tobit模型["beta"])[0])
        基准浓度 = float(1000.0 * 工况["入口浓度中心_g_Nm3"] * math.exp(-基准Y))
        最佳, 历史, 摘要 = 运行遗传算法(工况, 能耗模型, tobit模型, 标准化, 下界, 上界,
                                基准电耗, 基准, 基准Y, 模式="节能", seed=随机种子 + k)
        最大解, 最大历史, 最大摘要 = 运行遗传算法(工况, 能耗模型, tobit模型, 标准化, 下界, 上界,
                                  基准电耗, 基准, 基准Y, 模式="最大除尘指数", seed=随机种子 + 100 + k)
        # 两次搜索都可能发现更高的除尘指数，统一保留其中更优者，保证汇总字段自洽。
        if 摘要["预测除尘指数"] > 最大摘要["预测除尘指数"]:
            最大解, 最大摘要 = 最佳.copy(), 摘要.copy()
        历史["典型工况"] = f"工况{k}"; 最大历史["典型工况"] = f"工况{k}"
        历史组.extend([历史, 最大历史])
        行 = {
            "典型工况": f"工况{k}",
            "基准预测电耗_kW": 基准电耗, "优化预测电耗_kW": 摘要["预测电耗_kW"],
            "节电率": (基准电耗 - 摘要["预测电耗_kW"]) / 基准电耗,
            "基准预测出口浓度_mg_Nm3": 基准浓度,
            "优化预测出口浓度_mg_Nm3": 摘要["预测出口浓度_mg_Nm3"],
            "优化预测除尘指数": 摘要["预测除尘指数"],
            "相对电压平方倍率": 摘要["相对电压平方倍率"],
            "振打除尘指数修正": 摘要["振打除尘指数修正"],
            "除尘指数阈值": 摘要["除尘指数阈值"],
            "边界内最大除尘指数": 最大摘要["预测除尘指数"],
            "边界内最低预测浓度_mg_Nm3": 最大摘要["预测出口浓度_mg_Nm3"],
            "是否满足10mg_Nm3": int(摘要["是否可行"]),
        }
        for i in range(4):
            行[f"基准U{i+1}_kV"] = float(基准[i]); 行[f"优化U{i+1}_kV"] = float(最佳[i])
            行[f"基准tau{i+1}_s"] = int(基准[4+i]); 行[f"优化tau{i+1}_s"] = int(round(最佳[4+i]))
        结果行.append(行)
        检查行.append({
            "典型工况": f"工况{k}",
            "排放约束满足": int(摘要["预测出口浓度_mg_Nm3"] <= 排放限值 + 1e-8),
            "电压边界满足": int(np.all((最佳[:4] >= 下界[:4]) & (最佳[:4] <= 上界[:4]))),
            "周期边界及整数满足": int(np.all((最佳[4:] >= 下界[4:]) & (最佳[4:] <= 上界[4:])) and np.allclose(最佳[4:], np.rint(最佳[4:]))),
            "电压分组约束满足": int((最佳[0] + 最佳[1]) / 2 >= (最佳[2] + 最佳[3]) / 2 - 1e-8),
            "周期分组约束满足": int((最佳[4] + 最佳[5]) / 2 <= (最佳[6] + 最佳[7]) / 2 + 1e-8),
            "能耗低于基准": int(摘要["预测电耗_kW"] <= 基准电耗 + 1e-8),
            "边界内存在排放可行解": int(最大摘要["预测除尘指数"] >= 摘要["除尘指数阈值"] - 1e-8),
        })
    return pd.DataFrame(结果行), pd.concat(历史组, ignore_index=True), pd.DataFrame(检查行)


def 绘图(数据, 标签, gmm选择, 工况表, 能耗预测, cout预测, 优化表, 收敛表):
    设置绘图风格()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(gmm选择["聚类数K"], gmm选择["BIC"], marker="o", lw=2, color=深紫)
    best = gmm选择[gmm选择["是否最优"] == 1].iloc[0]
    axes[0].scatter(best["聚类数K"], best["BIC"], s=100, color=颜色[1], edgecolor="white", zorder=4, label="BIC最小")
    axes[0].set(xlabel="聚类数 K", ylabel="BIC", title="（a）聚类数选择"); axes[0].legend()
    样本 = np.arange(0, len(数据), 5)
    for k in sorted(np.unique(标签)):
        idx = 样本[标签[样本] == k]
        axes[1].scatter(数据.loc[idx, "入口温度_℃"], 数据.loc[idx, "入口粉尘浓度_g_Nm3"],
                        s=10, alpha=0.45, color=颜色[k % len(颜色)], label=f"工况{k+1}")
    axes[1].set(xlabel="烟气温度 / ℃", ylabel="入口粉尘浓度 / (g/Nm³)", title="（b）典型工况分布（每5分钟抽样）")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout(); 保存图(fig, "问题二_典型工况划分")

    测试 = 数据["数据集"].eq("测试集").to_numpy()
    未删失测试 = 测试 & 数据["未删失标记"].eq(1).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].scatter(数据.loc[测试, "总电耗_kW"], 能耗预测[测试], s=13, alpha=0.4, color=颜色[0], edgecolors="none")
    lim = [min(float(数据.loc[测试, "总电耗_kW"].min()), float(能耗预测[测试].min())), max(float(数据.loc[测试, "总电耗_kW"].max()), float(能耗预测[测试].max()))]
    axes[0].plot(lim, lim, ls="--", color=灰色, lw=1.2)
    axes[0].set(xlabel="实测总电耗 / kW", ylabel="预测总电耗 / kW", title="（a）测试集能耗预测")
    axes[1].scatter(数据.loc[未删失测试, "出口粉尘浓度_mg_Nm3"], cout预测[未删失测试], s=13, alpha=0.4, color=颜色[2], edgecolors="none")
    lim2 = [min(float(数据.loc[未删失测试, "出口粉尘浓度_mg_Nm3"].min()), float(cout预测[未删失测试].min())), max(float(数据.loc[未删失测试, "出口粉尘浓度_mg_Nm3"].max()), float(cout预测[未删失测试].max()))]
    axes[1].plot(lim2, lim2, ls="--", color=灰色, lw=1.2)
    axes[1].set(xlabel="实测出口浓度 / (mg/Nm³)", ylabel="预测出口浓度 / (mg/Nm³)", title="（b）测试集未删失浓度预测")
    fig.tight_layout(); 保存图(fig, "问题二_模型预测效果")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for k, 子 in 收敛表[收敛表["模式"] == "节能"].groupby("典型工况"):
        序号 = int(str(k).replace("工况", "")) - 1
        ax.plot(子["代数"], 子["当代最优适应度"], lw=1.7,
                color=颜色[序号 % len(颜色)], label=k)
    ax.set(xlabel="迭代代数", ylabel="当代最优罚函数值", title="遗传算法罚函数收敛过程")
    ax.legend(ncol=3); fig.tight_layout(); 保存图(fig, "问题二_遗传算法收敛")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(优化表)); width = 0.35
    axes[0].bar(x-width/2, 优化表["基准预测电耗_kW"], width, color=浅灰, label="基准策略")
    axes[0].bar(x+width/2, 优化表["优化预测电耗_kW"], width, color=颜色[0], label="达标最低电耗方案")
    axes[0].set_xticks(x, 优化表["典型工况"]); axes[0].set(ylabel="预测总电耗 / kW", title="（a）历史基准与达标方案能耗")
    axes[0].legend()
    axes[1].bar(x-width/2, 优化表["基准预测出口浓度_mg_Nm3"], width, color=浅灰, label="基准策略")
    axes[1].bar(x+width/2, 优化表["优化预测出口浓度_mg_Nm3"], width, color=颜色[1], label="达标最低电耗方案")
    axes[1].axhline(排放限值, ls="--", lw=1.4, color=深紫, label="10 mg/Nm³限值")
    axes[1].set_xticks(x, 优化表["典型工况"]); axes[1].set(ylabel="预测出口浓度 / (mg/Nm³)", title="（b）历史基准与达标方案排放")
    axes[1].legend()
    fig.tight_layout(); 保存图(fig, "问题二_优化前后对比")


def data_max_min(a, b):
    return min(float(np.min(a)), float(np.min(b))), max(float(np.max(a)), float(np.max(b)))


def 写报告(gmm选择, 工况表, 能耗评价, tobit评价, 优化表, 检查表, 边界表, tobit模型):
    最优K = int(gmm选择.loc[gmm选择["是否最优"] == 1, "聚类数K"].iloc[0])
    测试能耗 = 能耗评价[能耗评价["数据集"] == "测试集"].iloc[0]
    测试Tobit = tobit评价[tobit评价["数据集"] == "测试集"].iloc[0]
    可行数 = int(优化表["是否满足10mg_Nm3"].sum())
    机理系数 = np.asarray(tobit模型["beta"])[5:9]
    正机理系数数 = int(np.sum(机理系数 > 1e-10))
    结果行 = []
    for _, r in 优化表.iterrows():
        状态 = "满足" if r["是否满足10mg_Nm3"] else "未满足"
        变化率 = r["优化预测电耗_kW"] / r["基准预测电耗_kW"] - 1.0
        变化描述 = f"增加{变化率:.2%}" if 变化率 >= 0 else f"降低{-变化率:.2%}"
        结果行.append(f"- {r['典型工况']}：达标最低电耗方案浓度{r['优化预测出口浓度_mg_Nm3']:.3f} mg/Nm³（{状态}10 mg/Nm³），预测电耗{r['优化预测电耗_kW']:.2f} kW，较未达标历史基准{变化描述}。")
    内容 = f"""# 问题二计算结果

## 1. 方法与数据

问题二读取预处理阶段生成的10075条建模数据。使用训练集标准化参数，以温度、入口浓度和烟气流量建立完整协方差GMM；使用岭回归建立全局总电耗响应模型；利用左删失Tobit模型确定各典型工况的历史基准除尘指数，再依据Deutsch–Anderson机理按四级电场相对电压平方倍率进行外推，并叠加Tobit振打频率项给出的相对修正；最后使用实数—整数混合编码遗传算法优化四级电场电压与振打周期。

## 2. 典型工况划分

BIC选择的最优聚类数为 **{最优K}**。各工况中心、样本数和平均最大后验概率见 `典型工况统计.csv`，逐时刻标签见 `典型工况标记.csv`。

## 3. 模型检验

- 岭回归测试集：MAE={测试能耗['MAE_kW']:.3f} kW，RMSE={测试能耗['RMSE_kW']:.3f} kW，R²={测试能耗['R2']:.4f}；
- 左删失Tobit测试集：平均负对数似然={测试Tobit['平均负对数似然']:.4f}，未删失出口浓度MAE={测试Tobit['未删失出口浓度MAE_mg_Nm3']:.4f} mg/Nm³；
- 左删失Tobit误差标准差：{tobit模型['sigma']:.6f}。
- 四个电场的“电压平方/烟气流量”系数中有 **{正机理系数数}/4** 个严格大于0；其余系数位于非负约束边界，表示现有数据未识别出相应电压项的正向增益。
- 优化阶段不使用上述零电压系数进行远距离预测，而采用Deutsch–Anderson相对标定：以Tobit工况基准为锚点，电压指数取 **{Deutsch电压指数:.1f}**，四级电场采用等权重。

## 4. 遗传算法搜索结果

在题面未另行给出设备额定边界的情况下，控制边界取清洗后完整数据的最小值和最大值，振打周期按整数处理。各工况结果如下：

{chr(10).join(结果行)}

共 **{可行数}/{len(优化表)}** 类工况得到满足10 mg/Nm³排放约束的可行方案。若某工况未满足，上述数值仅表示历史运行边界内的最优候选；结果表同时给出边界内最大除尘指数及最低预测浓度，不将罚函数最小但违反排放约束的个体误报为可行方案。

## 5. 约束检查

完整检查见 `约束检查.csv`。检查内容包括排放限值、电压边界、周期边界及整数性、前后级分组均值约束、与历史基准的电耗比较和边界内可行性。历史基准未满足10 mg/Nm³，因此“电耗低于基准”仅为比较指标，不属于优化约束；优化目标是在达标方案中使预测电耗最低。本阶段不进行敏感性分析。

## 6. 图表

- [典型工况划分](../figures/问题二计算结果/问题二_典型工况划分.pdf)
- [模型预测效果](../figures/问题二计算结果/问题二_模型预测效果.pdf)
- [遗传算法收敛](../figures/问题二计算结果/问题二_遗传算法收敛.pdf)
- [优化前后对比](../figures/问题二计算结果/问题二_优化前后对比.pdf)

## 7. 运行环境

- Python：{platform.python_version()}
- 运行命令：`{Path(sys.executable)} -B code/问题二.py`
- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}
"""
    报告路径.write_text(内容, encoding="utf-8")


def main():
    开始 = datetime.now()
    结果目录.mkdir(parents=True, exist_ok=True)
    图片目录.mkdir(parents=True, exist_ok=True)
    报告路径.parent.mkdir(parents=True, exist_ok=True)
    数据 = pd.read_csv(数据路径, parse_dates=["时间戳"])

    gmm模型, gmm选择, 标签 = 拟合GMM(数据)
    工况表 = 计算工况统计(数据, 标签, gmm模型)
    能耗模型, 岭选择, 能耗评价, 能耗名称, 能耗预测, 能耗系数 = 拟合能耗模型(数据)
    标准化 = 训练标准化参数(数据)
    tobit模型, tobit评价, tobit参数, 除尘指数预测, 浓度预测, tobit名称 = 拟合除尘指数模型(数据, 标准化)
    下界, 上界, 边界表 = 决策边界(数据)
    优化表, 收敛表, 检查表 = 优化所有工况(
        数据, 标签, 工况表, 能耗模型, tobit模型, 标准化, 下界, 上界)

    标记表 = pd.DataFrame({
        "时间戳": 数据["时间戳"], "数据集": 数据["数据集"],
        "典型工况": [f"工况{x+1}" for x in 标签],
        "最大后验概率": gmm模型.predict_proba(数据[["入口温度_℃_标准化", "入口粉尘浓度_g_Nm3_标准化", "烟气流量_Nm3_h_标准化"]].to_numpy(dtype=float)).max(axis=1),
    })

    gmm选择.to_csv(结果目录 / "GMM模型选择.csv", index=False, encoding="utf-8-sig")
    工况表.to_csv(结果目录 / "典型工况统计.csv", index=False, encoding="utf-8-sig")
    标记表.to_csv(结果目录 / "典型工况标记.csv", index=False, encoding="utf-8-sig")
    岭选择.to_csv(结果目录 / "岭回归参数选择.csv", index=False, encoding="utf-8-sig")
    能耗评价.to_csv(结果目录 / "能耗模型评价.csv", index=False, encoding="utf-8-sig")
    能耗系数.to_csv(结果目录 / "能耗模型系数.csv", index=False, encoding="utf-8-sig")
    tobit评价.to_csv(结果目录 / "左删失Tobit模型评价.csv", index=False, encoding="utf-8-sig")
    tobit参数.to_csv(结果目录 / "左删失Tobit模型参数.csv", index=False, encoding="utf-8-sig")
    边界表.to_csv(结果目录 / "决策变量边界.csv", index=False, encoding="utf-8-sig")
    优化表.to_csv(结果目录 / "优化结果_Deutsch相对外推.csv", index=False, encoding="utf-8-sig")
    收敛表.to_csv(结果目录 / "遗传算法收敛历史.csv", index=False, encoding="utf-8-sig")
    检查表.to_csv(结果目录 / "约束检查.csv", index=False, encoding="utf-8-sig")

    参数json = {
        "最优聚类数": int(gmm模型.n_components),
        "最优岭参数": float(能耗模型.alpha),
        "左删失Tobit误差标准差": tobit模型["sigma"],
        "左删失Tobit收敛": tobit模型["收敛"],
        "Tobit结构": "机理约束：电压平方/烟气流量系数非负",
        "排放外推模型": "以Tobit工况基准为锚点的Deutsch-Anderson相对标定",
        "Deutsch电压指数": Deutsch电压指数,
        "Deutsch电场权重": Deutsch电场权重.tolist(),
        "决策边界来源": "清洗后完整数据最小值与最大值",
        "排放限值_mg_Nm3": 排放限值,
        "响应滞后_分钟": 3,
        "决策变量下界": 下界.tolist(), "决策变量上界": 上界.tolist(),
        "遗传算法参数": {"种群规模": 100, "最大代数": 300, "交叉概率": 0.8, "变异概率": 0.1, "精英数": 2, "随机种子": 随机种子},
        "可行工况数": int(优化表["是否满足10mg_Nm3"].sum()),
        "典型工况总数": int(len(优化表)),
    }
    (结果目录 / "问题二模型参数.json").write_text(json.dumps(参数json, ensure_ascii=False, indent=2), encoding="utf-8")

    绘图(数据, 标签, gmm选择, 工况表, 能耗预测, 浓度预测, 优化表, 收敛表)
    写报告(gmm选择, 工况表, 能耗评价, tobit评价, 优化表, 检查表, 边界表, tobit模型)

    日志 = [
        f"开始时间：{开始:%Y-%m-%d %H:%M:%S}", f"结束时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Python：{sys.executable} ({platform.python_version()})", f"输入数据：{数据路径}",
        f"最优聚类数：{gmm模型.n_components}", f"最优岭参数：{能耗模型.alpha}",
        f"左删失Tobit收敛：{tobit模型['收敛']}，sigma={tobit模型['sigma']:.6f}",
        f"满足10 mg/Nm3工况数：{int(优化表['是否满足10mg_Nm3'].sum())}/{len(优化表)}",
    ]
    (结果目录 / "运行日志.txt").write_text("\n".join(日志), encoding="utf-8")
    print("\n".join(日志[4:]))


if __name__ == "__main__":
    main()
