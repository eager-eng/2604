from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


项目根目录 = Path(__file__).resolve().parents[1]
默认压缩包路径 = 项目根目录 / "2604.zip"
默认结果目录 = 项目根目录 / "code" / "outputs" / "数据预处理结果"
默认图表目录 = 项目根目录 / "figures" / "数据预处理结果"
默认报告路径 = 项目根目录 / "reports" / "数据预处理结果报告.md"

原始字段 = [
    "timestamp",
    "Temp_C",
    "C_in_gNm3",
    "Q_Nm3h",
    "U1_kV",
    "U2_kV",
    "U3_kV",
    "U4_kV",
    "T1_s",
    "T2_s",
    "T3_s",
    "T4_s",
    "C_out_mgNm3",
    "P_total_kW",
]

中文字段映射 = {
    "timestamp": "时间戳",
    "Temp_C": "入口温度_℃",
    "C_in_gNm3": "入口粉尘浓度_g_Nm3",
    "Q_Nm3h": "烟气流量_Nm3_h",
    "U1_kV": "电场1电压_kV",
    "U2_kV": "电场2电压_kV",
    "U3_kV": "电场3电压_kV",
    "U4_kV": "电场4电压_kV",
    "T1_s": "电场1振打周期_s",
    "T2_s": "电场2振打周期_s",
    "T3_s": "电场3振打周期_s",
    "T4_s": "电场4振打周期_s",
    "C_out_mgNm3": "出口粉尘浓度_mg_Nm3",
    "P_total_kW": "总电耗_kW",
}

粉蓝紫配色 = {
    "粉": "#D7A6B4",
    "蓝": "#87B4CE",
    "紫": "#A89AC7",
    "浅粉": "#E8C8D1",
    "浅蓝": "#BED6E3",
    "浅紫": "#CEC4DE",
    "深灰": "#4E5661",
    "浅灰": "#D9DDE2",
    "青灰": "#9CBAB5",
}


def 配置日志(结果目录: Path) -> logging.Logger:
    结果目录.mkdir(parents=True, exist_ok=True)
    日志器 = logging.getLogger("数据预处理")
    日志器.setLevel(logging.INFO)
    日志器.handlers.clear()
    格式器 = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    文件处理器 = logging.FileHandler(结果目录 / "运行日志.txt", encoding="utf-8", mode="w")
    文件处理器.setFormatter(格式器)
    终端处理器 = logging.StreamHandler(sys.stdout)
    终端处理器.setFormatter(格式器)
    日志器.addHandler(文件处理器)
    日志器.addHandler(终端处理器)
    return 日志器


def 配置绘图样式() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Microsoft JhengHei"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#747B84",
            "axes.labelcolor": "#343A40",
            "xtick.color": "#4E5661",
            "ytick.color": "#4E5661",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
        }
    )


def 保存图表(图: plt.Figure, 图表目录: Path, 文件名: str) -> None:
    图表目录.mkdir(parents=True, exist_ok=True)
    图.savefig(图表目录 / f"{文件名}.pdf", facecolor="white")
    图.savefig(图表目录 / f"{文件名}.png", dpi=400, facecolor="white")
    plt.close(图)


def 从压缩包提取并转换(压缩包路径: Path, 输出路径: Path, 日志器: logging.Logger) -> None:
    if not 压缩包路径.exists():
        raise FileNotFoundError(f"未找到压缩包：{压缩包路径}")

    with zipfile.ZipFile(压缩包路径) as 压缩包:
        候选文件 = [名称 for 名称 in 压缩包.namelist() if 名称.lower().endswith(".xls")]
        if len(候选文件) != 1:
            raise ValueError(f"压缩包内应有且仅有一个 .xls 文件，实际找到 {len(候选文件)} 个")
        xls字节 = 压缩包.read(候选文件[0])

    输出路径.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="水泥电除尘数据_") as 临时目录:
        临时目录路径 = Path(临时目录)
        临时xls = 临时目录路径 / "Cement_ESP_Data.xls"
        临时脚本 = 临时目录路径 / "转换工作簿.ps1"
        临时xls.write_bytes(xls字节)

        临时脚本.write_text(
            """
param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
try {
    $workbook = $excel.Workbooks.Open($InputPath, 0, $true)
    try {
        $workbook.SaveAs($OutputPath, 51)
    }
    finally {
        $workbook.Close($false)
    }
}
finally {
    $excel.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
""".strip(),
            encoding="utf-8-sig",
        )
        日志器.info("正在通过隐藏 Excel 将旧版 .xls 转换为兼容 .xlsx 副本")
        结果 = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(临时脚本),
                "-InputPath",
                str(临时xls),
                "-OutputPath",
                str(输出路径),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if 结果.returncode != 0 or not 输出路径.exists():
            raise RuntimeError(f"Excel 兼容转换失败：{结果.stderr or 结果.stdout}")
        日志器.info("兼容副本已保存：%s", 输出路径)


def 读取原始数据(兼容副本路径: Path) -> pd.DataFrame:
    数据 = pd.read_excel(兼容副本路径, engine="openpyxl")
    缺失字段 = [字段 for 字段 in 原始字段 if 字段 not in 数据.columns]
    多余字段 = [字段 for 字段 in 数据.columns if 字段 not in 原始字段]
    if 缺失字段 or 多余字段:
        raise ValueError(f"字段不一致；缺失={缺失字段}，多余={多余字段}")
    return 数据.loc[:, 原始字段].copy()


def 检查并整理时间(数据: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    数据["timestamp"] = pd.to_datetime(数据["timestamp"], errors="coerce")
    无效时间数 = int(数据["timestamp"].isna().sum())
    if 无效时间数:
        raise ValueError(f"存在 {无效时间数} 个无效时间戳")
    数据 = 数据.sort_values("timestamp").reset_index(drop=True)

    重复时间数 = int(数据["timestamp"].duplicated().sum())
    时间差 = 数据["timestamp"].diff().dt.total_seconds().div(60)
    非一分钟间隔数 = int((时间差.iloc[1:] != 1).sum())
    质量信息 = {
        "原始行数": int(len(数据)),
        "原始列数": int(len(数据.columns)),
        "开始时间": 数据["timestamp"].min().isoformat(),
        "结束时间": 数据["timestamp"].max().isoformat(),
        "重复时间戳数": 重复时间数,
        "非1分钟间隔数": 非一分钟间隔数,
    }
    if 重复时间数 or 非一分钟间隔数:
        raise ValueError(f"时间序列不完整：重复={重复时间数}，非1分钟间隔={非一分钟间隔数}")
    return 数据, 质量信息


def 构造数据质量汇总(原始数据: pd.DataFrame, 质量信息: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    出口列 = "C_out_mgNm3"
    有效出口 = 原始数据[出口列].dropna()
    出口缺失数 = int(原始数据[出口列].isna().sum())
    上限数量 = int(np.isclose(有效出口.to_numpy(dtype=float), 50.0).sum())
    上限比例 = float(上限数量 / len(有效出口))

    汇总行 = [
        ("原始样本数", len(原始数据), "条"),
        ("原始字段数", len(原始数据.columns), "个"),
        ("时间范围", f"{质量信息['开始时间']} 至 {质量信息['结束时间']}", ""),
        ("重复时间戳", 质量信息["重复时间戳数"], "条"),
        ("非1分钟间隔", 质量信息["非1分钟间隔数"], "处"),
        ("出口浓度原始缺失", 出口缺失数, "条"),
        ("出口浓度有效值", len(有效出口), "条"),
        ("出口浓度等于50", 上限数量, "条"),
        ("有效出口浓度中50占比", 上限比例, "比例"),
        ("出口浓度最小值", float(有效出口.min()), "mg/Nm³"),
        ("出口浓度最大值", float(有效出口.max()), "mg/Nm³"),
    ]
    for 字段 in 原始字段:
        汇总行.append((f"{字段}缺失数", int(原始数据[字段].isna().sum()), "条"))
    汇总 = pd.DataFrame(汇总行, columns=["指标", "数值", "单位或说明"])
    扩展质量信息 = dict(质量信息)
    扩展质量信息.update(
        {
            "出口浓度原始缺失数": 出口缺失数,
            "出口浓度有效值数": int(len(有效出口)),
            "出口浓度等于50数量": 上限数量,
            "出口浓度等于50比例": 上限比例,
            "出口浓度最小值": float(有效出口.min()),
            "出口浓度最大值": float(有效出口.max()),
        }
    )
    return 汇总, 扩展质量信息


def 绘制预处理前图表(原始数据: pd.DataFrame, 图表目录: Path) -> None:
    绘图数据 = 原始数据.set_index("timestamp").resample("5min").mean(numeric_only=True).reset_index()
    时间 = 绘图数据["timestamp"]
    平均电压 = 绘图数据[[f"U{i}_kV" for i in range(1, 5)]].mean(axis=1)
    平均周期 = 绘图数据[[f"T{i}_s" for i in range(1, 5)]].mean(axis=1)

    图, 轴组 = plt.subplots(4, 2, figsize=(11.0, 9.2), sharex=True)
    序列配置 = [
        (绘图数据["Temp_C"], "入口温度（℃）", 粉蓝紫配色["粉"]),
        (绘图数据["C_in_gNm3"], "入口浓度（g/Nm³）", 粉蓝紫配色["蓝"]),
        (绘图数据["Q_Nm3h"], "烟气流量（Nm³/h）", 粉蓝紫配色["紫"]),
        (平均电压, "四电场平均电压（kV）", 粉蓝紫配色["青灰"]),
        (平均周期, "平均振打周期（s）", 粉蓝紫配色["浅紫"]),
        (绘图数据["C_out_mgNm3"], "出口浓度（mg/Nm³）", 粉蓝紫配色["粉"]),
        (绘图数据["P_total_kW"], "总电耗（kW）", 粉蓝紫配色["蓝"]),
    ]
    for 索引, (序列, 标签, 颜色) in enumerate(序列配置):
        轴 = 轴组.flat[索引]
        轴.plot(时间, 序列, color=颜色, linewidth=0.75)
        轴.set_ylabel(标签)
        轴.text(0.01, 0.90, f"({chr(97 + 索引)})", transform=轴.transAxes, color=粉蓝紫配色["深灰"])
        轴.margins(x=0)
    信息轴 = 轴组.flat[-1]
    信息轴.remove()
    图.text(
        0.67,
        0.20,
        "全局趋势按5分钟均值展示\n原始数据仍保持分钟级\n用于检查趋势、突变与边界",
        fontsize=11,
        color=粉蓝紫配色["深灰"],
        linespacing=1.8,
        ha="center",
        va="center",
    )
    for 轴 in 轴组[-1, :]:
        if 轴.has_data():
            轴.xaxis.set_major_locator(mdates.DayLocator())
            轴.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            轴.set_xlabel("日期")
    图.align_ylabels()
    图.subplots_adjust(left=0.11, right=0.98, bottom=0.07, top=0.98, hspace=0.43, wspace=0.30)
    保存图表(图, 图表目录, "预处理前_核心变量时序概览")

    缺失显示名称 = {
        "timestamp": "时间戳",
        "Temp_C": "入口温度",
        "C_in_gNm3": "入口粉尘浓度",
        "Q_Nm3h": "烟气流量",
        **{f"U{i}_kV": f"电场{i}电压" for i in range(1, 5)},
        **{f"T{i}_s": f"电场{i}振打周期" for i in range(1, 5)},
        "C_out_mgNm3": "出口粉尘浓度",
        "P_total_kW": "总电耗",
    }
    缺失数量 = 原始数据.isna().sum()
    缺失数量.index = [缺失显示名称.get(字段, 字段) for 字段 in 缺失数量.index]
    有效出口 = 原始数据["C_out_mgNm3"].dropna()
    图, 轴组 = plt.subplots(1, 2, figsize=(10.8, 4.3))
    轴组[0].barh(缺失数量.index, 缺失数量.values, color=粉蓝紫配色["浅蓝"], edgecolor="white")
    轴组[0].set_xlabel("缺失数量（条）")
    轴组[0].set_ylabel("")
    轴组[0].text(0.01, 0.96, "(a)", transform=轴组[0].transAxes, va="top")
    轴组[0].set_xlim(0, max(55, int(缺失数量.max() * 1.12)))
    for 行号, 数值 in enumerate(缺失数量.values):
        if 数值 > 0:
            轴组[0].text(数值 + 1, 行号, str(int(数值)), va="center", fontsize=9)

    bins = np.linspace(float(有效出口.min()), 50.0, 28)
    轴组[1].hist(有效出口, bins=bins, color=粉蓝紫配色["浅紫"], edgecolor="white", linewidth=0.6)
    上限数量 = int(np.isclose(有效出口.to_numpy(dtype=float), 50.0).sum())
    上限比例 = 上限数量 / len(有效出口)
    轴组[1].axvline(50, color=粉蓝紫配色["粉"], linewidth=1.8, linestyle="--")
    轴组[1].annotate(
        f"50 mg/Nm³：{上限数量} 条（{上限比例:.2%}）",
        xy=(50, 上限数量),
        xytext=(49.25, 上限数量 * 0.78),
        arrowprops={"arrowstyle": "->", "color": 粉蓝紫配色["深灰"]},
        color=粉蓝紫配色["深灰"],
        ha="left",
        fontsize=10,
    )
    轴组[1].set_xlabel("出口粉尘浓度（mg/Nm³）")
    轴组[1].set_ylabel("频数")
    轴组[1].text(0.01, 0.96, "(b)", transform=轴组[1].transAxes, va="top")
    图.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.96, wspace=0.34)
    保存图表(图, 图表目录, "预处理前_缺失与删失特征")


def 执行预处理(原始数据: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    数据 = 原始数据.rename(columns=中文字段映射).copy()
    出口列 = "出口粉尘浓度_mg_Nm3"
    数据["出口浓度原始缺失标记"] = 数据[出口列].isna().astype("int8")

    数据 = 数据.set_index("时间戳")
    数据[出口列] = 数据[出口列].interpolate(method="time", limit_direction="both")
    数据 = 数据.reset_index()
    数据["右删失标记"] = np.isclose(数据[出口列].to_numpy(dtype=float), 50.0).astype("int8")
    数据["未删失标记"] = (1 - 数据["右删失标记"]).astype("int8")

    数据["入口粉尘负荷_kg_h"] = 数据["入口粉尘浓度_g_Nm3"] * 数据["烟气流量_Nm3_h"] / 1000.0

    for i in range(1, 5):
        周期列 = f"电场{i}振打周期_s"
        频率列 = f"电场{i}振打频率_次_分钟"
        滚动列 = f"电场{i}_5分钟平均振打频率_次_分钟"
        数据[频率列] = 60.0 / 数据[周期列]
        数据[滚动列] = 数据[频率列].rolling(window=5, min_periods=5).mean()

    核心特征 = [
        "入口温度_℃",
        "入口粉尘浓度_g_Nm3",
        "烟气流量_Nm3_h",
        "入口粉尘负荷_kg_h",
        *[f"电场{i}电压_kV" for i in range(1, 5)],
        *[f"电场{i}_5分钟平均振打频率_次_分钟" for i in range(1, 5)],
    ]
    滞后基础特征 = [
        "入口温度_℃",
        "入口粉尘浓度_g_Nm3",
        "烟气流量_Nm3_h",
        "入口粉尘负荷_kg_h",
        *[f"电场{i}电压_kV" for i in range(1, 5)],
    ]
    候选滞后列: list[str] = []
    滞后数据: dict[str, pd.Series] = {}
    for 列名 in 滞后基础特征:
        for 滞后分钟 in (1, 3, 5):
            新列名 = f"{列名}_滞后{滞后分钟}分钟"
            滞后数据[新列名] = 数据[列名].shift(滞后分钟)
            候选滞后列.append(新列名)
    数据 = pd.concat([数据, pd.DataFrame(滞后数据, index=数据.index)], axis=1)

    起始日期 = 数据["时间戳"].dt.normalize().min()
    日序号 = (数据["时间戳"].dt.normalize() - 起始日期).dt.days
    数据["数据集"] = np.select(
        [日序号 <= 4, 日序号 == 5, 日序号 == 6],
        ["训练集", "验证集", "测试集"],
        default="超出范围",
    )
    if (数据["数据集"] == "超出范围").any():
        raise ValueError("时间范围不是预期的连续7天")

    待标准化列 = 核心特征 + 候选滞后列
    训练掩码 = 数据["数据集"].eq("训练集")
    标准化参数: dict[str, dict[str, float]] = {}
    标准化数据: dict[str, pd.Series] = {}
    for 列名 in 待标准化列:
        训练有效值 = 数据.loc[训练掩码, 列名].dropna()
        均值 = float(训练有效值.mean())
        标准差 = float(训练有效值.std(ddof=0))
        if not np.isfinite(标准差) or 标准差 <= 0:
            raise ValueError(f"变量 {列名} 的训练集标准差无效：{标准差}")
        标准化列 = f"{列名}_标准化"
        标准化数据[标准化列] = (数据[列名] - 均值) / 标准差
        标准化参数[列名] = {"均值": 均值, "标准差": 标准差}
    数据 = pd.concat([数据, pd.DataFrame(标准化数据, index=数据.index)], axis=1)

    for i in range(1, 5):
        电压标准化列 = f"电场{i}电压_kV_标准化"
        频率标准化列 = f"电场{i}_5分钟平均振打频率_次_分钟_标准化"
        数据[f"电场{i}电压_振打交互项"] = 数据[电压标准化列] * 数据[频率标准化列]

    动态必需列 = [
        *[f"电场{i}_5分钟平均振打频率_次_分钟" for i in range(1, 5)],
        *[f"{列名}_滞后5分钟" for 列名 in 滞后基础特征],
    ]
    参数信息: dict[str, object] = {
        "删失上限_mg_Nm3": 50.0,
        "线性插值字段": 出口列,
        "滚动窗口_分钟": 5,
        "候选滞后_分钟": [0, 1, 3, 5],
        "训练集日期": [str(起始日期.date()), str((起始日期 + pd.Timedelta(days=4)).date())],
        "验证集日期": str((起始日期 + pd.Timedelta(days=5)).date()),
        "测试集日期": str((起始日期 + pd.Timedelta(days=6)).date()),
        "核心特征": 核心特征,
        "滞后基础特征": 滞后基础特征,
        "动态必需列": 动态必需列,
        "标准化参数": 标准化参数,
    }
    return 数据, 参数信息


def 绘制预处理后图表(处理数据: pd.DataFrame, 图表目录: Path) -> None:
    时间 = 处理数据["时间戳"]
    插值掩码 = 处理数据["出口浓度原始缺失标记"].eq(1)
    图, 轴组 = plt.subplots(3, 1, figsize=(11.0, 7.8), sharex=True)
    轴组[0].plot(
        时间,
        处理数据["出口粉尘浓度_mg_Nm3"],
        color=粉蓝紫配色["蓝"],
        linewidth=0.8,
        label="插值后出口浓度",
    )
    轴组[0].scatter(
        时间[插值掩码],
        处理数据.loc[插值掩码, "出口粉尘浓度_mg_Nm3"],
        s=18,
        facecolor="white",
        edgecolor=粉蓝紫配色["粉"],
        linewidth=0.9,
        label="线性插值点",
        zorder=3,
    )
    轴组[0].set_ylabel("出口浓度（mg/Nm³）")
    轴组[0].legend(frameon=False, ncol=2, loc="lower right")
    轴组[0].text(0.01, 0.90, "(a)", transform=轴组[0].transAxes)

    轴组[1].plot(时间, 处理数据["入口粉尘负荷_kg_h"], color=粉蓝紫配色["紫"], linewidth=0.8)
    轴组[1].set_ylabel("入口粉尘负荷（kg/h）")
    轴组[1].text(0.01, 0.90, "(b)", transform=轴组[1].transAxes)

    线型 = ["-", "--", "-.", ":"]
    颜色 = [粉蓝紫配色["粉"], 粉蓝紫配色["蓝"], 粉蓝紫配色["紫"], 粉蓝紫配色["青灰"]]
    for i in range(1, 5):
        轴组[2].plot(
            时间,
            处理数据[f"电场{i}_5分钟平均振打频率_次_分钟"],
            color=颜色[i - 1],
            linestyle=线型[i - 1],
            linewidth=0.85,
            label=f"电场{i}",
        )
    轴组[2].set_ylabel("5分钟平均振打频率\n（次/分钟）")
    轴组[2].set_xlabel("日期")
    轴组[2].legend(frameon=False, ncol=4, loc="upper right")
    轴组[2].text(0.01, 0.90, "(c)", transform=轴组[2].transAxes)
    轴组[2].xaxis.set_major_locator(mdates.DayLocator())
    轴组[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for 轴 in 轴组:
        轴.margins(x=0)
    图.align_ylabels()
    图.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.98, hspace=0.27)
    保存图表(图, 图表目录, "预处理后_插值与动态特征")

    相关列 = [
        "入口温度_℃",
        "入口粉尘浓度_g_Nm3",
        "烟气流量_Nm3_h",
        "入口粉尘负荷_kg_h",
        *[f"电场{i}电压_kV" for i in range(1, 5)],
        *[f"电场{i}_5分钟平均振打频率_次_分钟" for i in range(1, 5)],
    ]
    简称 = {
        "入口温度_℃": "入口温度",
        "入口粉尘浓度_g_Nm3": "入口浓度",
        "烟气流量_Nm3_h": "烟气流量",
        "入口粉尘负荷_kg_h": "粉尘负荷",
        **{f"电场{i}电压_kV": f"电场{i}电压" for i in range(1, 5)},
        **{f"电场{i}_5分钟平均振打频率_次_分钟": f"电场{i}振打频率" for i in range(1, 5)},
    }
    相关系数 = 处理数据[相关列].rename(columns=简称).corr(method="spearman")
    颜色映射 = LinearSegmentedColormap.from_list(
        "粉蓝紫相关性", [粉蓝紫配色["蓝"], "#F8F7F8", 粉蓝紫配色["粉"]], N=256
    )
    图, 轴 = plt.subplots(figsize=(8.8, 7.4))
    sns.heatmap(
        相关系数,
        ax=轴,
        cmap=颜色映射,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.45,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7.5},
        cbar_kws={"label": "Spearman相关系数", "shrink": 0.78},
    )
    轴.set_xticklabels(轴.get_xticklabels(), rotation=42, ha="right")
    轴.set_yticklabels(轴.get_yticklabels(), rotation=0)
    图.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.98)
    保存图表(图, 图表目录, "预处理后_特征相关性")


def 写出结果文件(
    处理数据: pd.DataFrame,
    参数信息: dict[str, object],
    质量汇总: pd.DataFrame,
    结果目录: Path,
) -> dict[str, int]:
    结果目录.mkdir(parents=True, exist_ok=True)
    动态必需列 = list(参数信息["动态必需列"])
    建模数据 = 处理数据.dropna(subset=动态必需列).copy()
    数据规模 = {
        "预处理完整时序": int(len(处理数据)),
        "建模就绪数据": int(len(建模数据)),
        "原始训练集": int(处理数据["数据集"].eq("训练集").sum()),
        "原始验证集": int(处理数据["数据集"].eq("验证集").sum()),
        "原始测试集": int(处理数据["数据集"].eq("测试集").sum()),
        "建模训练集": int(建模数据["数据集"].eq("训练集").sum()),
        "建模验证集": int(建模数据["数据集"].eq("验证集").sum()),
        "建模测试集": int(建模数据["数据集"].eq("测试集").sum()),
    }
    参数信息["数据规模"] = 数据规模

    处理数据.to_csv(结果目录 / "预处理完整时序.csv", index=False, encoding="utf-8-sig")
    建模数据.to_csv(结果目录 / "建模就绪数据.csv", index=False, encoding="utf-8-sig")
    for 数据集名称 in ("训练集", "验证集", "测试集"):
        建模数据.loc[建模数据["数据集"].eq(数据集名称)].to_csv(
            结果目录 / f"{数据集名称}.csv", index=False, encoding="utf-8-sig"
        )
    质量汇总.to_csv(结果目录 / "数据质量汇总.csv", index=False, encoding="utf-8-sig")
    (结果目录 / "预处理参数.json").write_text(
        json.dumps(参数信息, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 数据规模


def 写出结果报告(
    报告路径: Path,
    图表目录: Path,
    质量信息: dict[str, object],
    数据规模: dict[str, int],
    结果目录: Path,
) -> None:
    报告路径.parent.mkdir(parents=True, exist_ok=True)
    图表相对目录 = Path("..") / "figures" / "数据预处理结果"
    报告内容 = f"""# 数据预处理结果报告

## 运行环境

- Python：{sys.version.split()[0]}
- 解释器：`{sys.executable}`
- 运行命令：`{sys.executable} code/数据预处理.py`
- 输入方式：原始 `.xls` 经隐藏 Excel 只读转换为兼容 `.xlsx` 副本。

## 原始数据质量

- 原始数据规模：{质量信息['原始行数']} 行、{质量信息['原始列数']} 列。
- 时间范围：{质量信息['开始时间']} 至 {质量信息['结束时间']}。
- 重复时间戳：{质量信息['重复时间戳数']} 条；非1分钟间隔：{质量信息['非1分钟间隔数']} 处。
- 出口浓度原始缺失：{质量信息['出口浓度原始缺失数']} 条，均按分钟序列执行线性插值并保留缺失标记。
- 有效出口浓度中，50 mg/Nm³记录为 {质量信息['出口浓度等于50数量']} 条，占 {质量信息['出口浓度等于50比例']:.2%}；预处理阶段仅生成右删失标记，不恢复潜变量。

## 特征构造与标准化

- 入口粉尘负荷：入口浓度与烟气流量乘积除以1000，单位为 kg/h。
- 振打频率：`60 / 振打周期`，单位为次/分钟。
- 动态特征：过去5分钟滚动平均振打频率，以及1、3、5分钟候选滞后。
- 标准化：连续输入仅使用训练集均值和标准差进行Z-score处理；出口浓度保持原单位。
- 交互项：同一电场的标准化电压乘以标准化5分钟平均振打频率。

## 数据集划分

| 数据 | 训练集 | 验证集 | 测试集 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 原始时间划分 | {数据规模['原始训练集']} | {数据规模['原始验证集']} | {数据规模['原始测试集']} | {数据规模['预处理完整时序']} |
| 建模就绪数据 | {数据规模['建模训练集']} | {数据规模['建模验证集']} | {数据规模['建模测试集']} | {数据规模['建模就绪数据']} |

建模数据仅删除由最大5分钟历史滞后造成的最前5行，验证集和测试集仍可利用此前时刻的历史信息。

## 必要可视化结果

- [预处理前核心变量时序概览]({(图表相对目录 / '预处理前_核心变量时序概览.pdf').as_posix()})
- [预处理前缺失与删失特征]({(图表相对目录 / '预处理前_缺失与删失特征.pdf').as_posix()})
- [预处理后插值与动态特征]({(图表相对目录 / '预处理后_插值与动态特征.pdf').as_posix()})
- [预处理后特征相关性]({(图表相对目录 / '预处理后_特征相关性.pdf').as_posix()})

图表仅覆盖时间连续性、缺失与删失特征、必要衍生变量和特征相关性，不额外生成装饰性图表。

## 输出文件

数据文件统一保存在 `{结果目录}`，包括完整时序、建模就绪数据、训练/验证/测试集、数据质量汇总、标准化参数和运行日志。
"""
    报告路径.write_text(报告内容, encoding="utf-8")


def 解析参数() -> argparse.Namespace:
    解析器 = argparse.ArgumentParser(description="水泥电除尘分钟级数据预处理与必要可视化")
    解析器.add_argument("--zip", type=Path, default=默认压缩包路径, help="包含原始 .xls 的压缩包路径")
    解析器.add_argument("--output-dir", type=Path, default=默认结果目录, help="数据结果目录")
    解析器.add_argument("--figure-dir", type=Path, default=默认图表目录, help="图表输出目录")
    解析器.add_argument("--report", type=Path, default=默认报告路径, help="结果报告路径")
    解析器.add_argument("--重新转换", action="store_true", help="即使兼容副本已存在也重新执行Excel转换")
    return 解析器.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    参数 = 解析参数()
    结果目录 = 参数.output_dir.resolve()
    图表目录 = 参数.figure_dir.resolve()
    报告路径 = 参数.report.resolve()
    日志器 = 配置日志(结果目录)
    配置绘图样式()

    兼容副本路径 = 结果目录 / "原始数据兼容副本.xlsx"
    if 参数.重新转换 or not 兼容副本路径.exists():
        从压缩包提取并转换(参数.zip.resolve(), 兼容副本路径, 日志器)
    else:
        日志器.info("复用已有兼容副本：%s", 兼容副本路径)

    日志器.info("读取兼容副本并检查原始数据")
    原始数据 = 读取原始数据(兼容副本路径)
    原始数据, 基础质量信息 = 检查并整理时间(原始数据)
    质量汇总, 质量信息 = 构造数据质量汇总(原始数据, 基础质量信息)
    日志器.info(
        "原始数据 %d 行；出口浓度缺失 %d 条；50 mg/Nm³占有效值 %.2f%%",
        len(原始数据),
        质量信息["出口浓度原始缺失数"],
        质量信息["出口浓度等于50比例"] * 100,
    )

    绘制预处理前图表(原始数据, 图表目录)
    处理数据, 参数信息 = 执行预处理(原始数据)
    绘制预处理后图表(处理数据, 图表目录)
    数据规模 = 写出结果文件(处理数据, 参数信息, 质量汇总, 结果目录)
    写出结果报告(报告路径, 图表目录, 质量信息, 数据规模, 结果目录)

    日志器.info("预处理完成：完整时序 %d 行，建模就绪 %d 行", 数据规模["预处理完整时序"], 数据规模["建模就绪数据"])
    日志器.info("数据结果：%s", 结果目录)
    日志器.info("图表结果：%s", 图表目录)
    日志器.info("结果报告：%s", 报告路径)


if __name__ == "__main__":
    main()
