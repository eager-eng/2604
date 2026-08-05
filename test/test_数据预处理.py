from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


项目根目录 = Path(__file__).resolve().parents[1]
结果目录 = 项目根目录 / "code" / "outputs" / "数据预处理结果"
图表目录 = 项目根目录 / "figures" / "数据预处理结果"
报告路径 = 项目根目录 / "reports" / "数据预处理结果报告.md"


class 数据预处理结果测试(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.完整数据 = pd.read_csv(结果目录 / "预处理完整时序.csv", encoding="utf-8-sig")
        cls.建模数据 = pd.read_csv(结果目录 / "建模就绪数据.csv", encoding="utf-8-sig")
        cls.质量汇总 = pd.read_csv(结果目录 / "数据质量汇总.csv", encoding="utf-8-sig")
        cls.参数 = json.loads((结果目录 / "预处理参数.json").read_text(encoding="utf-8"))

    def test_原始规模与时间连续性(self) -> None:
        self.assertEqual(len(self.完整数据), 10080)
        时间 = pd.to_datetime(self.完整数据["时间戳"])
        self.assertEqual(int(时间.duplicated().sum()), 0)
        self.assertTrue((时间.diff().dt.total_seconds().iloc[1:] == 60).all())

    def test_缺失插值与删失比例(self) -> None:
        self.assertEqual(int(self.完整数据["出口浓度原始缺失标记"].sum()), 50)
        self.assertEqual(int(self.完整数据["出口粉尘浓度_mg_Nm3"].isna().sum()), 0)
        指标映射 = dict(zip(self.质量汇总["指标"], self.质量汇总["数值"]))
        self.assertEqual(int(float(指标映射["出口浓度原始缺失"])), 50)
        self.assertAlmostEqual(float(指标映射["有效出口浓度中50占比"]), 0.5475, places=4)

    def test_数据集划分与最大滞后(self) -> None:
        完整计数 = self.完整数据["数据集"].value_counts().to_dict()
        self.assertEqual(完整计数, {"训练集": 7200, "验证集": 1440, "测试集": 1440})
        self.assertEqual(len(self.建模数据), 10075)
        建模计数 = self.建模数据["数据集"].value_counts().to_dict()
        self.assertEqual(建模计数, {"训练集": 7195, "验证集": 1440, "测试集": 1440})

    def test_训练集标准化(self) -> None:
        训练集 = self.完整数据[self.完整数据["数据集"].eq("训练集")]
        for 原列 in self.参数["核心特征"]:
            标准化列 = f"{原列}_标准化"
            有效值 = 训练集[标准化列].dropna().to_numpy(dtype=float)
            self.assertLess(abs(float(np.mean(有效值))), 1e-8, 标准化列)
            self.assertAlmostEqual(float(np.std(有效值, ddof=0)), 1.0, places=8, msg=标准化列)

    def test_输出文件存在且非空(self) -> None:
        数据文件 = [
            "原始数据兼容副本.xlsx",
            "预处理完整时序.csv",
            "建模就绪数据.csv",
            "训练集.csv",
            "验证集.csv",
            "测试集.csv",
            "数据质量汇总.csv",
            "预处理参数.json",
            "运行日志.txt",
        ]
        for 文件名 in 数据文件:
            路径 = 结果目录 / 文件名
            self.assertTrue(路径.exists(), 文件名)
            self.assertGreater(路径.stat().st_size, 0, 文件名)

        for 图名 in [
            "预处理前_核心变量时序概览",
            "预处理前_缺失与删失特征",
            "预处理后_插值与动态特征",
            "预处理后_特征相关性",
        ]:
            for 扩展名 in ("png", "pdf"):
                路径 = 图表目录 / f"{图名}.{扩展名}"
                self.assertTrue(路径.exists(), 路径.name)
                self.assertGreater(路径.stat().st_size, 0, 路径.name)

        self.assertTrue(报告路径.exists())
        self.assertGreater(报告路径.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
