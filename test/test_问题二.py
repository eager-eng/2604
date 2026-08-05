# -*- coding: utf-8 -*-
"""问题二输出的可复现性、模型质量与优化约束测试。"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


项目目录 = Path(__file__).resolve().parents[1]
结果目录 = 项目目录 / "code" / "outputs" / "问题二计算结果"
图片目录 = 项目目录 / "figures" / "问题二计算结果"
报告路径 = 项目目录 / "reports" / "问题二计算结果.md"


class 问题二测试(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.GMM = pd.read_csv(结果目录 / "GMM模型选择.csv")
        cls.工况 = pd.read_csv(结果目录 / "典型工况统计.csv")
        cls.标记 = pd.read_csv(结果目录 / "典型工况标记.csv")
        cls.能耗评价 = pd.read_csv(结果目录 / "能耗模型评价.csv")
        cls.Tobit评价 = pd.read_csv(结果目录 / "左删失Tobit模型评价.csv")
        cls.Tobit参数 = pd.read_csv(结果目录 / "左删失Tobit模型参数.csv")
        cls.优化 = pd.read_csv(结果目录 / "优化结果_Deutsch相对外推.csv")
        cls.检查 = pd.read_csv(结果目录 / "约束检查.csv")
        cls.参数 = json.loads((结果目录 / "问题二模型参数.json").read_text(encoding="utf-8"))

    def test_GMM模型选择(self):
        self.assertEqual(self.GMM["聚类数K"].tolist(), [2, 3, 4, 5, 6])
        self.assertTrue((self.GMM["是否收敛"] == 1).all())
        self.assertEqual(int(self.GMM["是否最优"].sum()), 1)
        最优K = int(self.GMM.loc[self.GMM["BIC"].idxmin(), "聚类数K"])
        self.assertEqual(最优K, int(self.参数["最优聚类数"]))
        self.assertEqual(len(self.工况), 最优K)
        self.assertEqual(len(self.标记), 10075)

    def test_能耗与Tobit模型(self):
        测试能耗 = self.能耗评价.loc[self.能耗评价["数据集"] == "测试集"].iloc[0]
        self.assertGreater(float(测试能耗["R2"]), 0.8)
        self.assertGreaterEqual(float(测试能耗["MAE_kW"]), 0)
        self.assertEqual(len(self.Tobit参数), 14)  # 13个回归系数（含常数项）与sigma
        物理项 = self.Tobit参数[self.Tobit参数["参数"].str.contains("电压平方/烟气流量", na=False)]
        self.assertEqual(len(物理项), 4)
        self.assertTrue((物理项["估计值"] >= -1e-12).all())
        self.assertTrue(bool(self.参数["左删失Tobit收敛"]))
        self.assertGreater(float(self.参数["左删失Tobit误差标准差"]), 0)
        self.assertTrue(np.isfinite(self.Tobit评价.select_dtypes(include=[np.number])).all().all())

    def test_遗传算法结果与约束(self):
        K = int(self.参数["最优聚类数"])
        self.assertEqual(len(self.优化), K)
        self.assertEqual(len(self.检查), K)
        for 列 in ["电压边界满足", "周期边界及整数满足", "电压分组约束满足", "周期分组约束满足"]:
            self.assertTrue((self.检查[列] == 1).all(), 列)
        计算排放达标 = (self.优化["优化预测出口浓度_mg_Nm3"] <= 10 + 1e-9).astype(int)
        np.testing.assert_array_equal(计算排放达标, self.优化["是否满足10mg_Nm3"])
        self.assertEqual(int(self.优化["是否满足10mg_Nm3"].sum()), int(self.参数["可行工况数"]))
        self.assertTrue((self.优化["边界内最低预测浓度_mg_Nm3"] > 0).all())
        self.assertTrue((self.优化["边界内最低预测浓度_mg_Nm3"]
                         <= self.优化["优化预测出口浓度_mg_Nm3"] + 1e-9).all())
        self.assertEqual(int(self.参数["可行工况数"]), int(self.参数["典型工况总数"]))
        self.assertTrue((self.优化["优化预测出口浓度_mg_Nm3"] <= 10 + 1e-8).all())
        self.assertTrue((self.优化["相对电压平方倍率"] > 1).all())
        self.assertEqual(float(self.参数["Deutsch电压指数"]), 2.0)

    def test_当前不可行结论如实记录(self):
        if int(self.参数["可行工况数"]) == 0:
            self.assertTrue((self.检查["边界内存在排放可行解"] == 0).all())
            self.assertTrue((self.优化["边界内最低预测浓度_mg_Nm3"] > 10).all())
            报告 = 报告路径.read_text(encoding="utf-8")
            self.assertIn("0/6", 报告)
            self.assertIn("边界内的最优候选", 报告)

    def test_输出文件完整(self):
        数据文件 = [
            "GMM模型选择.csv", "典型工况统计.csv", "典型工况标记.csv",
            "岭回归参数选择.csv", "能耗模型评价.csv", "能耗模型系数.csv",
            "左删失Tobit模型评价.csv", "左删失Tobit模型参数.csv",
            "决策变量边界.csv", "优化结果_Deutsch相对外推.csv", "遗传算法收敛历史.csv",
            "约束检查.csv", "问题二模型参数.json", "运行日志.txt",
        ]
        for 文件 in 数据文件:
            路径 = 结果目录 / 文件
            self.assertTrue(路径.exists() and 路径.stat().st_size > 0, 文件)
        for 名称 in ["问题二_典型工况划分", "问题二_模型预测效果", "问题二_遗传算法收敛", "问题二_优化前后对比"]:
            for 后缀 in [".png", ".pdf"]:
                路径 = 图片目录 / f"{名称}{后缀}"
                self.assertTrue(路径.exists() and 路径.stat().st_size > 0, str(路径))
        self.assertTrue(报告路径.exists() and 报告路径.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
