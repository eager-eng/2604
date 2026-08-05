# -*- coding: utf-8 -*-
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


项目目录 = Path(__file__).resolve().parents[1]
结果目录 = 项目目录 / "code" / "outputs" / "问题三计算结果"
图片目录 = 项目目录 / "figures" / "问题三计算结果"


class 问题三结果测试(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.工况 = pd.read_csv(结果目录 / "对比工况特征.csv")
        cls.操作 = pd.read_csv(结果目录 / "最优操作参数对比.csv")
        cls.扰动 = pd.read_csv(结果目录 / "局部双向扰动明细.csv")
        cls.汇总 = pd.read_csv(结果目录 / "边际效益汇总.csv")

    def test_对比工况为负荷极值(self):
        self.assertEqual(set(self.工况["工况编号"]), {4, 5})
        self.assertLess(self.工况.loc[self.工况["工况编号"] == 4, "入口负荷中心_kg_h"].iloc[0],
                        self.工况.loc[self.工况["工况编号"] == 5, "入口负荷中心_kg_h"].iloc[0])

    def test_最优策略满足排放限值(self):
        self.assertTrue((self.操作["预测出口浓度_mg_Nm3"] <= 10.0 + 1e-8).all())

    def test_双向扰动完整(self):
        self.assertEqual(len(self.扰动), 32)
        counts = self.扰动.groupby(["论文工况", "变量"])["扰动方向"].nunique()
        self.assertTrue((counts == 2).all())

    def test_基准数值与问题二一致(self):
        q2 = pd.read_csv(项目目录 / "code" / "outputs" / "问题二计算结果" / "优化结果_Deutsch相对外推.csv")
        for k in (4, 5):
            p2 = float(q2.loc[q2["典型工况"] == f"工况{k}", "优化预测电耗_kW"].iloc[0])
            p3 = float(self.操作.loc[self.操作["原始工况"] == f"工况{k}", "预测总电耗_kW"].iloc[0])
            self.assertAlmostEqual(p2, p3, places=8)

    def test_边际效益非负(self):
        for col in ["减排边际效益_mg_Nm3每kW", "节能边际效益_kW每mg_Nm3"]:
            values = self.汇总[col].dropna().to_numpy(dtype=float)
            self.assertTrue(np.all(values >= 0))

    def test_双赢方向为两工况的tau3延长(self):
        win = self.扰动[self.扰动["是否节能减排双赢"] == 1]
        self.assertEqual(len(win), 2)
        self.assertEqual(set(win["变量"]), {"tau3"})
        self.assertEqual(set(win["扰动方向"]), {"+"})
        self.assertTrue((win["排放约束满足"] == 1).all())

    def test_产物存在且非空(self):
        files = [
            结果目录 / "对比工况特征.csv", 结果目录 / "最优操作参数对比.csv",
            结果目录 / "局部双向扰动明细.csv", 结果目录 / "边际效益汇总.csv",
            结果目录 / "运行日志.txt", 项目目录 / "reports" / "问题三计算结果.md",
            图片目录 / "问题三_最优电压与振打周期对比.png",
            图片目录 / "问题三_最优电压与振打周期对比.pdf",
            图片目录 / "问题三_局部扰动边际效益热力图.png",
            图片目录 / "问题三_局部扰动边际效益热力图.pdf",
        ]
        for path in files:
            self.assertTrue(path.exists(), str(path))
            self.assertGreater(path.stat().st_size, 0, str(path))


if __name__ == "__main__":
    unittest.main()
