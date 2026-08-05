# -*- coding: utf-8 -*-
"""问题一输出的可复现性与关键约束测试。"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


项目目录 = Path(__file__).resolve().parents[1]
结果目录 = 项目目录 / "code" / "outputs" / "问题一计算结果"
图片目录 = 项目目录 / "figures" / "问题一计算结果"


class 问题一测试(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.滞后 = pd.read_csv(结果目录 / "滞后选择结果.csv")
        cls.参数 = pd.read_csv(结果目录 / "Tobit参数估计.csv")
        cls.指标 = pd.read_csv(结果目录 / "模型评价指标.csv")
        cls.峰值 = pd.read_csv(结果目录 / "潜在浓度与峰值序列.csv")
        cls.电场 = pd.read_csv(结果目录 / "四电场振打影响统计.csv")
        cls.模型参数 = json.loads((结果目录 / "模型参数.json").read_text(encoding="utf-8"))

    def test_候选滞后与收敛(self):
        self.assertEqual(self.滞后["滞后分钟"].tolist(), [0, 1, 3, 5])
        self.assertTrue((self.滞后["是否收敛"] == 1).all())
        self.assertEqual(int(self.滞后["是否最优"].sum()), 1)
        计算最优 = int(self.滞后.loc[self.滞后["验证集平均负对数似然"].idxmin(), "滞后分钟"])
        self.assertEqual(计算最优, int(self.模型参数["最优滞后_分钟"]))

    def test_模型规模与评价指标(self):
        self.assertEqual(len(self.参数), 18)  # 17个回归系数（含常数项）与sigma
        self.assertTrue(np.isfinite(self.参数.loc[self.参数["参数"] != "误差标准差σ", "估计值"]).all())
        指标 = dict(zip(self.指标["指标"], self.指标["数值"]))
        self.assertEqual(int(指标["测试集样本数"]), 1440)
        self.assertGreater(指标["测试集未删失样本数"], 0)
        self.assertTrue(np.isfinite(指标["未删失样本MAE_mg_Nm3"]))
        self.assertGreaterEqual(指标["未删失样本MAE_mg_Nm3"], 0)

    def test_潜在浓度重构规则(self):
        self.assertEqual(len(self.峰值), 10075)
        删失 = self.峰值["右删失标记"] == 1
        未删失 = ~删失
        self.assertTrue((self.峰值.loc[删失, "潜在浓度估计_mg_Nm3"] >= 50).all())
        np.testing.assert_allclose(
            self.峰值.loc[未删失, "潜在浓度估计_mg_Nm3"],
            self.峰值.loc[未删失, "出口粉尘浓度_mg_Nm3"],
            atol=1e-10,
        )
        插值峰 = self.峰值.loc[self.峰值["出口浓度原始缺失标记"] == 1, "峰值标记"]
        self.assertTrue((插值峰 == 0).all())
        self.assertEqual(int(self.峰值["峰值标记"].sum()), int(self.模型参数["峰值数量"]))

    def test_四电场分组统计(self):
        self.assertEqual(len(self.电场), 16)
        self.assertEqual(set(self.电场["电场"]), {"电场1", "电场2", "电场3", "电场4"})
        self.assertEqual(set(self.电场["振打频率组"]), {"低", "中低", "中高", "高"})
        self.assertTrue(self.电场["峰值发生概率"].between(0, 1).all())
        self.assertTrue((self.电场["等效振打周期_s"] > 0).all())

    def test_输出文件完整(self):
        数据文件 = [
            "滞后选择结果.csv", "Tobit参数估计.csv", "模型评价指标.csv",
            "潜在浓度与峰值序列.csv", "四电场振打影响统计.csv", "模型参数.json", "运行日志.txt",
        ]
        for 文件 in 数据文件:
            路径 = 结果目录 / 文件
            self.assertTrue(路径.exists() and 路径.stat().st_size > 0, 文件)
        for 名称 in ["问题一_滞后选择与模型检验", "问题一_参数影响与显著性", "问题一_潜在浓度重构与峰值识别", "问题一_四电场振打影响"]:
            for 后缀 in [".png", ".pdf"]:
                路径 = 图片目录 / f"{名称}{后缀}"
                self.assertTrue(路径.exists() and 路径.stat().st_size > 0, str(路径))
        报告 = 项目目录 / "reports" / "问题一计算结果.md"
        self.assertTrue(报告.exists() and 报告.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
