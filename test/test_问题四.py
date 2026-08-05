# -*- coding: utf-8 -*-
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


项目目录 = Path(__file__).resolve().parents[1]
结果目录 = 项目目录 / "code" / "outputs" / "问题四计算结果"
图片目录 = 项目目录 / "figures" / "问题四计算结果"


class 问题四结果测试(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.历史可行 = pd.read_csv(结果目录 / "5标准历史边界可行性检查.csv")
        cls.可行 = pd.read_csv(结果目录 / "5标准扩界可行性检查.csv")
        cls.对比 = pd.read_csv(结果目录 / "10与5标准电耗对比.csv")
        cls.极值 = pd.read_csv(结果目录 / "极值工况参数对比.csv")
        cls.边界 = pd.read_csv(结果目录 / "问题四扩展决策变量边界.csv")
        cls.参数 = json.loads((结果目录 / "问题四参数与汇总.json").read_text(encoding="utf-8"))

    def test_除尘指数增量(self):
        self.assertAlmostEqual(self.参数["除尘指数增量"], np.log(2), places=12)

    def test_可行性判定一致(self):
        expected = (self.可行["边界内最低预测浓度_mg_Nm3"] <= 5.0 + 1e-10).astype(int)
        self.assertTrue((expected == self.可行["5标准是否可行"]).all())
        self.assertTrue((self.可行["分组约束满足"] == 1).all())

    def test_历史边界与扩界结论(self):
        self.assertEqual(int(self.历史可行["5标准是否可行"].sum()), 4)
        self.assertEqual(set(self.历史可行.loc[self.历史可行["5标准是否可行"] == 0, "工况编号"]), {3, 4})
        self.assertEqual(int(self.可行["5标准是否可行"].sum()), 6)
        self.assertAlmostEqual(self.参数["电压上限扩展比例"], 0.04, places=12)

    def test_不报告不可行工况的电耗(self):
        self.assertTrue((self.对比["5标准是否可行"] == 1).all())
        self.assertTrue(self.对比["5标准预测电耗_kW"].notna().all())

    def test_可行解满足5标准(self):
        good = self.对比[self.对比["5标准是否可行"] == 1]
        self.assertTrue((good["5标准预测出口浓度_mg_Nm3"] <= 5.0 + 1e-8).all())
        self.assertTrue((good["电压分组违反"].abs() <= 1e-8).all())
        self.assertTrue((good["周期分组违反"].abs() <= 1e-8).all())

    def test_5标准策略不越界(self):
        good = self.对比[self.对比["5标准是否可行"] == 1]
        bounds = {r["变量"]: (float(r["下限"]), float(r["上限"])) for _, r in self.边界.iterrows()}
        for i in range(1, 5):
            lo, hi = bounds[f"U{i}"]
            self.assertTrue(good[f"5标准U{i}_kV"].between(lo-1e-10, hi+1e-10).all())
            lo, hi = bounds[f"tau{i}"]
            self.assertTrue(good[f"5标准tau{i}_s"].between(lo, hi).all())
            self.assertTrue(np.allclose(good[f"5标准tau{i}_s"], np.rint(good[f"5标准tau{i}_s"])))

    def test_极值工况定义正确(self):
        self.assertEqual(set(self.极值["典型工况"]), {"工况5", "工况6"})
        self.assertEqual(set(self.极值["工况类型"]), {"高入口浓度", "高负荷重载"})

    def test_输出存在且非空(self):
        files = [
            结果目录 / "5标准可行性检查.csv", 结果目录 / "10与5标准电耗对比.csv",
            结果目录 / "5标准历史边界可行性检查.csv", 结果目录 / "5标准扩界可行性检查.csv",
            结果目录 / "极值工况参数对比.csv", 结果目录 / "遗传算法收敛历史.csv",
            结果目录 / "沿用的决策变量边界.csv", 结果目录 / "问题四扩展决策变量边界.csv",
            结果目录 / "问题四参数与汇总.json",
            结果目录 / "运行日志.txt", 项目目录 / "reports" / "问题四计算结果.md",
            图片目录 / "问题四_各工况5标准可行性.png", 图片目录 / "问题四_各工况5标准可行性.pdf",
            图片目录 / "问题四_可行工况电耗增幅.png", 图片目录 / "问题四_可行工况电耗增幅.pdf",
            图片目录 / "问题四_极值工况参数对比.png", 图片目录 / "问题四_极值工况参数对比.pdf",
        ]
        for path in files:
            self.assertTrue(path.exists(), str(path))
            self.assertGreater(path.stat().st_size, 0, str(path))


if __name__ == "__main__":
    unittest.main()
