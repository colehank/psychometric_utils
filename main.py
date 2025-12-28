"""
Psychometric库快速开始示例
"""

import numpy as np
import pandas as pd
from psychometric import ItemAnalysis, Reliability, Validity


def quick_start_demo():
    """快速开始演示"""
    print("=" * 80)
    print("Psychometric库 - 快速开始演示")
    print("=" * 80)

    # 生成示例数据
    print("\n1. 生成示例问卷数据...")
    np.random.seed(42)
    n_samples = 150

    # 模拟一个2因子问卷（每个因子5题）
    factor1 = np.random.randn(n_samples)
    factor2 = np.random.randn(n_samples)

    data = pd.DataFrame()

    # Factor 1的题目
    for i in range(1, 6):
        loading = 0.7
        error = np.random.randn(n_samples) * 0.4
        score = factor1 * loading + error
        score = (score - score.min()) / (score.max() - score.min()) * 4 + 1
        data[f'Q{i}'] = np.round(score).clip(1, 5)

    # Factor 2的题目
    for i in range(6, 11):
        loading = 0.7
        error = np.random.randn(n_samples) * 0.4
        score = factor2 * loading + error
        score = (score - score.min()) / (score.max() - score.min()) * 4 + 1
        data[f'Q{i}'] = np.round(score).clip(1, 5)

    print(f"   生成了{len(data)}个样本，{len(data.columns)}个题目")
    print(f"\n数据预览：")
    print(data.head())

    # 项目分析
    print("\n2. 项目分析...")
    ia = ItemAnalysis(data)
    item_result = ia.analyze()
    print(f"   难易度分析：✓")
    print(f"   CITC分析：✓")
    print(f"   极端组检验：✓")

    # 信度分析
    print("\n3. 信度分析...")
    rel = Reliability(data)
    alpha = rel.cronbach_alpha()
    print(f"   Cronbach's α = {alpha['alpha']:.4f} ({alpha['quality']})")

    # 效度分析
    print("\n4. 效度分析...")
    val = Validity(data)
    try:
        efa_result = val.efa(n_factors=2, rotation='varimax')
        print(f"   KMO = {efa_result['kmo']['overall']:.4f}")
        print(f"   提取{efa_result['n_factors']}个因子")
        print(f"   累积方差解释 = {efa_result['variance'].iloc[2, -1]:.2%}")

        # AVE和CR
        factor_structure = efa_result['factor_structure']
        ave_cr = val.ave_cr(factor_structure)
        print(f"\n   AVE和CR结果：")
        for _, row in ave_cr.iterrows():
            print(f"   {row['factor']}: AVE={row['ave']:.3f}, CR={row['cr']:.3f}")

    except Exception as e:
        print(f"   效度分析错误: {e}")

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print("\n更多示例请查看 examples/usage_examples.py")
    print("运行测试套件：uv run tests/test_all.py")


if __name__ == "__main__":
    quick_start_demo()
