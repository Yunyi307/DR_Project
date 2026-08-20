import json
from pathlib import Path
import pandas as pd


def load_metric_from_candidates(model_dir, possible_files, metric_keys):
    """依次尝试读取候选 JSON 文件和候选字段名"""
    for fname in possible_files:
        json_path = model_dir / fname
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for k in metric_keys:
                    if k in data and data[k] is not None:
                        return float(data[k])
            except Exception as e:
                continue
    return None


def verify_and_generate_table4():
    # 设定各个模型可能的文件夹名称与 JSON 文件名备选列表
    models_config = [
        {
            'name': 'EffNet-CE',
            'dir_candidates': ['effnet_b4_ce'],
            'aptos_files': ['test_metrics.json'],
            'idrid_files': [
                'external_idrid_metrics.json',
                'external_idrid_calibrated_metrics.json',
            ],
        },
        {
            'name': 'EffNet-Focal',
            'dir_candidates': ['effnet_b4_focal'],
            'aptos_files': ['test_metrics.json'],
            'idrid_files': [
                'external_idrid_metrics.json',
                'external_idrid_calibrated_metrics.json',
            ],
        },
        {
            'name': 'ViT-B/16',
            'dir_candidates': [
                'vit_b16',
                'vit_b_16',
                'vit_base',
                'effnet_vit',
                'vit',
            ],
            'aptos_files': ['test_metrics.json'],
            'idrid_files': [
                'external_idrid_metrics.json',
                'external_idrid_calibrated_metrics.json',
            ],
        },
        {
            'name': 'Swin-B',
            'dir_candidates': [
                'swin_b',
                'swin_base',
                'swin',
                'effnet_swin',
            ],
            'aptos_files': ['test_metrics.json'],
            'idrid_files': [
                'external_idrid_metrics.json',
                'external_idrid_calibrated_metrics.json',
            ],
        },
        {
            'name': '5-Fold Ensemble (Ours)',
            'dir_candidates': ['effnet_b4_baseline_5fold', 'effnet_b4_5fold'],
            'aptos_files': ['test_metrics_5fold.json', 'test_metrics.json'],
            'idrid_files': [
                'external_idrid_5fold_calibrated_metrics.json',
                'external_idrid_metrics.json',
            ],
        },
        {
            'name': 'ViT-B/16',
            'dir_candidates': ['vit_b16_focal', 'vit_b16_ce', 'vit_b16', 'vit_base'],
            'aptos_files': ['test_metrics.json'],
            'idrid_files': ['external_idrid_metrics.json', 'external_idrid_calibrated_metrics.json']
        },
        {
            'name': 'Swin-B',
            'dir_candidates': ['swin_b_focal', 'swin_b_ce', 'swin_b', 'swin_base'],  # 👈 这里加入了真实的 swin_b_focal
            'aptos_files': ['test_metrics.json'],
            'idrid_files': ['external_idrid_metrics.json', 'external_idrid_calibrated_metrics.json']
        },
    ]

    base_outputs_dir = Path('../outputs')
    qwk_keys = ['qwk', 'quadratic_weighted_kappa', 'test_qwk']
    auc_keys = [
        'referable_auc',
        'ref_auc',
        'auc',
        'test_referable_auc',
        'test_auc',
    ]

    results = []

    print('🔍 开始全自动化扫描并提取 Table 4 指标...\n')

    for m in models_config:
        # 寻找实际存在的模型目录
        target_dir = None
        for d_name in m['dir_candidates']:
            candidate_dir = base_outputs_dir / d_name
            if candidate_dir.exists():
                target_dir = candidate_dir
                break

        if target_dir is None:
            results.append({
                'Model': m['name'],
                'APTOS QWK': 'N/A',
                'IDRiD QWK': 'N/A',
                'Gap': 'N/A',
                'APTOS Ref-AUC': 'N/A',
                'IDRiD Ref-AUC': 'N/A',
                'Status': '❌ 目录未匹配',
            })
            continue

        # 读取指标
        aptos_qwk = load_metric_from_candidates(
            target_dir, m['aptos_files'], qwk_keys
        )
        idrid_qwk = load_metric_from_candidates(
            target_dir, m['idrid_files'], qwk_keys
        )

        aptos_auc = load_metric_from_candidates(
            target_dir, m['aptos_files'], auc_keys
        )
        idrid_auc = load_metric_from_candidates(
            target_dir, m['idrid_files'], auc_keys
        )

        # 计算 Gap
        gap = (
            (idrid_qwk - aptos_qwk)
            if (idrid_qwk is not None and aptos_qwk is not None)
            else None
        )

        results.append({
            'Model': m['name'],
            'APTOS QWK': f'{aptos_qwk:.3f}' if aptos_qwk is not None else 'N/A',
            'IDRiD QWK': f'{idrid_qwk:.3f}' if idrid_qwk is not None else 'N/A',
            'Gap': f'{gap:+.3f}' if gap is not None else 'N/A',
            'APTOS Ref-AUC': f'{aptos_auc:.3f}' if aptos_auc is not None else 'N/A',
            'IDRiD Ref-AUC': f'{idrid_auc:.3f}' if idrid_auc is not None else 'N/A',
            'Status': f'✅ ({target_dir.name})'
        })
    # 打印终端结果表
    df = pd.DataFrame(results)
    print('=' * 85)
    print('📊 Table 4 最新精确提取结果汇总表：')
    print('=' * 85)
    print(df.to_string(index=False))
    print('=' * 85)

    print('\n📋 Markdown 格式 (复制更新至 Word 表格)：')
    print(df.drop(columns=['Status']).to_markdown(index=False))


if __name__ == '__main__':
    verify_and_generate_table4()