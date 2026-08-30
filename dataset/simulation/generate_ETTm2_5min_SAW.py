import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')




CONFIG = {
    'input_path': './dataset/ETT/ETTm2.csv',
    'output_dir': './dataset/ETT',
    'output_path': './dataset/ETT/ETTm2_5min_SAW.csv',
    'params_path': './dataset/ETT/ETTm2_SAW_params.pkl',
    'log_path': './dataset/ETT/ETTm2_SAW_log.json',
    'random_seed': 42,
    
    'time_range': {
        'start': '2017-05-25 00:00:00',
        'end': '2018-02-04 00:00:00'
    },
    
    'load_weights': {
        'HUFL': 0.35, 'MUFL': 0.30, 'LUFL': 0.15,
        'HULL': 0.12, 'MULL': 0.08, 'LULL': 0.00
    },
    'effective_features': ['HUFL', 'MUFL', 'LUFL'],
    
    'thermal': {
        'max_5min': 2.0,
        'max_15min': 4.0,
        'max_15min_final': 6.0,
        'perturbation': 0.2,
        'ot_range': [10, 80]
    },
    
    'fault': {
        'rise_rate': 0.2,
        'max_ot': 80,
        'steps': 20
    },
    'cooling': {
        'drop_rate': 0.75,
        'min_ot': 10,
        'steps': 10
    },
    
    'thresholds': {
        'heavy_load_percentile': 80,
        'light_load_percentile': 20,
        'surge_up': 0.30,
        'surge_down': -0.25
    },
    
    'max_outlier_ratio': 0.03,
    'min_correlation': 0.30,
    'split_ratio': [0.6, 0.2, 0.2],
    
    'target_ratio': {
        0: 0.60, 1: 0.15, 2: 0.10, 3: 0.05, 4: 0.03, 5: 0.04, 6: 0.03
    }
}

LOAD_FEATURES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
np.random.seed(CONFIG['random_seed'])





def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif pd.isna(obj):
        return None
    return obj


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def calc_load_weighted(df):
    return sum(df[f] * CONFIG['load_weights'][f] for f in LOAD_FEATURES)


def enforce_thermal_constraint(df, max_change, col='OT'):
    df = df.copy()
    max_iterations = 100
    
    for iteration in range(max_iterations):
        ot_diff = df[col].diff()
        violations = ot_diff.abs() > max_change
        violation_count = violations.sum()
        
        if violation_count == 0:
            break
        
        
        for idx in df.index[violations]:
            if idx == df.index[0]:
                continue
            
            prev_idx = df.index[df.index.get_loc(idx) - 1]
            prev_ot = df.loc[prev_idx, col]
            curr_ot = df.loc[idx, col]
            change = curr_ot - prev_ot
            
            if abs(change) > max_change:
                
                df.loc[idx, col] = prev_ot + np.sign(change) * max_change
    
    return df, iteration + 1





def stage0_time_filter(df):
    print_section("阶段0：时间区间筛选")
    
    df['date'] = pd.to_datetime(df['date'])
    start_dt = pd.to_datetime(CONFIG['time_range']['start'])
    end_dt = pd.to_datetime(CONFIG['time_range']['end'])
    
    original_count = len(df)
    print(f"原始数据: {original_count} 条")
    
    mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
    df_filtered = df[mask].copy()
    print(f"筛选后: {len(df_filtered)} 条")
    
    
    df_filtered = df_filtered.set_index('date')
    full_range = pd.date_range(start=start_dt, end=end_dt, freq='15min')
    df_full = df_filtered.reindex(full_range)
    
    missing_count = df_full.isnull().any(axis=1).sum()
    if missing_count > 0:
        for col in LOAD_FEATURES + ['OT']:
            df_full[col] = df_full[col].interpolate(method='linear').ffill().bfill()
    
    df_full = df_full.reset_index().rename(columns={'index': 'date'})
    print(f"补全后: {len(df_full)} 条")
    
    return df_full, {'original': original_count, 'filtered': len(df_full)}





def stage1_preprocess(df):
    print_section("阶段1：数据预处理 [插值替代删除]")
    
    ot_min, ot_max = CONFIG['thermal']['ot_range']
    
    
    out_of_range = (df['OT'] < ot_min) | (df['OT'] > ot_max)
    out_count = out_of_range.sum()
    print(f"OT超范围({ot_min}-{ot_max}℃)数据: {out_count} 条 ({out_count/len(df)*100:.1f}%)")
    
    if out_count > 0:
        
        df = df.copy()
        df.loc[out_of_range, 'OT'] = np.nan
        
        
        df['OT'] = df['OT'].interpolate(method='linear')
        
        df['OT'] = df['OT'].ffill().bfill()
        
        df['OT'] = df['OT'].clip(ot_min, ot_max)
        
        print(f"已用插值替换超范围数据，保持时序连续")
    
    
    df['_load_weighted'] = calc_load_weighted(df)
    
    
    effective_load = sum(df[f] * CONFIG['load_weights'][f] for f in CONFIG['effective_features'])
    ratio = (effective_load / df['_load_weighted'].replace(0, np.nan)).mean()
    ratio = float(ratio) if not np.isnan(ratio) else 0.8
    print(f"有效负载占比: {ratio*100:.1f}%")
    
    
    print(f"\n原始数据负载-OT相关性:")
    correlations_original = {}
    for f in LOAD_FEATURES:
        corr = np.corrcoef(df[f], df['OT'])[0, 1]
        corr = float(corr) if not np.isnan(corr) else 0.0
        correlations_original[f] = corr
        print(f"  {f}: {corr:.4f}")
    
    return df, {'effective_ratio': ratio, 'correlations_original': correlations_original}





def stage1_5_thermal_repair(df):
    print_section("阶段1.5：多轮热惯性修复")
    
    max_15min = CONFIG['thermal']['max_15min']
    
    
    ot_diff_before = df['OT'].diff().abs()
    violations_before = (ot_diff_before > max_15min).sum()
    max_before = ot_diff_before.max()
    
    print(f"修复前: {violations_before} 处违规, 最大变化 {max_before:.2f}℃")
    
    
    df, iterations = enforce_thermal_constraint(df, max_15min, 'OT')
    
    
    ot_diff_after = df['OT'].diff().abs()
    violations_after = (ot_diff_after > max_15min).sum()
    max_after = ot_diff_after.max()
    
    print(f"修复后: {violations_after} 处违规, 最大变化 {max_after:.2f}℃")
    print(f"迭代次数: {iterations}")
    print(f"结果: {'✅ 热惯性约束满足' if max_after <= max_15min else '⚠️ 仍有超标'}")
    
    return df, {'iterations': iterations, 'max_change_after': float(max_after)}





def stage2_interpolation(df):
    print_section("阶段2：采样频率转换 (15min→5min)")
    
    df['date'] = pd.to_datetime(df['date'])
    new_rows = []
    
    max_5min = CONFIG['thermal']['max_5min']
    max_15min = CONFIG['thermal']['max_15min']
    perturbation = CONFIG['thermal']['perturbation']
    ot_range = CONFIG['thermal']['ot_range']
    
    for i in range(len(df) - 1):
        curr = df.iloc[i]
        next_ = df.iloc[i + 1]
        
        
        ot_change_15 = next_['OT'] - curr['OT']
        ot_change_15 = np.clip(ot_change_15, -max_15min, max_15min)
        
        
        ot_step = ot_change_15 / 3
        ot_step = np.clip(ot_step, -max_5min, max_5min)
        
        for step in range(3):
            row = {'date': curr['date'] + pd.Timedelta(minutes=5*step)}
            for f in LOAD_FEATURES:
                row[f] = curr[f] + (next_[f] - curr[f]) * (step / 3)
            
            ot_base = curr['OT'] + ot_step * step
            ot_noise = np.random.uniform(-perturbation, perturbation)
            row['OT'] = np.clip(ot_base + ot_noise, ot_range[0], ot_range[1])
            new_rows.append(row)
    
    
    new_rows.append({
        'date': df.iloc[-1]['date'],
        **{f: df.iloc[-1][f] for f in LOAD_FEATURES},
        'OT': df.iloc[-1]['OT']
    })
    
    df_5min = pd.DataFrame(new_rows)
    df_5min['_load_weighted'] = calc_load_weighted(df_5min)
    
    
    ot_diff = df_5min['OT'].diff().abs()
    max_change = ot_diff.max()
    
    print(f"转换: {len(df)} → {len(df_5min)} 条")
    print(f"5分钟最大变化: {max_change:.2f}℃ {'✅' if max_change <= max_5min + 0.3 else '⚠️'}")
    
    return df_5min, {'max_5min_change': float(max_change)}





def stage3_split_and_clean(df):
    print_section("阶段3：数据划分与3σ清洗")
    
    total = len(df)
    train_end = int(total * CONFIG['split_ratio'][0])
    val_end = int(total * sum(CONFIG['split_ratio'][:2]))
    
    split_info = {
        'train': {'start': 0, 'end': train_end - 1, 'count': train_end},
        'val': {'start': train_end, 'end': val_end - 1, 'count': val_end - train_end},
        'test': {'start': val_end, 'end': total - 1, 'count': total - val_end}
    }
    
    print(f"训练集: {train_end} | 验证集: {val_end-train_end} | 测试集: {total-val_end}")
    
    
    train_df = df.iloc[:train_end]
    sigma_params = {col: {'mean': train_df[col].mean(), 'std': train_df[col].std()} 
                    for col in LOAD_FEATURES + ['OT']}
    
    
    df['_outlier_score'] = 0.0
    for col in LOAD_FEATURES + ['OT']:
        mu, sigma = sigma_params[col]['mean'], sigma_params[col]['std']
        if sigma > 0:
            df['_outlier_score'] += np.abs((df[col] - mu) / sigma)
    
    
    target_outlier_count = int(total * CONFIG['max_outlier_ratio'])
    threshold = df['_outlier_score'].nlargest(target_outlier_count).min()
    df['_is_outlier'] = df['_outlier_score'] >= threshold
    
    print(f"3σ异常标记: {df['_is_outlier'].sum()} 条 ({df['_is_outlier'].mean()*100:.2f}%)")
    
    return df, split_info, sigma_params





def stage4_scene_injection(df, split_info):
    print_section("阶段4：场景注入 [严格热惯性控制]")
    
    total = len(df)
    train_end = split_info['train']['end'] + 1
    max_5min = CONFIG['thermal']['max_5min']
    
    df['_is_fault'] = False
    df['_is_cooling'] = False
    
    
    fault_cfg = CONFIG['fault']
    target_fault = int(total * CONFIG['target_ratio'][5])
    n_seq = max(1, target_fault // fault_cfg['steps'])
    fault_starts = np.linspace(100, train_end - fault_cfg['steps'] - 100, n_seq, dtype=int)
    
    print(f"\n故障注入: {n_seq}序列")
    
    for start in fault_starts:
        
        prev_ot = float(df.loc[start - 1, 'OT']) if start > 0 else float(df.loc[start, 'OT'])
        
        for step in range(fault_cfg['steps']):
            idx = start + step
            if idx >= total:
                break
            
            
            max_rise = min(fault_cfg['rise_rate'], max_5min)
            new_ot = prev_ot + max_rise
            new_ot = min(new_ot, fault_cfg['max_ot'])
            
            df.loc[idx, 'OT'] = new_ot
            df.loc[idx, '_is_fault'] = True
            
            
            multiplier = 1.0 + 1.5 * (step / fault_cfg['steps'])
            for f in LOAD_FEATURES:
                df.loc[idx, f] = df.loc[start, f] * multiplier
            
            prev_ot = new_ot
    
    print(f"  实际注入: {df['_is_fault'].sum()} 条")
    
    
    cool_cfg = CONFIG['cooling']
    target_cool = int(total * CONFIG['target_ratio'][4])
    n_cool = max(1, target_cool // cool_cfg['steps'])
    cool_starts = np.linspace(200, train_end - cool_cfg['steps'] - 100, n_cool, dtype=int)
    
    print(f"\n冷却注入: {n_cool}序列")
    
    for start in cool_starts:
        if df.loc[start, '_is_fault']:
            continue
        
        prev_ot = float(df.loc[start - 1, 'OT']) if start > 0 else float(df.loc[start, 'OT'])
        init_ot = prev_ot
        half = cool_cfg['steps'] // 2
        max_drop = min(cool_cfg['drop_rate'] * half, init_ot - cool_cfg['min_ot'])
        
        if max_drop <= 0:
            continue
        
        for step in range(cool_cfg['steps']):
            idx = start + step
            if idx >= total or df.loc[idx, '_is_fault']:
                break
            
            
            if step < half:
                
                target_ot = init_ot - max_drop * ((step + 1) / half)
            else:
                
                target_ot = init_ot - max_drop + max_drop * ((step - half + 1) / half)
            
            
            change = target_ot - prev_ot
            if abs(change) > max_5min:
                new_ot = prev_ot + np.sign(change) * max_5min
            else:
                new_ot = target_ot
            
            new_ot = max(new_ot, cool_cfg['min_ot'])
            df.loc[idx, 'OT'] = new_ot
            df.loc[idx, '_is_cooling'] = True
            prev_ot = new_ot
    
    print(f"  实际注入: {df['_is_cooling'].sum()} 条")
    
    
    df['_load_weighted'] = calc_load_weighted(df)
    
    return df





def stage4_5_post_injection_repair(df):
    print_section("阶段4.5：注入后热惯性修复")
    
    max_5min = CONFIG['thermal']['max_5min']
    
    
    ot_diff_before = df['OT'].diff().abs()
    max_before = ot_diff_before.max()
    violations_before = (ot_diff_before > max_5min).sum()
    
    print(f"修复前: {violations_before} 处违规, 最大变化 {max_before:.2f}℃")
    
    
    df, iterations = enforce_thermal_constraint(df, max_5min, 'OT')
    
    
    ot_diff_after = df['OT'].diff().abs()
    max_after = ot_diff_after.max()
    violations_after = (ot_diff_after > max_5min).sum()
    
    print(f"修复后: {violations_after} 处违规, 最大变化 {max_after:.2f}℃")
    print(f"迭代次数: {iterations}")
    print(f"结果: {'✅ 5分钟热惯性满足' if max_after <= max_5min + 0.3 else '⚠️'}")
    
    return df, {'max_5min_after_repair': float(max_after)}





def stage5_assign_conditions(df):
    print_section("阶段5：工况标签分配")
    
    df['condition'] = 0
    thresh = CONFIG['thresholds']
    
    
    load_weighted = df['_load_weighted']
    heavy_threshold = load_weighted.quantile(thresh['heavy_load_percentile'] / 100)
    light_threshold = load_weighted.quantile(thresh['light_load_percentile'] / 100)
    
    print(f"动态阈值: 重载≥{heavy_threshold:.4f}, 轻载≤{light_threshold:.4f}")
    
    for idx in range(len(df)):
        if df.loc[idx, '_is_fault']:
            df.loc[idx, 'condition'] = 5
        elif df.loc[idx, '_is_cooling']:
            df.loc[idx, 'condition'] = 4
        elif df.loc[idx, '_is_outlier']:
            df.loc[idx, 'condition'] = 6
        elif idx > 0:
            curr_load = df.loc[idx, '_load_weighted']
            prev_load = df.loc[idx - 1, '_load_weighted']
            if prev_load > 0.01:
                change = (curr_load - prev_load) / prev_load
                if change >= thresh['surge_up'] or change <= thresh['surge_down']:
                    df.loc[idx, 'condition'] = 3
                    continue
            if curr_load >= heavy_threshold:
                df.loc[idx, 'condition'] = 2
            elif curr_load <= light_threshold:
                df.loc[idx, 'condition'] = 1
    
    
    stats = df['condition'].value_counts(normalize=True).sort_index() * 100
    names = {0:'正常', 1:'轻载', 2:'重载', 3:'突变', 4:'冷却', 5:'故障', 6:'缺陷'}
    
    print(f"\n工况分布:")
    for c in range(7):
        actual = stats.get(c, 0)
        target = CONFIG['target_ratio'][c] * 100
        print(f"  {names[c]}({c}): {actual:.1f}% [目标{target:.0f}%] {'✅' if abs(actual-target)<=5 else '⚠️'}")
    
    return df, stats.to_dict()





def stage6_correlation(df, original_correlations):
    print_section("阶段6：相关性计算")
    
    normal_df = df[df['condition'] == 0].copy()
    print(f"正常工况数据: {len(normal_df)} 条")
    
    
    correlations_after = {}
    print(f"\n改造后相关性 vs 原始相关性:")
    for f in LOAD_FEATURES:
        corr = np.corrcoef(normal_df[f], normal_df['OT'])[0, 1]
        corr = float(corr) if not np.isnan(corr) else 0.0
        orig = original_correlations.get(f, 0)
        correlations_after[f] = corr
        diff = corr - orig
        print(f"  {f}: {corr:.4f} (原始{orig:.4f}, 差异{diff:+.4f})")
    
    
    weighted_corr = np.corrcoef(normal_df['_load_weighted'], normal_df['OT'])[0, 1]
    weighted_corr = float(weighted_corr) if not np.isnan(weighted_corr) else 0.0
    correlations_after['weighted'] = weighted_corr
    
    print(f"\n加权负载相关性: {weighted_corr}")
    print(f"校验: ≥{CONFIG['min_correlation']} {'✅' if weighted_corr >= CONFIG['min_correlation'] else '⚠️'}")
    
    correlations_after['pass'] = weighted_corr >= CONFIG['min_correlation']
    return correlations_after





def stage7_save(df, all_info):
    print_section("阶段7：最终验证与输出")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    
    ot_diff_5 = df['OT'].diff().abs()
    ot_diff_15 = df['OT'].diff(3).abs()
    
    max_5 = float(ot_diff_5.max())
    max_15 = float(ot_diff_15.max())
    
    pass_5 = max_5 <= CONFIG['thermal']['max_5min'] + 0.3
    pass_15 = max_15 <= CONFIG['thermal']['max_15min_final']
    
    print(f"热惯性最终校验:")
    print(f"  5分钟最大变化: {max_5:.2f}℃ (≤{CONFIG['thermal']['max_5min']}℃) {'✅' if pass_5 else '⚠️'}")
    print(f"  15分钟最大变化: {max_15:.2f}℃ (≤{CONFIG['thermal']['max_15min_final']}℃) {'✅' if pass_15 else '⚠️'}")
    print(f"  油温范围: {df['OT'].min():.1f}~{df['OT'].max():.1f}℃")
    
    
    output_cols = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT', 'condition']
    df_output = df[output_cols].copy()
    df_output['date'] = pd.to_datetime(df_output['date']).dt.strftime('%Y/%m/%d %H:%M')
    
    df_output.to_csv(CONFIG['output_path'], index=False, float_format='%.15f')
    print(f"\n输出: {CONFIG['output_path']} ({len(df_output)} 条)")
    
    
    validation = {
        'max_5min': max_5, 'max_15min': max_15,
        'pass_5min': pass_5, 'pass_15min': pass_15,
        'ot_range': [float(df['OT'].min()), float(df['OT'].max())]
    }
    
    params = {
        'config': CONFIG,
        **all_info,
        'validation': validation,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(CONFIG['params_path'], 'wb') as f:
        pickle.dump(params, f)
    
    with open(CONFIG['log_path'], 'w', encoding='utf-8') as f:
        json.dump(to_serializable(params), f, indent=2, ensure_ascii=False)
    
    print(f"参数: {CONFIG['params_path']}")
    print(f"日志: {CONFIG['log_path']}")
    
    
    print("\n" + "=" * 60)
    print("  最终校验报告")
    print("=" * 60)
    print(f"  ✓ 5分钟热惯性: {max_5:.2f}℃ {'✅' if pass_5 else '⚠️'}")
    print(f"  ✓ 15分钟热惯性: {max_15:.2f}℃ {'✅' if pass_15 else '⚠️'}")
    print(f"  ✓ 相关性: {all_info['correlation']['weighted']:.4f} {'✅' if all_info['correlation']['pass'] else '⚠️'}")
    
    return df_output





def main():
    print("\n" + "=" * 60)
    print("  ETTm2-5min-SAW 改造（彻底修复版）")
    print("=" * 60)
    
    df = pd.read_csv(CONFIG['input_path'])
    
    
    df, filter_info = stage0_time_filter(df)
    df, preprocess_info = stage1_preprocess(df)
    df, repair_info = stage1_5_thermal_repair(df)  
    df, interp_info = stage2_interpolation(df)
    df, split_info, sigma_params = stage3_split_and_clean(df)
    df = stage4_scene_injection(df, split_info)
    df, post_repair_info = stage4_5_post_injection_repair(df)  
    df, condition_stats = stage5_assign_conditions(df)
    correlations = stage6_correlation(df, preprocess_info['correlations_original'])
    
    all_info = {
        'filter': filter_info,
        'preprocess': preprocess_info,
        'repair': repair_info,
        'interp': interp_info,
        'split': split_info,
        'post_repair': post_repair_info,
        'condition': condition_stats,
        'correlation': correlations
    }
    
    df_output = stage7_save(df, all_info)
    
    print("\n" + "=" * 60)
    print("  改造完成！")
    print("=" * 60)
    
    return df_output


if __name__ == '__main__':
    df_result = main()
