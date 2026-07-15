#!/usr/bin/env python3
"""
生成 batch_output 文件夹中的文件列表，并校验文件完整性。
"""
import json
from pathlib import Path
from collections import defaultdict


def analyze_and_generate_file_list(output_dir='data_generation/batch_output'):
    """
    扫描目录，校验文件完整性，并生成用于UI的文件列表。
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"错误：目录 {output_dir} 不存在")
        return [], {}

    # --- 1. 收集和分组所有任务文件 ---
    # 实际文件格式:
    #   - task_{id}_{domain}_{timestamp}.json
    #   - task_{id}_{domain}_interactive_{timestamp}.html
    #   - task_{id}_{domain}_{timestamp}.pkl
    tasks = defaultdict(lambda: {'json': None, 'html': None, 'pkl': None})
    other_files = []
    
    for file in output_path.iterdir():
        if not file.is_file():
            continue

        if file.name.startswith('task_'):
            suffix = file.suffix
            
            if suffix == '.html' and '_interactive_' in file.name:
                # HTML 文件: task_{id}_{domain}_interactive_{timestamp}.html
                # 提取基准名: 移除 _interactive 部分
                base_name = file.name.replace('_interactive_', '_').rsplit('.', 1)[0]
                tasks[base_name]['html'] = file.name
            elif suffix == '.json':
                # JSON 文件: task_{id}_{domain}_{timestamp}.json
                base_name = file.name.rsplit('.', 1)[0]
                tasks[base_name]['json'] = file.name
            elif suffix == '.pkl':
                # PKL 文件: task_{id}_{domain}_{timestamp}.pkl
                base_name = file.name.rsplit('.', 1)[0]
                tasks[base_name]['pkl'] = file.name
        elif file.suffix in ['.json', '.html']:
            other_files.append(file.name)

    # --- 2. 生成UI所需的文件列表 (json + html) ---
    ui_file_list = other_files
    for task_name in sorted(tasks.keys()):
        if tasks[task_name]['json']:
            ui_file_list.append(tasks[task_name]['json'])
        if tasks[task_name]['html']:
            ui_file_list.append(tasks[task_name]['html'])
    
    ui_file_list.sort()
    
    return ui_file_list, tasks


def print_validation_report(tasks):
    """
    打印文件完整性校验报告。
    """
    if not tasks:
        print("⚠️  目录中未找到任何 'task_*' 相关文件进行校验。")
        return

    json_count = sum(1 for v in tasks.values() if v['json'] is not None)
    html_count = sum(1 for v in tasks.values() if v['html'] is not None)
    pkl_count = sum(1 for v in tasks.values() if v['pkl'] is not None)
    
    print("--- 文件完整性校验报告 ---")
    print(f"📊 文件总数统计 (基于 {len(tasks)} 个独特的任务ID):")
    print(f"  - JSON 文件: {json_count}")
    print(f"  - HTML 文件: {html_count}")
    print(f"  - PKL 文件: {pkl_count}")

    if json_count == html_count == pkl_count:
        print("\n✅ 所有任务的文件数量一致。")
    else:
        print("\n⚠️  注意：各类文件数量不一致，可能存在缺失。")

    incomplete_tasks = {}
    for task_name, files in sorted(tasks.items()):
        if not (files['json'] and files['html'] and files['pkl']):
            missing = []
            if not files['json']: missing.append('json')
            if not files['html']: missing.append('html')
            if not files['pkl']: missing.append('pkl')
            incomplete_tasks[task_name] = missing
            
    if incomplete_tasks:
        print(f"\n❌ 找到 {len(incomplete_tasks)} 个文件不完整的任务:")
        for task_name, missing_files in incomplete_tasks.items():
            print(f"  - 任务: {task_name}, 缺失: {', '.join(missing_files)}")
    else:
        print("\n🎉 所有任务的文件都是完整的！")


def save_file_list(file_list, output_dir='data_generation', output_file='list_files.json'):
    """
    保存文件列表到 JSON 文件 (用于UI)。
    """
    output_path = Path(output_dir) / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(file_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 已为UI生成文件列表: {output_path} (包含 {len(file_list)} 个条目)")


def main():
    ui_file_list, tasks_data = analyze_and_generate_file_list()
    
    if tasks_data:
        print_validation_report(tasks_data)
    
    if ui_file_list:
        save_file_list(ui_file_list)
        
        print("\n📝 UI文件列表示例（前10个）:")
        for file in ui_file_list[:10]:
            print(f"  - {file}")
        
        if len(ui_file_list) > 10:
            print(f"  ... 还有 {len(ui_file_list) - 10} 个文件")
    else:
        print("\n⚠️  未找到任何 .json 或 .html 文件用于生成UI列表。")


if __name__ == '__main__':
    main()