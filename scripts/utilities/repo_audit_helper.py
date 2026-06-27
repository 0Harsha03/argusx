import os
import json
from pathlib import Path
import datetime

ROOT = Path('.').resolve()
ignore_dirs = {'.git'}

data = {
    'tree': [],
    'stats': {
        'total_folders': 0,
        'total_files': 0,
        'total_size_bytes': 0,
        'ext_counts': {}
    },
    'large_files': [],
    'model_files': [],
    'eval_files': [],
    'temp_files': [],
    'cache_dirs': [],
    'docs': []
}

def format_size(size):
    return f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"

def walk_repo(path, level=0):
    entries = sorted(list(os.scandir(path)), key=lambda e: (not e.is_dir(), e.name.lower()))
    for entry in entries:
        if entry.name in ignore_dirs:
            continue
            
        try:
            stat = entry.stat()
            mod_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        except Exception:
            continue
            
        rel_path = Path(entry.path).relative_to(ROOT).as_posix()
        
        if entry.is_dir():
            data['stats']['total_folders'] += 1
            data['tree'].append(f"{'  ' * level}📁 {entry.name}/")
            if entry.name in ['__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.coverage']:
                data['cache_dirs'].append(rel_path)
            walk_repo(entry.path, level + 1)
        else:
            data['stats']['total_files'] += 1
            size = stat.st_size
            data['stats']['total_size_bytes'] += size
            
            ext = os.path.splitext(entry.name)[1].lower()
            data['stats']['ext_counts'][ext] = data['stats']['ext_counts'].get(ext, 0) + 1
            
            data['tree'].append(f"{'  ' * level}📄 {entry.name} ({format_size(size)}, {mod_time})")
            
            if size > 5 * 1024 * 1024:
                data['large_files'].append({'path': rel_path, 'size': size})
                
            if ext in ['.pkl', '.joblib', '.pt', '.bin']:
                data['model_files'].append({'path': rel_path, 'size': size, 'name': entry.name})
                
            if 'eval' in entry.name.lower() or ext in ['.csv'] or 'output' in entry.name.lower():
                if entry.name != 'cysecbench.csv': # just heuristic
                    data['eval_files'].append({'path': rel_path, 'name': entry.name})
                    
            if entry.name.startswith('test_') or entry.name.startswith('temp_') or entry.name.startswith('scratch') or 'study' in entry.name.lower() or ext == '.ipynb':
                data['temp_files'].append({'path': rel_path, 'name': entry.name})
                
            if ext in ['.md', '.txt'] and ('readme' in entry.name.lower() or 'audit' in entry.name.lower() or 'doc' in entry.name.lower()):
                data['docs'].append({'path': rel_path, 'name': entry.name})

walk_repo(ROOT)

with open('repo_audit_results.json', 'w') as f:
    json.dump(data, f, indent=2)
