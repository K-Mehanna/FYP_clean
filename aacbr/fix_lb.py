import json
path = 'leaderboard_trained.json'
with open(path) as f:
    data = json.load(f)
before = len(data)
data = [r for r in data if 'brats_png_v14a_gamma4' not in r.get('config', {}).get('checkpoint', '')]
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
print(f'removed {before - len(data)} entries ({before} -> {len(data)})')
