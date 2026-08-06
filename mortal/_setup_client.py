"""Called by setup_client.bat to patch config_v2.py paths and disable AMP"""
import os, re, sys

root = sys.argv[1].replace('\\', '/')
server_ip = sys.argv[2]
device = sys.argv[3]

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_v2.py')

def patch(filepath, replacements):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    for pattern, repl in replacements:
        content = re.sub(pattern, repl, content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# config_v2.py: disable AMP, set device, server IP, and local paths
# device/state_file 多处存在，统一替换为目标机环境
patch(config_path, [
    (r"'enable_amp':\s*True", "'enable_amp': False"),
    (r"'device':\s*'[^']*'", f"'device': '{device}'"),
    (r"'host':\s*'[^']*'", f"'host': '{server_ip}'"),
    (r"'log_dir':\s*'[^']*train_play[^']*'", f"'log_dir': '{root}/train_play'"),
    (r"'buffer_dir':\s*'[^']*'", f"'buffer_dir': '{root}/buffer'"),
    (r"'drain_dir':\s*'[^']*'", f"'drain_dir': '{root}/drain'"),
    (r"'opponents_dir':\s*'[^']*'", f"'opponents_dir': '{root}/opponents'"),
    (r"'state_file':\s*'[^']*baseline[^']*'", f"'state_file': '{root}/baseline_v1/mortal.pth'"),
])

print(f"  config_v2.py: enable_amp=false, device={device}")
print(f"  config_v2.py: server={server_ip}, paths={root}/...")
