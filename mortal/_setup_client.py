"""Called by setup_client.bat to patch conf paths and disable AMP"""
import os, re, sys

root = sys.argv[1].replace('\\', '/')
server_ip = sys.argv[2]
device = sys.argv[3]

script_dir = os.path.dirname(os.path.abspath(__file__))
conf_dir = os.path.join(script_dir, 'conf')

def patch(filepath, replacements):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    for pattern, repl in replacements:
        content = re.sub(pattern, repl, content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# base.toml: disable AMP, set device
patch(os.path.join(conf_dir, 'base.toml'), [
    (r"enable_amp\s*=\s*true", "enable_amp = false"),
    (r"device\s*=\s*'cuda:\d+'", f"device = '{device}'"),
])

# online.toml: fix paths and server IP
patch(os.path.join(conf_dir, 'online.toml'), [
    (r"host\s*=\s*'[^']*'", f"host = '{server_ip}'"),
    (r"log_dir\s*=\s*'[^']*'", f"log_dir = '{root}/train_play'"),
    (r"buffer_dir\s*=\s*'[^']*'", f"buffer_dir = '{root}/buffer'"),
    (r"drain_dir\s*=\s*'[^']*'", f"drain_dir = '{root}/drain'"),
    (r"opponents_dir\s*=\s*'[^']*'", f"opponents_dir = '{root}/opponents'"),
])

# baseline.train.state_file in online.toml
patch(os.path.join(conf_dir, 'online.toml'), [
    (r"state_file\s*=\s*'[^']*baseline[^']*'",
     f"state_file = '{root}/baseline_v1/mortal.pth'"),
])

print(f"  base.toml:    enable_amp=false, device={device}")
print(f"  online.toml:  server={server_ip}, paths={root}/...")
