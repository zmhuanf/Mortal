Client 部署需修改的参数：

conf/online.toml
  [online.remote] host         = server 机器 IP（端口 port 与 server 一致）
  [train_play.default] log_dir = 目标机本地绝对路径
  [baseline.train] state_file  = 部署包 baseline_v1/mortal.pth 的绝对路径
  [baseline.train] device      = 目标机 GPU 编号（单卡 cuda:0 可不动）

conf/base.toml
  [control] device             = 目标机 GPU 编号（单卡 cuda:0 可不动）

==================================================
目标机环境安装（国内源）

Python 版本必须为 3.11（libriichi.pyd 按 3.11 编译，不能换版本）

GPU 版 PyTorch（1050Ti 是 Pascal sm_61，torch 2.6+ 已不支持，须装 cu121 旧版）：
pip install torch==2.5.1+cu121 --find-links https://mirrors.aliyun.com/pytorch-wheels/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

其它新卡（Turing sm_75 及以上）可用 cu128 新版本，与本机 2.11.0+cu128 一致：
pip install torch==2.11.0+cu128 --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple

其余依赖（清华源）：
pip install numpy toml tqdm tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

需安装 Microsoft Visual C++ Redistributable（libriichi.pyd 依赖）
