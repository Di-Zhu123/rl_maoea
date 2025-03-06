    conda create --name contrib python=3.10
    conda activate contrib
    pip install stable-baselines3 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install sb3-contrib -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install evox -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install gym -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install pymoo -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install shimmy>=2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
Note:    
Stable-Baselines3 requires python 3.9+ and PyTorch >= 2.3
