# Finaltest
# 项目简介

随着金融环境的发展和变化，个人贷款成为了贷款市场中重要的组成部分。本研究在对数据集进行探索性分析的基础上，建立 LightGBM 模型进行预测；同时综合对比 Catboost 模型的运行效果，对构建科学有效的个人贷款风险预测模型进行了探索。研究发现，对于个人贷款违约预测，LightGBM 模型与 Catboost 模型均表现出了一定的预测精确度；调参前 CatBoost 的 AUC 值优于 LightGBM，但调参后 LightGBM 效果更好，反映了 LightGBM 在经过精细调参后能够更有效地利用数据特征。实际情况中，应根据计算成本和应用场景进行选择。

## 复现说明

### （1）版本信息
python 3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]

### （2）推荐使用的开发环境
- Jupyter Notebook
- Spyder
- VsCode

### （3）复现步骤
#### ① 环境准备
- 确保安装了 Python，并且版本与代码兼容。
- 安装必要的库：pandas, numpy, matplotlib, seaborn, datetime, warnings, lightgbm, catboost, sklearn, bayes_opt。
- 准备数据文件：确保拥有与 data_train 和 data_testA 相对应的 CSV 文件，并且文件路径与代码中的路径一致。

#### ② 导入库
- 导入 pandas, numpy, matplotlib, seaborn, datetime, warnings 等库。版本信息见（4）。

#### ③ 读取数据
- 使用 pandas 的 read_csv 方法从指定路径读取训练和测试数据。

#### ④ 数据探索
- 查看数据集的初始状态，包括数据集大小，统计信息，以及缺失值情况。
- 可视化不同特征的分布情况。

#### ⑤ 特征处理
- 对数据集中的特征进行预处理，包括处理缺失值、转换类别型变量、数据离散化、异常值处理等。

#### ⑥ 模型训练
- 定义 LightGBM 和 CatBoost 模型的训练函数。
- 使用 K 折交叉验证进行模型训练。

#### ⑦ 模型调优
- 使用贝叶斯优化对 LightGBM 模型的参数进行优化。
- 定义一个 ModelOptimizer 类来处理参数优化。
- 使用优化后的参数重新训练模型。

#### ⑧ 模型评估
- 使用 ROC 曲线和 AUC 评估模型性能。

#### ⑨ 结果可视化
- 绘制 ROC 曲线并显示 AUC 值。

#### *注意事项
- a. 在执行代码之前，请确保数据文件的路径正确，并已放置在相应位置。
- b. 根据您的环境配置（如不同的操作系统或文件路径），可能需要调整代码中的文件路径。
- c. 如果遇到库导入错误，请检查是否已安装相应的 Python 库。
- d. 确保在执行代码之前已经完成所有必要的环境设置。

### （4）installed_packages_list
- [['aiobotocore==2.5.0', 'aiofiles==22.1.0', 'aiohttp==3.8.5', 'aioitertools==0.7.1', 'aiosignal==1.2.0', 'aiosqlite==0.18.0', 'alabaster==0.7.12', 'anaconda-anon-usage==0.4.2', 'anaconda-catalogs==0.2.0', 'anaconda-client==1.12.1', 'anaconda-cloud-auth==0.1.3', 'anaconda-navigator==2.5.0', 'anaconda-project==0.11.1', 'anyio==3.5.0', 'appdirs==1.4.4', 'argcomplete==3.1.6', 'argon2-cffi-bindings==21.2.0', 'argon2-cffi==21.3.0', 'arrow==1.2.3', 'astroid==2.14.2', 'astropy==5.1', 'asttokens==2.0.5', 'async-timeout==4.0.2', 'atomicwrites==1.4.0', 'attrs==22.1.0', 'automat==20.2.0', 'autopep8==1.6.0', 'babel==2.11.0', 'backcall==0.2.0', 'backports.functools-lru-cache==1.6.4', 'backports.tempfile==1.0', 'backports.weakref==1.0.post1', 'bayesian-optimization==1.4.3', 'bcrypt==3.2.0', 'beautifulsoup4==4.12.2', 'binaryornot==0.4.4', 'black==0.0', 'bleach==4.1.0', 'bokeh==3.2.1', 'boltons==23.0.0', 'botocore==1.29.76', 'bottleneck==1.3.5', 'branca==0.7.0', 'brotlipy==0.7.0', 'calmap==0.0.11', 'catboost==1.2.2', 'certifi==2023.7.22', 'cffi==1.15.1', 'cfgv==3.4.0', 'chardet==4.0.0', 'charset-normalizer==2.0.4', 'click==8.0.4', 'cloudpickle==2.2.1', 'clyent==1.2.2', 'colorama==0.4.6', 'colorcet==3.0.1', 'comm==0.1.2', 'conda-build==3.26.1', 'conda-content-trust==0.2.0', 'conda-index==0.3.0', 'conda-libmamba-solver==23.7.0', 'conda-pack==0.6.0', 'conda-package-handling==2.2.0', 'conda-package-streaming==0.9.0', 'conda-repo-cli==1.0.75', 'conda-token==0.4.0', 'conda-verify==3.4.2', 'conda==23.7.4', 'constantly==15.1.0', 'contourpy==1.0.5', 'cookiecutter==1.7.3', 'cryptography==41.0.3', 'cssselect==1.1.0', 'cycler==0.11.0', 'cython==3.0.7', 'cytoolz==0.12.0', 'daal4py==2023.1.1', 'dask==2023.6.0', 'datasets==2.12.0', 'datashader==0.15.2', 'datashape==0.5.4', 'debugpy==1.6.7', 'decorator==5.1.1', 'defusedxml==0.7.1', 'diff-match-patch==20200713', 'dill==0.3.6', 'distlib==0.3.7', 'distributed==2023.6.0', 'docstring-to-markdown==0.11', 'docutils==0.18.1', 'entrypoints==0.4', 'et-xmlfile==1.1.0', 'executing==0.8.3', 'factor-analyzer==0.5.0', 'fastjsonschema==2.16.2', 'filelock==3.13.1', 'flake8==6.0.0', 'flask==2.2.2', 'folium==0.15.1', 'fonttools==4.25.0', 'frozenlist==1.3.3', 'fsspec==2023.4.0', 'future==0.18.3', 'gensim==4.3.0', 'glob2==0.7', 'graphviz==0.20.1', 'greenlet==2.0.1', 'h5py==3.9.0', 'heapdict==1.0.1', 'holoviews==1.17.1', 'huggingface-hub==0.15.1', 'hvplot==0.8.4', 'hyperlink==21.0.0', 'identify==2.5.33', 'idna==3.4', 'imagecodecs==2023.1.23', 'imageio==2.26.0', 'imagesize==1.4.1', 'imbalanced-learn==0.10.1', 'importlib-metadata==6.0.0', 'incremental==21.3.0', 'inflection==0.5.1', 'iniconfig==1.1.1', 'intake==0.6.8', 'intervaltree==3.1.0', 'ipykernel==6.25.0', 'ipython-genutils==0.2.0', 'ipython==8.15.0', 'ipywidgets==8.0.4', 'isort==5.9.3', 'itemadapter==0.3.0', 'itemloaders==1.0.4', 'itsdangerous==2.0.1', 'jaraco.classes==3.2.1', 'jedi==0.18.1', 'jellyfish==1.0.1', 'jinja2-time==0.2.0', 'jinja2==3.1.2', 'jmespath==0.10.0', 'joblib==1.2.0', 'json5==0.9.6', 'jsonpatch==1.32', 'jsonpointer==2.1', 'jsonschema==4.17.3', 'jupyter-client==7.4.9', 'jupyter-console==6.6.3', 'jupyter-core==5.3.0', 'jupyter-events==0.6.3', 'jupyter-server-fileid==0.9.0', 'jupyter-server-ydoc==0.8.0', 'jupyter-server==1.23.4', 'jupyter-ydoc==0.2.4', 'jupyter==1.0.0', 'jupyterlab-pygments==0.1.2', 'jupyterlab-server==2.22.0', 'jupyterlab-widgets==3.0.5', 'jupyterlab==3.6.3', 'kaleido==0.2.1', 'keyring==23.13.1', 'kiwisolver==1.4.4', 'lazy-loader==0.2', 'lazy-object-proxy==1.6.0', 'libarchive-c==2.9', 'libmambapy==1.5.1', 'lightgbm==4.2.0', 'linkify-it-py==2.0.0', 'llvmlite==0.40.0', 'lmdb==1.4.1', 'locket==1.0.0', 'lxml==4.9.3', 'lz4==4.3.2', 'markdown-it-py==2.2.0', 'markdown==3.4.1', 'markupsafe==2.1.1', 'matplotlib-inline==0.1.6', 'matplotlib==3.7.2', 'mccabe==0.7.0', 'mdit-py-plugins==0.3.0', 'mdurl==0.1.0', 'menuinst==1.4.19', 'mistune==0.8.4', 'mizani==0.9.3', 'mkl-fft==1.3.8', 'mkl-random==1.2.4', 'mkl-service==2.4.0', 'more-itertools==8.12.0', 'mpmath==1.3.0', 'msgpack==1.0.3', 'multidict==6.0.2', 'multipledispatch==0.6.0', 'multiprocess==0.70.14', 'munkres==1.1.4', 'mypy-extensions==1.0.0', 'navigator-updater==0.4.0', 'nbclassic==0.5.5', 'nbclient==0.5.13', 'nbconvert==6.5.4', 'nbformat==5.9.2', 'nest-asyncio==1.5.6', 'networkx==3.1', 'nltk==3.8.1', 'nodeenv==1.8.0', 'notebook-shim==0.2.2', 'notebook==6.5.4', 'numba==0.57.1', 'numexpr==2.8.4', 'numpy==1.24.3', 'numpydoc==1.5.0', 'openpyxl==3.0.10', 'packaging==23.1', 'pandas==2.0.3', 'pandocfilters==1.5.0', 'panel==1.2.3', 'param==1.13.0', 'paramiko==2.8.1', 'parsel==1.6.0', 'parso==0.8.3', 'partd==1.4.0', 'pathlib==1.0.1', 'pathspec==0.10.3', 'patsy==0.5.3', 'pep8==1.7.1', 'pexpect==4.8.0', 'pickleshare==0.7.5', 'pillow==9.4.0', 'pip==23.2.1', 'pipx==1.2.1', 'pkce==1.0.3', 'pkginfo==1.9.6', 'platformdirs==3.10.0', 'plotly==5.9.0', 'plotnine==0.12.4', 'pluggy==1.0.0', 'ply==3.11', 'poyo==0.5.0', 'pre-commit==3.6.0', 'prometheus-client==0.14.1', 'prompt-toolkit==3.0.36', 'protego==0.1.16', 'psutil==5.9.0', 'ptyprocess==0.7.0', 'pure-eval==0.2.2', 'py-cpuinfo==8.0.0', 'pyarrow==11.0.0', 'pyasn1-modules==0.2.8', 'pyasn1==0.4.8', 'pycodestyle==2.10.0', 'pycosat==0.6.4', 'pycparser==2.21', 'pyct==0.5.0', 'pycurl==7.45.2', 'pydantic==1.10.8', 'pydispatcher==2.0.5', 'pydocstyle==6.3.0', 'pyerfa==2.0.0', 'pyflakes==3.0.1', 'pygments==2.15.1', 'pyjwt==2.4.0', 'pylint-venv==2.3.0', 'pylint==2.16.2', 'pyls-spyder==0.4.0', 'pynacl==1.5.0', 'pyodbc==4.0.34', 'pyopenssl==23.2.0', 'pyparsing==3.0.9', 'pyqt5-sip==12.11.0', 'pyqt5==5.15.7', 'pyqtwebengine==5.15.4', 'pyrsistent==0.18.0', 'pysocks==1.7.1', 'pytest==7.4.0', 'python-dateutil==2.8.2', 'python-dotenv==0.21.0', 'python-json-logger==2.0.7', 'python-lsp-black==1.2.1', 'python-lsp-jsonrpc==1.0.0', 'python-lsp-server==1.7.2', 'python-slugify==5.0.2', 'python-snappy==0.6.1', 'pytoolconfig==1.2.5', 'pytz==2023.3.post1', 'pyviz-comms==2.3.0', 'pywavelets==1.4.1', 'pywin32-ctypes==0.2.0', 'pywin32==305.1', 'pywinpty==2.0.10', 'pyyaml==6.0', 'pyzmq==23.2.0', 'qdarkstyle==3.0.2', 'qstylizer==0.2.2', 'qtawesome==1.2.2', 'qtconsole==5.4.2', 'qtpy==2.2.0', 'queuelib==1.5.0', 'regex==2022.7.9', 'requests-file==1.5.1', 'requests-toolbelt==1.0.0', 'requests==2.31.0', 'responses==0.13.3', 'rfc3339-validator==0.1.4', 'rfc3986-validator==0.1.1', 'rope==1.7.0', 'rtree==1.0.1', 'ruamel-yaml-conda==0.17.21', 'ruamel.yaml==0.17.21', 's3fs==2023.4.0', 'safetensors==0.3.2', 'scikit-image==0.20.0', 'scikit-learn-intelex==20230426.121932', 'scikit-learn==1.3.0', 'scipy==1.11.1', 'scrapy==2.8.0', 'seaborn==0.12.2', 'send2trash==1.8.0', 'service-identity==18.1.0', 'setuptools==68.0.0', 'sip==6.6.2', 'six==1.16.0', 'smart-open==5.2.1', 'sniffio==1.2.0', 'snowballstemmer==2.2.0', 'sortedcontainers==2.4.0', 'soupsieve==2.4', 'sphinx==5.0.2', 'sphinxcontrib-applehelp==1.0.2', 'sphinxcontrib-devhelp==1.0.2', 'sphinxcontrib-htmlhelp==2.0.0', 'sphinxcontrib-jsmath==1.0.1', 'sphinxcontrib-qthelp==1.0.3', 'sphinxcontrib-serializinghtml==1.1.5', 'spyder-kernels==2.4.4', 'spyder==5.4.3', 'sqlalchemy==1.4.39', 'stack-data==0.2.0', 'statsmodels==0.14.0', 'sympy==1.11.1', 'tables==3.8.0', 'tabulate==0.8.10', 'tbb==0.2', 'tblib==1.7.0', 'tenacity==8.2.2', 'terminado==0.17.1', 'text-unidecode==1.3', 'textdistance==4.2.1', 'threadpoolctl==2.2.0', 'three-merge==0.1.1', 'tifffile==2023.4.12', 'tinycss2==1.2.1', 'tldextract==3.2.0', 'toad==0.1.3', 'tokenizers==0.13.2', 'toml==0.10.2', 'tomlkit==0.11.1', 'toolz==0.12.0', 'tornado==6.3.2', 'tqdm==4.65.0', 'traitlets==5.7.1', 'transformers==4.32.1', 'twisted-iocpsupport==1.0.2', 'twisted==22.10.0', 'typing-extensions==4.7.1', 'tzdata==2023.3', 'uc-micro-py==1.0.1', 'ujson==5.4.0', 'unidecode==1.2.0', 'urllib3==1.26.16', 'userpath==1.9.1', 'virtualenv==20.24.7', 'w3lib==1.21.0', 'watchdog==2.1.6', 'wcwidth==0.2.5', 'webencodings==0.5.1', 'websocket-client==0.58.0', 'werkzeug==2.2.3', 'whatthepatch==1.0.2', 'wheel==0.38.4', 'widgetsnbextension==4.0.5', 'win-inet-pton==1.1.0', 'wordcloud==1.9.3', 'wrapt==1.14.1', 'xarray==2023.6.0', 'xgboost==2.0.3', 'xlwings==0.29.1', 'xxhash==2.0.2', 'xyzservices==2022.9.0', 'y-py==0.5.9', 'yapf==0.31.0', 'yarl==1.8.1', 'ypy-websocket==0.8.2', 'zict==2.2.0', 'zipp==3.11.0', 'zope.interface==5.4.0', 'zstandard==0.19.0']]
