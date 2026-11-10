#!/usr/bin/env python3
"""Build the standalone CPU backend without changing existing Torch binaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sysconfig
import time
ROOT=Path(__file__).resolve().parents[1]
torch_spec=importlib.util.find_spec('torch')
includes=[str(ROOT/'turboreg/include'),'/usr/include/eigen3',sysconfig.get_paths()['include'],
          str(Path(torch_spec.origin).parent/'include')]
sources=[ROOT/'turboreg/src/sim3_native.cpp',ROOT/'bindings/native_binding.cpp']
target=ROOT/'bindings'/('turboreg_native'+sysconfig.get_config_var('EXT_SUFFIX'))
command=['g++','-std=c++17','-O3','-DNDEBUG','-DEIGEN_DONT_PARALLELIZE','-shared','-fPIC']
command+=['-I'+p for p in includes]+[str(p) for p in sources]+['-o',str(target)]
subprocess.run(command,check=True)
meta=dict(command=command,created_unix=time.time(),binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
          compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
          source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[ROOT/'turboreg/include/turboreg/sim3_native.hpp']})
(ROOT/'build').mkdir(exist_ok=True)
(ROOT/'build/sim3_native_build.json').write_text(json.dumps(meta,indent=2)+'\n')
print(target)
