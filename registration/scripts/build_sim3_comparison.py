"""Build isolated comparison solvers without changing frozen TurboReg binaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sysconfig

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'bindings/sim3_comparison.cpp'
target=ROOT/'build/sim3_completion'/('sim3_comparison'+sysconfig.get_config_var('EXT_SUFFIX'))
target.parent.mkdir(parents=True,exist_ok=True)
torch_include=Path(importlib.util.find_spec('torch').origin).parent/'include'
command=['g++','-std=c++17','-O3','-DNDEBUG','-DEIGEN_DONT_PARALLELIZE','-shared','-fPIC',
         '-I/usr/include/eigen3','-I'+sysconfig.get_paths()['include'],'-I'+str(torch_include),str(source),'-o',str(target)]
subprocess.run(command,check=True)
metadata={'command':command,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0]}
(target.parent/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(target)
