#!/usr/bin/env python3
"""Generate a DA3-Streaming ablation config by patching a base yaml.

Usage:
  python3 gen_ablation_config.py <base.yaml> <out.yaml> key=value [key=value ...]

Keys use dotted paths, e.g.:
  python3 gen_ablation_config.py configs/turboreg_config.yaml /tmp/x.yaml \
      TurboReg.epsilon_scale=0.10 Model.chunk_size=60 Model.overlap=30
"""
import sys
import yaml


def main():
    base_path, out_path = sys.argv[1], sys.argv[2]
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    for pair in sys.argv[3:]:
        dotted, _, raw = pair.partition('=')
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        keys = dotted.split('.')
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    with open(out_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f'written: {out_path}')


if __name__ == '__main__':
    main()
