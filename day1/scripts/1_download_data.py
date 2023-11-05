from pathlib import Path

import requests
from tqdm import tqdm

urls = [
  ("https://osf.io/agvxh/download", "neuropixels", "steinmetz_part0.npz"),
  ("https://osf.io/uv3mw/download", "neuropixels", "steinmetz_part1.npz"),
  ("https://osf.io/ehmw2/download", "neuropixels", "steinmetz_part2.npz"),
  ("https://osf.io/4bjns/download", "lfp", "steinmetz_st.npz"),
  ("https://osf.io/ugm9v/download", "lfp", "steinmetz_wav.npz"),
  ("https://osf.io/kx3v9/download", "lfp", "steinmetz_lfp.npz"),
]

base_path = Path("data/raw")

for url, subdir, fname in tqdm(urls):
    path: Path = base_path / subdir / fname
    path.parent.mkdir(parents=True, exist_ok=True)  
    r = requests.get(url)
    r.raise_for_status()
    path.write_bytes(r.content)
