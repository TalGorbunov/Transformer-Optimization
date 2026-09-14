"""Upload the transfer tarballs to a PRIVATE HF dataset repo (the 'tunnel' to Nebius).
Run from the repo root with the bridge token (the default env token is read-only):
    HF_TOKEN=$(cat ~/.hf_bridge_token) .venv/bin/python outputs/_scratch/dbg/hf_tunnel_up.py
The repo is created under whatever account that token belongs to; the printed repo id is what
the Nebius side downloads from (with its own copy of ~/.hf_bridge_token)."""
from huggingface_hub import HfApi

FILES = ("wip_f33d4c8.tgz", "mmred_filtered_test.tgz")

api = HfApi()
me = api.whoami()
repo = f"{me['name']}/to-nebius-tunnel"
print("account:", me["name"], "| repo:", repo)
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for f in FILES:
    api.upload_file(path_or_fileobj=f"outputs/_scratch/dbg/{f}", path_in_repo=f,
                    repo_id=repo, repo_type="dataset")
info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
print("private:", info.private, "| files:", [(s.rfilename, s.size) for s in info.siblings])
