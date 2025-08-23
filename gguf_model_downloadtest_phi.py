from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="TheBloke/phi-2-GGUF",
    filename="phi-2.Q4_K_M.gguf",
    local_dir="models/gguf",
    local_dir_use_symlinks=False
)
print("Saved to:", path)
