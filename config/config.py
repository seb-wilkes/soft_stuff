from dataclasses import dataclass

@dataclass
class Config:
    target_txt: str
    rephrase_str_prefix: str = ""
    rephrase_str_suffix: str = ""
    tsl: int = 12 # token sequence length
    n_inst: int = 2
    scale: float = 0.6
    device: str = "cuda"
    lr: float = 1e-2
    n_steps: int = 1000
    num_of_tokens_to_generate: int = 30
    verbose: bool = False
    save_file_location: str = None
    