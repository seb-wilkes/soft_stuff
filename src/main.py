import torch
import json
from transformer_lens import HookedTransformer
from huggingface_hub import login
from config import Config
from soft_embed import SoftEmbed

# RUN CONSTANTS
DTYPE = torch.bfloat16

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    
    # Load configuration
    config = load_config("../config/config.json") # assuming repo structure is maintained

    if config['login_details'] is not None:
        login(config['login_details'])
    else:
        raise ValueError("Please provide login details in config.json")

    # Load the model
    model = HookedTransformer.from_pretrained(config['model_id'], dtype=torch.bfloat16, device=config['model_config']['device'])

    # Create Config instance for convenience
    model_config = Config(**config['model_config'], target_txt=config['target_string'])

    # Create SoftEmbed instance
    soft_embed = SoftEmbed(model, model_config)

    # Train the soft embed
    soft_embed.train()

    # Qualitative test
    soft_embed.qualitative_test(model_config.num_of_tokens_to_generate)
    
    if "{soft_embeds}" not in model_config.rephrase_str_prefix or "{soft_embeds}" not in model_config.rephrase_str_suffix:
        raise ValueError("rephrase_str_prefix or rephrase_str_suffix must contain {soft_embeds}")

    full_prompt = [
        {"role": "system", "content": model_config.rephrase_str_prefix},
        {"role": "user", "content": model_config.rephrase_str_suffix}
    ]

    full_stuff = model.tokenizer.apply_chat_template(full_prompt, tokenize=False)
    prefix_str, suffix_str = full_stuff.split("{soft_embeds}")

    soft_embed.interrogation(prefix_str, suffix_str, model_config.num_of_tokens_to_generate)

if __name__ == "__main__":
    main()