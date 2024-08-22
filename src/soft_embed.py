import torch
from torch import nn
from optimiser_classes import ConstrainedAdam
from tqdm import tqdm
import csv

# set up the prompt structure using {soft_embeds} as a common identifier
soft_optimisation_prompt = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "{soft_embeds}"}
    ]

class SoftEmbed(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self._initialize_tokens()
        self._initialize_soft_tokens()
        # now get the opening and closing format strings to be used later in the optimisation process
        strings_for_optimisation = model.tokenizer.apply_chat_template(soft_optimisation_prompt, tokenize=False)
        self.opening_format, self.closing_format = strings_for_optimisation.split("{soft_embeds}")
        self.verbose = self.cfg.verbose
        self.save_loc = self.cfg.save_file_location
        if self.save_loc is not None and ".csv" not in self.save_loc:
            self.save_loc = None
            print("Not saving to a csv file as the file location does not end in .csv")

    def _initialize_tokens(self):
        self.target_tokens = self.model.tokenizer.encode(self.cfg.target_txt, return_tensors='pt').to(self.cfg.device)
        self.target_tokens = self.target_tokens.repeat(self.cfg.n_inst, 1)
        self.blah_tokens = torch.ones(self.cfg.n_inst, self.cfg.tsl, dtype=torch.long).to(self.cfg.device)
        self.opening_format_tokens = self.model.tokenizer.encode(self.opening_format, return_tensors='pt').repeat(self.cfg.n_inst, 1).to(self.cfg.device)
        self.closing_format_tokens = self.model.tokenizer.encode(self.closing_format, return_tensors='pt').repeat(self.cfg.n_inst, 1).to(self.cfg.device)
        self.opening_token_gap = self.opening_format_tokens.shape[1]
        self.concat_tokens = torch.cat([self.opening_format_tokens, self.blah_tokens, self.closing_format_tokens, self.target_tokens], dim=1)

    def _initialize_soft_tokens(self):
        soft_tokens = torch.randn(self.cfg.n_inst, self.cfg.tsl, self.model.cfg.d_model, device=self.cfg.device, dtype=self.model.dtype)
        soft_tokens /= soft_tokens.norm(dim=-1, keepdim=True)
        self.soft_tokens = nn.Parameter(soft_tokens)

    def forward(self):
        def hook(act, hook):
            assert hook.name == "hook_embed"
            # act [inst, tsl + target_tokens.shape[1], d_model]
            act[:, self.opening_token_gap:self.opening_token_gap+self.cfg.tsl, :] = self.soft_tokens * self.cfg.scale
            return act
        loss = self.model.run_with_hooks(self.concat_tokens,
                                         return_type="loss",
                                         loss_per_token=True,
                                         fwd_hooks=[("hook_embed", hook)])

        loss = loss[:, -self.target_tokens.shape[1]:].sum()

        return loss

    def train(self):
        opt = ConstrainedAdam([self.soft_tokens], [self.soft_tokens], lr=self.cfg.lr)
        with tqdm(total=self.cfg.n_steps) as pbar:
            for _ in pbar:
                opt.zero_grad()
                loss = self.forward()
                loss.backward()
                opt.step()
                pbar.set_postfix({"Loss": loss.item().mean()})

    @torch.no_grad()
    def generate_next_token(self, response_tokens):
        # response_tokens have shape [n_inst, target_length]
        tokens_so_far = torch.cat([self.opening_format_tokens, self.blah_tokens, self.closing_format_tokens, response_tokens], dim=1)
        def hook(act, hook):
            assert hook.name == "hook_embed"
            # act [inst, tsl + target_tokens.shape[1], d_model]
            act[:, self.opening_token_gap:self.opening_token_gap+self.cfg.tsl, :] = self.soft_tokens * self.cfg.scale

            return act
        logits = self.model.run_with_hooks(tokens_so_far,
                                           return_type="logits",
                                           fwd_hooks=[("hook_embed", hook)]
                                           )[:,-1, :]
        preds = logits.argmax(dim=-1) # [n_inst,]
        new_response_tokens = torch.cat([response_tokens, preds.unsqueeze(1)], dim=1)

        return new_response_tokens

    @torch.no_grad()
    def qualitative_test(self, max_new_tokens):
        # generate some text
        response_tokens = torch.ones(self.cfg.n_inst, 0, dtype=torch.long).to(self.cfg.device)
        for i in range(max_new_tokens):
            response_tokens = self.generate_next_token(response_tokens)

        print(self.model.tokenizer.batch_decode(response_tokens))

    @torch.no_grad()
    def generate_next_token_given_rephrase(self, rephrase_tokens_prefix, rephrase_tokens_suffix, response_tokens):
        """Here we output the next token in the sequence, for a
        trained soft embedding. Note, rephrase_tokens_X [n_inst, xxx]"""
        # response_tokens have shape [n_inst, target_length]
        tokens_so_far = torch.cat([rephrase_tokens_prefix, self.blah_tokens, rephrase_tokens_suffix, response_tokens], dim=1)
        def hook(act, hook):
            assert hook.name == "hook_embed"
            # act [inst, tsl + target_tokens.shape[1], d_model]
            rephrase_prefix_length = rephrase_tokens_prefix.shape[1]
            act[:, rephrase_prefix_length:rephrase_prefix_length+self.cfg.tsl, :] = self.soft_tokens
            return act
        logits = self.model.run_with_hooks(tokens_so_far,
                                           return_type="logits",
                                           fwd_hooks=[("hook_embed", hook)]
                                           )[:,-1, :]
        preds = logits.argmax(dim=-1) # [n_inst,]
        new_response_tokens = torch.cat([response_tokens, preds.unsqueeze(1)], dim=1)

        return new_response_tokens

    @torch.no_grad()
    def interrogation(self, rephrase_str_prefix, rephrase_str_suffix, max_new_tokens):
        # generate some text
        rephrase_tokens_prefix = self.model.tokenizer.encode(rephrase_str_prefix, return_tensors='pt').to(self.cfg.device)
        rephrase_tokens_suffix = self.model.tokenizer.encode(rephrase_str_suffix, return_tensors='pt').to(self.cfg.device)
        # now repeat to make it n_inst
        rephrase_tokens_prefix = rephrase_tokens_prefix.repeat(self.cfg.n_inst, 1)
        rephrase_tokens_suffix = rephrase_tokens_suffix.repeat(self.cfg.n_inst, 1)

        response_tokens = torch.ones(self.cfg.n_inst, 0, dtype=torch.long).to(self.cfg.device)
        for i in range(max_new_tokens):
            response_tokens = self.generate_next_token_given_rephrase(rephrase_tokens_prefix, rephrase_tokens_suffix, response_tokens)

        decoded_string_list = self.model.tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
        if self.verbose:
            for i, string in enumerate(decoded_string_list):
                print(f"-------------INSTANCE {i}------------")
                print(string)
                print()
        if self.save_loc is not None:
            with open(self.save_loc, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(decoded_string_list)