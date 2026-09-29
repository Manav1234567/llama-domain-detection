import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

def load_model_bundle(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cpu", dtype=torch.float32, tl_template="llama-3-8b-instruct"):
    print(f"Loading {model_name}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hooked_model = HookedTransformer.from_pretrained_no_processing(
        tl_template,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    hooked_model.eval()
    return hooked_model, hf_model, tokenizer

class PolygraphFeatureExtractor:
    def __init__(self, model, hf_gen_model, hf_tokenizer, device):
        self.model = model
        self.hf_gen_model = hf_gen_model
        self.hf_tokenizer = hf_tokenizer
        self.device = device
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.current_prompt_len = 0
        self.reset_storage()

    def reset_storage(self):
        self.storage = {
            "residual_means": {}, "residual_last": {},
            "mlp_first": {}, "mlp_last": {},
            "attn_entropy_mean": {}, "attn_entropy_max": {},
            "lookback_ratio_mean": {}, "lookback_ratio_last": {},
        }

    def register_hooks(self):
        self.hooks = []
        for layer in range(self.n_layers):
            self.hooks.append(self.model.blocks[layer].hook_resid_post.add_hook(
                lambda val, hook, l=layer: self._hook_residual(val, hook, l)))
            self.hooks.append(self.model.blocks[layer].attn.hook_pattern.add_hook(
                lambda val, hook, l=layer: self._hook_attention(val, hook, l)))
            self.hooks.append(self.model.blocks[layer].mlp.hook_post.add_hook(
                lambda val, hook, l=layer: self._hook_mlp(val, hook, l)))

    def remove_hooks(self):
        self.model.reset_hooks()

    def _hook_residual(self, value, hook, layer_idx):
        tensor = value.detach().to(torch.float32).cpu()[0]
        self.storage["residual_means"][layer_idx] = tensor.mean(dim=0).numpy()
        self.storage["residual_last"][layer_idx] = tensor[-1, :].numpy()
        return value

    def _hook_mlp(self, value, hook, layer_idx):
        tensor = value.detach().to(torch.float32).cpu()[0]
        self.storage["mlp_first"][layer_idx] = tensor[0, :].numpy()
        self.storage["mlp_last"][layer_idx] = tensor[-1, :].numpy()
        return value

    def _hook_attention(self, value, hook, layer_idx):
        matrix = value.detach().to(torch.float32).cpu()[0]
        epsilon = 1e-9
        entropy = -torch.sum(matrix * torch.log(matrix + epsilon), dim=-1)
        self.storage["attn_entropy_mean"][layer_idx] = torch.mean(entropy, dim=-1).numpy()
        self.storage["attn_entropy_max"][layer_idx] = torch.max(entropy, dim=-1)[0].numpy()
        
        last_token_attn = matrix[:, -1, :]
        p_len = self.current_prompt_len
        prompt_attn = torch.sum(last_token_attn[:, :p_len], dim=-1)
        gen_attn = torch.sum(last_token_attn[:, p_len:], dim=-1)
        ratio = prompt_attn / (gen_attn + epsilon)
        self.storage["lookback_ratio_mean"][layer_idx] = torch.mean(ratio, dim=-1).numpy()
        self.storage["lookback_ratio_last"][layer_idx] = torch.max(ratio, dim=-1)[0].numpy()
        return value

    def generate_native(self, prompt, max_new_tokens=50):
        inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.hf_gen_model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
