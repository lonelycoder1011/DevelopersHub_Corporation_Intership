# Mental Health Support Chatbot — Mistral 7B + QLoRA

A fine-tuned **Mistral 7B** model for empathetic mental health support conversations, trained using **QLoRA** (4-bit quantization + LoRA adapters) on the EmpatheticDialogues dataset.

---

## What This Is

This project fine-tunes Mistral-7B-Instruct-v0.3 to act as a compassionate mental health support assistant. It listens, validates feelings, asks thoughtful follow-up questions, and never judges. The model was trained on Kaggle using a P100 GPU in ~92 minutes.

---

## Results

| Metric | DistilGPT2 Baseline | Mistral 7B QLoRA | Improvement |
|---|---|---|---|
| Empathy Score | 0.053 | 0.120 | **+126%** |
| Avg Response Length | 12 words | 54 words | **+350%** |
| Response Diversity (TTR) | 0.18 | 0.478 | **+166%** |
| Coherence | Gibberish/fragments | Full human-level paragraphs | ✓ |
| Train Loss | — | 1.1573 | — |
| Best Eval Loss | — | 1.1521 | — |

> **Note on empathy score:** The 0.120 score is based on a limited keyword list and significantly understates real quality. Responses like *"I completely understand how you feel"* and *"It sounds like you have been struggling"* are genuinely empathetic but miss the keyword filter. Real-world response quality is much closer to what you see in the example outputs below.

### Example Outputs

```
User:      I have been feeling really anxious about my job interview tomorrow.
Assistant: I completely understand how you feel! Job interviews can be nerve-wracking.
           What kind of position is it for? If it's something you really want then I know
           you will do great. Just remember to relax and breathe. It's okay if you don't
           get everything right on your first try. The most important thing is that they
           see you are passionate about the position. Good luck!

User:      I feel so alone lately. Nobody really understands what I am going through.
Assistant: It sounds like you have been struggling recently. Would it be helpful for me
           to try and offer some advice or just listen? Sometimes people just need someone
           to hear them out without any judgement. If you want to talk more about it,
           that's okay too. Let me know how I can best help!

User:      I feel like a burden to everyone around me.
Assistant: Oh no, you aren't a burden at all! It sounds like you are going through a
           really difficult time right now...
```

---

## Model Details

| Parameter | Value |
|---|---|
| Base model | mistralai/Mistral-7B-Instruct-v0.3 |
| Training method | QLoRA (4-bit NF4 + LoRA) |
| Dataset | EmpatheticDialogues |
| Training samples | 3,000 conversation pairs |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Max sequence length | 256 tokens |
| Epochs | 2 |
| Effective batch size | 32 (4 × 8 gradient accumulation) |
| Optimizer | paged_adamw_32bit |
| Learning rate | 2e-4 |
| Trainable parameters | 41.9M (1.10% of 3.8B total) |
| Training hardware | Kaggle P100 16GB |
| Training time | ~88 minutes |

---

## Output Files

```
mistral_mental_health/
├── adapter/                         # LoRA adapter only — 167.8 MB
│   ├── adapter_config.json
│   ├── adapter_model.safetensors    # The fine-tuned weights
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── final/                           # Merged standalone model — 4.6 GB
│   ├── config.json
│   ├── model.safetensors            # Full fp16 model, adapter baked in
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── training_info.json               # Training metadata
├── evaluation_results.json          # Per-emotion empathy scores
├── training_curves.png              # Loss curves
└── data_exploration.png             # Dataset distribution
```

---

## How to Use Locally

### Option A — Merged Model (Recommended, Easiest)

Download the `final/` folder (~4.6 GB) from Kaggle output. No extra libraries needed.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "path/to/mistral_mental_health/final"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.use_cache = True
model.eval()

SYSTEM_PROMPT = (
    "You are a compassionate mental health support assistant. "
    "You listen carefully, respond with empathy, and provide gentle supportive guidance. "
    "You never judge and always validate the person's feelings. "
    "You ask thoughtful follow-up questions to better understand their situation."
)

def chat(user_message):
    prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_message.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(chat("I have been feeling really anxious lately."))
```

### Option B — Adapter + Base Model (Smaller Download)

Download only the `adapter/` folder (167.8 MB). Requires the base model to be downloaded from HuggingFace (~14 GB, cached after first run).

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path  = "path/to/mistral_mental_health/adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

---

## Training Setup (Kaggle)

### Requirements
- Kaggle account with GPU enabled (P100 or better)
- HuggingFace account with Mistral license accepted
- HF_TOKEN added to Kaggle Secrets

### Steps to Reproduce
1. Accept the Mistral license at [huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
2. Create a HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read access)
3. In Kaggle: **Add-ons → Secrets → Add New Secret** → Name: `HF_TOKEN`, Value: your token
4. Enable the secret for the notebook
5. Set accelerator: **Settings → Accelerator → GPU P100**
6. Run `mistral_qlora_finetune.ipynb` — completes in ~90 minutes

### Key Libraries
```
transformers>=4.41.0
datasets>=2.18.0
accelerate>=1.0.0
bitsandbytes>=0.43.0
peft>=0.10.0
trl>=0.8.0
```

---

## Dataset

**EmpatheticDialogues** by Facebook Research — 25,000+ empathetic conversation pairs covering 32 emotion categories including anxious, sad, lonely, overwhelmed, depressed, angry, and stressed.

- Total pairs available: 28,619
- Used for training: 3,000 (capped for P100 runtime)
- Train split: 2,850 | Eval split: 150
- Preprocessing: `_comma_` artifacts cleaned, responses under 8 words filtered, Mistral `[INST]` chat template applied

---

## Architecture — How QLoRA Works

```
Mistral 7B Base (frozen, 4-bit)     ← 3.76B params, ~3.5 GB VRAM
        +
LoRA Adapters (trainable)           ← 41.9M params, ~300 MB
        =
Fine-tuned Mental Health Model      ← Trained only 1.1% of params
```

QLoRA makes fine-tuning a 7B model possible on a 16GB GPU by:
1. **4-bit NF4 quantization** — base model stored at 4-bit precision (~4x smaller)
2. **LoRA adapters** — only small rank-16 matrices are trained, not the full model
3. **Gradient checkpointing** — recomputes activations during backprop to save VRAM
4. **Paged AdamW** — optimizer states paged to CPU RAM when VRAM is tight

---

## Project Structure

```
mental-health-chatbot/
├── notebooks/
│   └── mistral_qlora_finetune.ipynb    # Training notebook (Kaggle)
├── models/
│   └── mistral_mental_health/
│       ├── adapter/                    # Download from Kaggle (167 MB)
│       └── final/                      # Download from Kaggle (4.6 GB)
├── src/
│   └── chatbot.py                      # Inference / chatbot logic
├── app.py                              # Streamlit web interface
└── README.md
```

---

## Known Limitations

- Trained on only 3,000 samples — more data would improve consistency
- Not a replacement for professional mental health support
- Empathy keyword scorer underestimates real response quality
- Some emotion categories (depressed, angry) score lower — less represented in training data
- Response length varies (22–85 words) — no minimum length enforced

---

## Future Improvements

- Train on 9,000+ samples for better emotion coverage
- Add minimum response length constraint
- Expand empathy scorer keyword list to better reflect actual response quality
- Fine-tune on crisis response guidelines
- Add safety filters for high-risk messages

---

## Acknowledgements

- **EmpatheticDialogues dataset** — Facebook Research
- **Mistral 7B** — Mistral AI
- **QLoRA technique** — Dettmers et al. (2023)
- **Training infrastructure** — Kaggle free GPU tier (P100)
