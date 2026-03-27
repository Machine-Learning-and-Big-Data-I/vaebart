# 📊 Model Architecture Overview

This project proposes a **Dual Latent Variable Schema** built on top of a **BART-based encoder–decoder architecture** to transform **complex sentences into more simplified sentences**.

<img width="1099" height="683" alt="image" src="https://github.com/user-attachments/assets/d45ee157-edb4-4325-8a5d-1b4fa0f6e6f1" />

---

## 🔄 Workflow Summary

The model consists of three main stages:

1. **Input Preparation (Training Data)**
2. **Latent Representation Learning (Encoder + Dual Latent Variables)**
3. **Sequence Generation (Decoder)**

---

## 1. 📥 Training Data

The model is trained using paired text data:

- **Simple Sentences (Kalimat Sederhana)** → input to the decoder
- **Complete Sentences (Kalimat Lengkap)** → input to the encoder (target reference)

This setup enables the model to learn how to **expand or enrich simpler sentences into more detailed ones**.

---

## 2. 🧠 Encoder & Dual Latent Variables

### 🔹 BART Encoder

- The **complete sentence** is first processed by the **BART Encoder Block**
- Steps:
  - Token Embedding
  - Positional Encoding
  - Transformer layers → produce **hidden states**

---

### 🔹 Pooling Strategy

- The encoder’s last hidden states are passed through a pooling layer
- There are 3 pooling strategy such as: mean pooling (recommended), max pooling, weighted mean pooling and SOWE.
- These parameters define the latent distributions

```
    def pool(self, hidden_states, attention_mask):
        # hidden_states shape: bs, seq, hidden_dim
        output_vectors = []
        if self.pooling_strategy == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        elif self.pooling_strategy == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(hidden_states, 1)[0]
            output_vectors.append(max_over_time)

        elif self.pooling_strategy == "weightedmean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            # hidden_states shape: bs, seq, hidden_dim
            weights = (
                    torch.arange(start=1, end=hidden_states.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(hidden_states.size())
                    .float().to(hidden_states.device)
                )
            assert weights.shape == hidden_states.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        elif self.pooling_strategy == 'sowe':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            output_vectors.append(sum_embeddings)

        else:
            raise Exception("Wrong pooling strategy!")
        output_vector = torch.cat(output_vectors, 1)
        return output_vector
```

---

### 🔹 Latent Variable Modeling

The key innovation is the **Dual Latent Variable Schema**, where the encoder output is split into:

- **Z_sem (Semantic Latent Variable)**  
  Captures *meaning/context*

- **Z_syn (Syntactic Latent Variable)**  
  Captures *structure/grammar*

These are derived from:
- Semantic hidden states
- Syntactic hidden states

Then combined into a unified latent variable:

```
sem_hid, syn_hid = pooled[:,:512], pooled[:,512:]
sem_mu, sem_logvar = self.sem_mu(sem_hid), self.sem_logvar(sem_hid)
syn_mu, syn_logvar = self.syn_mu(syn_hid), self.syn_logvar(syn_hid)

sem_z = self.reparameterize(sem_mu, sem_logvar)
syn_z = self.reparameterize(syn_mu, syn_logvar)
z = sem_z * syn_z

sem_kl = self.regularization_loss(sem_mu, sem_logvar)
syn_kl = self.regularization_loss(syn_mu, syn_logvar)
kl_loss = (sem_kl+syn_kl)/2
```

---

## 3. 🔁 Decoder (Text Generation)

### 🔹 Input to Decoder

- The **simple sentence** is fed into the decoder:
  - Token Embedding
  - Positional Encoding

---

### 🔹 BART Decoder Block

The decoder uses:

- **Self-attention**
- **Cross-attention with encoder latent representation (Z)**

In cross-attention:
- Queries (Q) come from decoder
- Keys (K) and Values (V) come from encoder + latent variables

---

### 🔹 Output Generation

- Hidden states → Output probabilities
- Decoding strategy: **Beam Search**
- Final output: **Generated simplified sentence**

---

# 🎯 Parameter Settings

Based on the current implementation, the model uses the following parameters:

## 🔹 VAEBartForConditionalGeneration Parameters
```json
{
  "latent_dim": 512,
  "pooling_strategy": "mean",
  "enc_output": "last"
}
```

## 🔹 Learning Parameters
```json
{
  "generation": {
    "max_length": 70,
    "min_length": 10,
    "no_repeat_ngram_size": 2,
    "num_beams": 5
  },
  "training": {
    "num_epochs": 2,
    "learning_rate": 1e-05
  }
}
```

---

## ▶️ How to Run

You can run the project with the default parameters using:

```bash
python run.py
```

---

## ⚙️ How to Customize Parameters

If you want to modify the parameters, you can do so by updating the relevant parts of the code as described below.

---

### 1. 🧠 Modify Model Parameters

To change the **VAEBartForConditionalGeneration** parameters (`latent_dim`, `pooling_strategy`, `enc_output`), edit the `vaebart.py` file:

```python
class VAEBartModel(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config, latent_dim=512, pooling_strategy='mean', enc_output='last'):
        # You can adjust latent_dim, pooling_strategy, and enc_output here
        super().__init__(config)
```

---

### 2. 📊 Modify Generation Parameters

To adjust generation settings, edit the `run.py` file:

```python
pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(
    pretrained_model_name,
    model_cls=VAEBartForConditionalGeneration
)

# You can adjust the generation parameters here
hf_config.max_length = 70
hf_config.min_length = 10
hf_config.no_repeat_ngram_size = 2
hf_config.num_beams = 5
```

---

### 3. 🚀 Modify Training Parameters

To change training-related parameters, update the training section in `run.py`:

```python
learn = Learner(
    dls,
    model,
    opt_func=partial(Adam),
    loss_func=CrossEntropyLossFlat(),
    cbs=learn_cbs,
    splitter=partial(blurr_seq2seq_splitter, arch='bart'),
)

learn.freeze()
learn.summary()
learn.lr_find(suggest_funcs=[minimum, steep, valley, slide])

# You can adjust the number of epochs and learning rate here
learn.fit_one_cycle(2, lr_max=1e-05, cbs=fit_cbs)
```

---

## 💡 Notes

- Make sure any changes to `latent_dim` are consistent with the model architecture.
- Adjusting generation parameters will affect the quality and diversity of generated text.
- Training parameters such as learning rate and epochs can significantly impact model performance.
