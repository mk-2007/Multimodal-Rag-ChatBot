# Multimodal RAG Demo Results

Source document: *Attention Is All You Need* (Vaswani et al., 2017)

## Query 1: What BLEU score did the Transformer (big) model achieve on the WMT 2014 English-to-German translation task, and how does it compare to GNMT + RL?

**Retrieved context (shared vector store, mixed modalities):**

- `[0.803]` **text** | `page 8 body text` | page 8 | _6 Results_
- `[0.785]` **text** | `page 7 body text` | page 7 | _5 Training_
- `[0.777]` **table** | `table_2` | page 8 | _Table 2 (see rendered page image table_2_page8.png)_
- `[0.737]` **text** | `page 8 body text` | page 8 | _6 Results_
- `[0.734]` **text** | `page 7 body text` | page 7 | _5 Training_

**Grounded answer:**

The Transformer (big) model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task.

Compared to GNMT + RL, the Transformer (big) model outperforms it; according to Table 2, the GNMT + RL model achieved a BLEU score of 24.6, while the GNMT + RL Ensemble achieved 26.30. The Transformer (big) model outperforms these previous state-of-the-art models (including ensembles) by more than 2.0 BLEU.

---

## Query 2: Describe the overall encoder-decoder architecture of the Transformer as shown in the model diagram -- what are the main blocks and how do they connect?

**Retrieved context (shared vector store, mixed modalities):**

- `[0.784]` **figure** | `figure_1` | page 3 | _Figure 1: The Transformer - model architecture._
- `[0.773]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.743]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.737]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.728]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_

**Grounded answer:**

The Transformer architecture consists of an encoder stack and a decoder stack, each composed of $N=6$ identical layers (Context 1, Context 2, Context 3).

### Encoder Architecture
*   **Input Processing:** The process begins with "Inputs" passing through an "Input Embedding" block, to which "Positional Encoding" is added (Context 1).
*   **Layers:** Each encoder layer consists of two sub-layers: a **Multi-Head Attention** block and a **Feed Forward** network (Context 1, Context 2).
*   **Connections:** Each sub-layer utilizes a residual connection followed by an "Add & Norm" (layer normalization) operation. The output of each sub-layer is defined as $\text{LayerNorm}(x + \text{Sublayer}(x))$ (Context 1, Context 3).

### Decoder Architecture
*   **Input Processing:** "Outputs (shifted right)" pass through an "Output Embedding" and are added to a "Positional Encoding" (Context 1).
*   **Layers:** Each decoder layer contains three sub-layers:
    1.  **Masked Multi-Head Attention:** This prevents positions from attending to subsequent positions to preserve the auto-regressive property (Context 1, Context 3).
    2.  **Multi-Head Attention:** This sub-layer receives its queries from the previous decoder layer and its keys and values from the final output of the encoder stack (Context 1, Context 5).
    3.  **Feed Forward:** A position-wise fully connected network (Context 1, Context 4).
*   **Connections:** Similar to the encoder, each sub-layer is followed by a residual connection and an "Add & Norm" layer (Context 1, Context 3).

### Final Output
The final output of the decoder stack passes through a **Linear** layer followed by a **Softmax** layer to produce "Output Probabilities" (Context 1). All sub-layers and embedding layers produce outputs of dimension $d_{model} = 512$ (Context 3).

---

## Query 3: What is multi-head attention and why is it used instead of a single attention function?

**Retrieved context (shared vector store, mixed modalities):**

- `[0.745]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.740]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.715]` **figure** | `figure_3` | page 4 | _Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Atte_
- `[0.697]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.694]` **text** | `page 6 body text` | page 6 | _4 Why Self-Attention_

**Grounded answer:**

Multi-head attention is a mechanism where queries, keys, and values are linearly projected $h$ times with different, learned linear projections into $d_k$, $d_k$, and $d_v$ dimensions. The attention function is then performed on these projected versions in parallel, after which the resulting outputs are concatenated and projected again to produce final values.

It is used instead of a single attention function because:
* **Joint Representation:** It allows the model to jointly attend to information from different representation subspaces at different positions. 
* **Avoiding Averaging:** With a single attention head, averaging inhibits the model's ability to attend to information from different subspaces.
* **Interpretability:** Individual attention heads can learn to perform different tasks and exhibit behavior related to the syntactic and semantic structure of sentences.

The total computational cost of this approach remains similar to that of single-head attention with full dimensionality because the dimension of each head is reduced.

---
