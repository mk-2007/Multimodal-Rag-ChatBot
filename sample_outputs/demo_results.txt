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

On the WMT 2014 English-to-German translation task, the **Transformer (big)** model achieved a BLEU score of **28.4**.

### Comparison to GNMT + RL:
* **GNMT + RL (single model):** The Transformer (big) model outperforms the GNMT + RL single model, which achieved a BLEU score of **24.6** (an improvement of 3.8 BLEU points).
* **GNMT + RL Ensemble:** The Transformer (big) model also outperforms the GNMT + RL Ensemble, which achieved a BLEU score of **26.30** (an improvement of 2.1 BLEU points). 

*(Source: Context 1, Context 2, and Table 2)*

---

## Query 2: Describe the overall encoder-decoder architecture of the Transformer as shown in the model diagram -- what are the main blocks and how do they connect?

**Retrieved context (shared vector store, mixed modalities):**

- `[0.784]` **figure** | `figure_1` | page 3 | _Figure 1: The Transformer - model architecture._
- `[0.773]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.743]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.737]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.728]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_

**Grounded answer:**

Based on the model diagram in **Figure 1** (supported by the text in Section 3), the Transformer architecture consists of an **encoder stack** on the left and a **decoder stack** on the right, connected via an attention mechanism, and culminating in an output stage. 

Here is how the main blocks are structured and connected:

### 1. The Encoder Stack (Left)
* **Input and Embedding:** The process begins with **Inputs**, which pass through an **Input Embedding** block. A **Positional Encoding** (represented by a wave symbol) is added to this embedding.
* **Encoder Layers ($N\times$ repetitions):** This combined signal enters a stack that repeats $N$ times (where $N = 6$). Each layer contains:
  * **Multi-Head Attention:** The signal branches, with one path splitting into three inputs entering the "Multi-Head Attention" block, and a residual connection bypassing it. Both paths merge at an **Add & Norm** layer.
  * **Feed Forward:** The output then splits again: one path enters a position-wise **Feed Forward** block, and a residual connection bypasses it. Both paths merge at a second **Add & Norm** layer.
* The final output of the encoder stack is sent directly to the decoder stack.

### 2. The Decoder Stack (Right)
* **Output and Embedding:** The target sequence, **Outputs (shifted right)**, passes through an **Output Embedding** block and has a **Positional Encoding** added to it.
* **Decoder Layers ($N\times$ repetitions):** This signal enters a stack that also repeats $N$ times (where $N = 6$). Each layer consists of three sub-layers:
  * **Masked Multi-Head Attention:** To prevent positions from attending to subsequent positions, the signal first passes through a "Masked Multi-Head Attention" block and an **Add & Norm** layer (utilizing a residual bypass).
  * **Encoder-Decoder Multi-Head Attention:** The signal then enters a second "Multi-Head Attention" block. This block acts as the connection point between the encoder and decoder: it receives **two inputs** (keys and values) directly from the final output of the encoder stack, and its **third input** (queries) from the decoder's preceding "Add & Norm" layer. This is followed by another **Add & Norm** layer with a residual connection.
  * **Feed Forward:** The signal then passes through a **Feed Forward** block, paired with a final residual bypass and an **Add & Norm** layer.

### 3. The Output Stage
* At the very top, the final output of the decoder stack passes through a **Linear** layer followed by a **Softmax** layer to generate the final **Output Probabilities**.

---

## Query 3: What is multi-head attention and why is it used instead of a single attention function?

**Retrieved context (shared vector store, mixed modalities):**

- `[0.745]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.740]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.715]` **figure** | `figure_3` | page 4 | _Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Atte_
- `[0.697]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.694]` **text** | `page 6 body text` | page 6 | _4 Why Self-Attention_

**Grounded answer:**

Based on the provided context, **multi-head attention** is an attention mechanism where, instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, the model:
1. Linearly projects the queries, keys, and values $h$ times using different, learned linear projections to $d_k$, $d_k$, and $d_v$ dimensions, respectively (Context 2).
2. Performs the attention function in parallel across these projected versions (yielding $d_v$-dimensional outputs) (Context 2).
3. Concatenates these outputs and projects them once more to produce the final values (Context 2, Figure 2).

**Why it is used instead of a single attention function:**
It is used because it allows the model to **jointly attend to information from different representation subspaces at different positions** (Context 1, Context 2). If a single attention head were used instead, averaging would inhibit this capability (Context 1, Context 2).

---
