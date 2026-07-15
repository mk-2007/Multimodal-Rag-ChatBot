# Multimodal RAG - Image Query Result

Source document: *Attention Is All You Need* (Vaswani et al., 2017)

**Query image:** `images\figure_1_page3.png`

**Question asked:** Explain this diagram

**Stage 1 - image described for search:**

> This image displays the architecture of the Transformer model, a deep learning neural network composed of an encoder stack on the left and a decoder stack on the right. Key components visible include "Input/Output Embedding" layers, "Positional Encoding," "Multi-Head Attention" blocks, "Feed Forward" layers, and "Add & Norm" operations, with the stacks repeated "Nx" times. The diagram illustrates the data flow from inputs and right-shifted outputs through these layers, ultimately producing "Output Probabilities" via "Linear" and "Softmax" layers.

**Retrieved context (shared vector store, mixed modalities):**

- `[0.831]` **figure** | `figure_1` | page 3 | _Figure 1: The Transformer - model architecture._
- `[0.775]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.756]` **figure** | `figure_3` | page 4 | _Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Atte_
- `[0.752]` **figure** | `figure_2` | page 4 | _Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Atte_
- `[0.743]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_

**Grounded answer (dual-grounded: image + retrieved context):**

This diagram, Figure 1 from the paper, illustrates the Transformer model architecture, which utilizes an encoder-decoder structure to process sequential data without recurrence.

**Encoder (Left side):**
*   The encoder consists of a stack of $N$ identical layers.
*   The input sequence passes through an "Input Embedding" and is combined with "Positional Encoding."
*   Each encoder layer contains two sub-layers: a "Multi-Head Attention" mechanism and a position-wise fully connected "Feed Forward" network.
*   As shown by the arrows and boxes, each sub-layer employs a residual connection followed by layer normalization (the "Add & Norm" blocks).

**Decoder (Right side):**
*   The decoder is also a stack of $N$ identical layers.
*   It takes the "Outputs (shifted right)" as input, which are processed through an "Output Embedding" and "Positional Encoding."
*   Each decoder layer contains three sub-layers:
    1.  A "Masked Multi-Head Attention" sub-layer, which prevents positions from attending to subsequent positions.
    2.  A second "Multi-Head Attention" sub-layer that performs attention over the final output of the encoder stack (indicated by the connection from the encoder to the decoder).
    3.  A "Feed Forward" network.
*   Like the encoder, each sub-layer in the decoder is followed by an "Add & Norm" operation.

**Final Output:**
*   The final output of the decoder stack passes through a "Linear" layer and a "Softmax" layer to generate the "Output Probabilities" for the sequence.
