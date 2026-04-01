"""
Evaluation dataset — 20 question/answer pairs based on foundational NLP papers.
These are grounded in well-known papers you should have in your data/papers/ folder:
- Attention Is All You Need (Transformer)
- BERT paper
- RAG paper (Lewis et al.)
- QLoRA paper
- GPT-2 / GPT-3 papers

The 'ground_truth' field is the reference answer RAGAS uses to score your RAG system.
"""

EVAL_QUESTIONS = [
    {
        "question": "What is the attention mechanism in the Transformer model?",
        "ground_truth": "The attention mechanism in the Transformer maps a query and a set of key-value pairs to an output, computed as a weighted sum of the values where weights are computed by a compatibility function of the query with the corresponding keys."
    },
    {
        "question": "What does BERT stand for and what is it?",
        "ground_truth": "BERT stands for Bidirectional Encoder Representations from Transformers. It is a language representation model designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context."
    },
    {
        "question": "What is the difference between encoder and decoder in the Transformer?",
        "ground_truth": "The encoder maps an input sequence to a sequence of continuous representations. The decoder takes those representations and generates an output sequence one element at a time in an auto-regressive manner."
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention linearly projects queries, keys and values h times with different learned projections, performs attention in parallel on each, then concatenates and projects the results, allowing the model to attend to information from different representation subspaces at different positions."
    },
    {
        "question": "What training objectives does BERT use?",
        "ground_truth": "BERT uses two training objectives: Masked Language Model (MLM) where random tokens are masked and predicted, and Next Sentence Prediction (NSP) where the model predicts whether two sentences are consecutive."
    },
    {
        "question": "What is Retrieval Augmented Generation (RAG)?",
        "ground_truth": "RAG is a method that combines parametric memory (LLM) with non-parametric memory (retrieved documents) to generate responses. It retrieves relevant documents using a dense retriever and conditions the generator on both the input and retrieved documents."
    },
    {
        "question": "What is positional encoding in the Transformer?",
        "ground_truth": "Positional encoding adds information about the position of tokens in the sequence since the Transformer contains no recurrence or convolution. Sine and cosine functions of different frequencies are used to generate position encodings."
    },
    {
        "question": "What is QLoRA?",
        "ground_truth": "QLoRA is an efficient fine-tuning approach that backpropagates gradients through a frozen 4-bit quantized pretrained language model into Low Rank Adapters (LoRA), reducing memory usage enough to fine-tune a 65B parameter model on a single 48GB GPU."
    },
    {
        "question": "What is the role of the feed-forward network in the Transformer?",
        "ground_truth": "Each encoder and decoder layer contains a fully connected feed-forward network applied to each position separately and identically, consisting of two linear transformations with a ReLU activation in between."
    },
    {
        "question": "How does BERT handle downstream tasks?",
        "ground_truth": "BERT handles downstream tasks by fine-tuning with task-specific inputs and outputs. A classification token [CLS] is added at the beginning and used as the aggregate representation for classification tasks."
    },
    {
        "question": "What is the scaled dot-product attention formula?",
        "ground_truth": "Scaled dot-product attention is computed as Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V, where Q, K, V are queries, keys and values, and d_k is the dimension of keys used for scaling."
    },
    {
        "question": "What is LoRA and how does it work?",
        "ground_truth": "LoRA (Low-Rank Adaptation) freezes pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks."
    },
    {
        "question": "What datasets were used to pre-train BERT?",
        "ground_truth": "BERT was pre-trained on the BooksCorpus (800M words) and English Wikipedia (2,500M words), concatenating all Wikipedia documents and using only the text passages."
    },
    {
        "question": "What is the difference between BERT base and BERT large?",
        "ground_truth": "BERT Base has 12 transformer layers, 768 hidden dimensions and 12 attention heads with 110M parameters. BERT Large has 24 transformer layers, 1024 hidden dimensions and 16 attention heads with 340M parameters."
    },
    {
        "question": "Why is the Transformer architecture faster to train than RNNs?",
        "ground_truth": "Transformers are faster to train than RNNs because self-attention operations can be parallelised across all positions, whereas RNNs process sequences sequentially. The path length between positions is also constant rather than linear."
    },
    {
        "question": "What is dropout and how is it used in the Transformer?",
        "ground_truth": "Dropout is a regularisation technique that randomly sets units to zero during training. In the Transformer, dropout is applied to the output of each sub-layer before adding to the residual connection, and to the attention weights."
    },
    {
        "question": "What is the key advantage of RAG over pure parametric models?",
        "ground_truth": "RAG can access and update non-parametric memory at test time without retraining, allowing it to incorporate new knowledge, provide citations for its answers, and reduce hallucination by grounding answers in retrieved documents."
    },
    {
        "question": "What is layer normalisation in the Transformer?",
        "ground_truth": "Layer normalisation is applied after each sub-layer in the Transformer using a residual connection: LayerNorm(x + Sublayer(x)). It normalises across the features of each individual training example."
    },
    {
        "question": "What quantisation technique does QLoRA use?",
        "ground_truth": "QLoRA uses 4-bit NormalFloat (NF4) quantisation, which is information theoretically optimal for normally distributed weights, along with double quantisation to reduce memory footprint and paged optimizers to manage memory spikes."
    },
    {
        "question": "What is the WordPiece tokenisation used in BERT?",
        "ground_truth": "WordPiece tokenisation splits words into subword units from a learned vocabulary of 30,000 tokens. Unknown words are split into known subword pieces, with continuation subwords marked with ## prefix."
    }
]
