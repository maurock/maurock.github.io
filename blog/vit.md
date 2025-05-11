---
layout: post
title: Building a Vision Transformer from Scratch
subtitle: From zero to a working Vision Transformer, understanding every step
description: Mauro Comi's academic profile
date: 8 April 2025  
---

<!-- <div class="toc">
    <strong>Table of Content</strong>
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li>Seeing the World Through Transformers</li>
        <ul style="margin: 10px 0; padding-left: 20px;">        
            <li>Step 1: Patchifying the Image, From Pixels to Patches</li>
            <li>Step 2: Patch Embedding</li>
        </ul>
    </ul>
</div> -->

<iframe width="100%" height="315" src="https://www.youtube.com/embed/ywzPAurbc1s?si=1LaR3CTa9II-mUfr" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Think about an image of a squirrel floating in space. How can a computer understand what the image represents? What if we could teach a machine to *read* an image more like how we read text, by looking at meaningful chunks and understanding their relationships? **Introducing Vision Transformers**. Transformers power all the latest advances in AI, from Gemini to ChatGPT, Claude, LLaMa, and pretty much every big model you've heard of in the last two years. They are known to be ground-breaking language models, but they are also extremely powerful at solving computer vision tasks.

In this post, we'll understand what a Vision Transformer is and how it works in detail. We'll build and train a Vision Transformer from scratch to classify images. I'll focus on intuition, explaining the *why* behind each component, and walk through the code step-by-step. 
I have included some <span style='background-color: #ffdfba91'>quizzes</span> and <span style='background-color: #c3b1e16b;'>recaps </span> along the way to reinforce understanding - spaced repetition! 🙌 

We'll build and train using **Google's JAX** and the new [NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) library. NNX makes building complex models like ViTs more explicit and Pythonic. **Why JAX/NNX**? JAX is extremely efficient thanks to `jit` (just-in-time) compilation and fast automatic differentiation (`grad`). It also runs seamlessly on CPU/GPU/TPU. This great post covers its benefits: [PyTorch is Dead. Long Live Jax](https://neel04.github.io/my-website/blog/pytorch_rant/) (*Disclaimer*: I wouldn't say PyTorch is dead, I love Pytorch. But JAX is equally cool!).NNX is the new deep learning framework for JAX, which is currently [Google's recommended framework](https://flax.readthedocs.io/en/latest/why.html) and provides a more familiar object-oriented programming model (like PyTorch) while retaining JAX's functional principles.

**Important resources**:

<img src="https://openmoji.org/data/color/svg/1F4DC.svg" alt="Github" class="icon"> Original paper: 
[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), Dosovitskiy et al.<br>
<img src="https://openmoji.org/data/color/svg/E045.svg" alt="Github" class="icon"> [GitHub repo for this post]() 

<h3><span style="color: rgb(215, 215, 215);">#</span> Seeing the World Through Transformers</h3> 
Remember our squirrel image? How can a Transformer, designed for text sequences, process it? The core idea is simple:

- **Patchify**: We first divide the image into smaller, fixed-size patches.
- **Flatten & Embed**: Each patch is just a collection of raw pixel data. To make sense of it, we need to convert each patch into a meaningful vector, or *embedding*. We flatten each patch and linearly project it into an embedding vector.
- **Add Position Info**: A Transformer doesn't inherently know patch order. It processes all patches in parallel. We add positional embeddings to retain spatial information.
- **CLS Token**: We add a special classification token embedding at position 0 in the embedding sequence.
- **Attention Magic**: Each embedding "looks" at all the other embeddings and decides which ones are most important to understand its own context.
- **Classify**: We use the output corresponding to the CLS token to predict the image's class.

<div style="text-align: center;">
    <img src="../img/blog/vit/vit.png" style="width: 400px; margin: auto; text-align: center;" alt="Image of the ViT pipeline form the ViT paper. It shows the same steps outlined above"><br>
    <em>Overview of the main steps in a Vision Transformer. Source: <a href="https://arxiv.org/abs/2010.11929">original paper</a></em>
</div>
<br>

This is a high-level overview of the Vision Transformer architecture.
Let's build each part.

<h4><span style="color: rgb(215, 215, 215);">##</span> Step 1: Patchifying the Image, From Pixels to Patches</h4> 
<div style="text-align: center; margin: 10px 0">
<video autoplay loop muted playsinline width="100%" height="400">
    <source src="../img/blog/vit/patch.mp4" type="video/mp4">
</video>
</div>

First, we need to break the image down into patches. Think of this like cutting a photo into a grid of smaller squares.
Instead of processing pixel by pixel like a Convolutional Neural Network does, we treat chunks of the image as the fundamental units.
This mimicks what happens in a sentence, where the fundamental input units are individual words (technically, chunks of words or *tokens*).
Slicing the image into tiny patches allows the Transformer to later find relationships between different parts of the image.

We use the einops library for elegant tensor manipulation. Einops is beautiful, but can be cryptic. Here's what we do below in the `patchify_image()` function:
- We want to rearrange the pattern from `(Batch, Height, Width, Channels)` to `(Batch, Num_Patches`, `Patch_Dimension`
- `"... (h p1) (w p2) c -> ... (h w) (p1 p2 c)"`: 
    - `(h p1)`: The Height dimension is composed of `h` patches, each of height `p1` (patch_size).
    - `(w p2)`: The Width dimension is composed of `w` patches, each of width `p2` (patch_size).
    - `c`: The Channels dimension (in our case `c`=3, since it's an RGB image).
- `-> ... (h w)`: The output sequence length is `h * w` (total number of patches, N).
- `(p1 p2 c)`: Each element in the sequence is a flattened patch of size `p1 * p2 * c` (Patch Dimension, D_patch).

<pre class="code-block-pastel"><code class="language-python">
    # From transformer.py
    import einops
    import jax.numpy as jnp
    import functools
    from flax import nnx
    import jaxtyping as jt
    
    Float = jt.Float
    Array = jt.Array
    
    @functools.partial(jax.jit, static_argnames=("patch_size",))
    def patchify_image(
        image: Float[Array, "B H W C"], patch_size: int
    ) -> Float[Array, "B N D_patch"]: # N = Num Patches, D_patch = Patch Dimension
        # Rearranges image from (Batch, Height, Width, Channels)
        # to (Batch, Num_Patches, Patch_Height * Patch_Width * Channels)
        patches = einops.rearrange(
            image,
            "... (h p1) (w p2) c -> ... (h w) (p1 p2 c)", # h, w are counts of patches
            p1=patch_size, # p1 = patch height
            p2=patch_size, # p2 = patch width
        )
        return patches
    </code>
</pre>
This function takes a batch of images `(B, H, W, C)` and reshapes it into a batch of sequences of flattened patches `(B, NumPatches, PatchSize*PatchSize*C)`.

Assuming:
- input image is `(1, 224, 224, 3)` and patch_size is 16
- then the output shape would be `(1, (224/16)*(224/16), 16*16*3)` = `(1, 196, 768)`

The decorator `@functools.partial(jax.jit, static_argnames=("patch_size",))` applied to the function makes sure that JAX compiles this function for massive speedups on GPUs/TPUs. The variable `static_argnames=("patch_size",)` tells the JAX compiler that `patch_size` is *static*: JAX doesn't know that, since `patch_size` is not a JAX tensor, and compilation only works on static shapes.

<h4><span style="color: rgb(215, 215, 215);">##</span> Step 2: Patch Embedding</h4> 

Raw pixel patches aren't ideal for a Transformer.
Pixel colors are simply intensity values, telling us the contribution of the <span style="color:red">Red</span>, <span style="color:green">Green</span>, and <span style="color:blue">Blue</span> channels.
Pixels do not encode semantic information, and they are very sensitive to noise and minor shifts in lighting. 
We need to map pixels to more useful, richer representations: the goal is to train our model to extract useful features for the task at hand.
In other words, we need to project raw pixel colors into a learned embedding space, similarly to how words are embedded into high dimensional vectors in NLP.

We use a standard Linear layer, a simple matrix multiplication + bias, to transform each flattened patch (e.g., 768 dimensions if patch size is 16x16 and 3 channels) into a desired embedding dimension (`dim_emb`, often 768 or higher). This layer learns to extract useful features from the raw patches during training.
We define an `nnx.Module` for this.
NNX encourages organizing parameters and logic together within modules following an object-oriented style.
We define a `PatchEmbedding` class inheriting from `nnx.Module`. Notice that `nnx.Linear` requires `rngs` (random generators). This is a key aspect of JAX/NNX: randomness (for initialization) must be explicitly handled via Random Number Generator keys to ensures reproducibility.


<pre class="code-block-pastel"><code class="language-python">
# From transformer.py
class PatchEmbedding(nnx.Module):
    def __init__(self, patch_size: int, out_features: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.out_features = out_features 
        self.channels = 3 # Assuming RGB images
        # A linear layer to project flattened patches
        self.linear = nnx.Linear(
            in_features=self.patch_size**2 * self.channels,
            out_features=self.out_features,
            rngs=rngs # NNX requires explicit RNG handling
        )

    def __call__(self, inputs: Float[Array, "B H W C"]) -> Float[Array, "B NP D"]:
        # 1. Patchify the image using the function from Step 1
        patches = patchify_image(inputs, self.patch_size) # (B, N, P*P*C)
        # Apply the learned linear projection
        projection = self.linear(patches) # (B, N, D), where D=out_features
        return projection
    </code>
</pre> 
This module takes the raw image, patchifies it internally using our previous function, and then passes the patches through a `nnx.Linear` layer to get the final patch embeddings.

<div class="quiz">
    <strong>Quiz</strong><br>
    <p>If an image is 32x32 pixels and the patch size is 8x8, how many patches will be generated?</p>
    <details>
    <summary>
    Click for the answer
    </summary>
        <p><i>(32/8) * (32/8) = 4 * 4 = 16 patches</i></p>   
    </details>
</div>

<div class="quiz">
    <strong>Quiz</strong><br>
        <p>Why do we use Linear Projection from raw pixels to patch embeddings?</p>
    <details>
    <summary>
    Click for the answer
    </summary>
        <p><i>Raw pixel patches (such as the 768 numbers extracted by flattening image patches) are not the most informative representation. A linear layer  projects these raw patches into a different vector. This projection is learned during training, allowing the model to find the most useful, compact representation for each patch, focusing on important features.</i></p>  
    </details>
</div>

<h4><span style="color: rgb(215, 215, 215);">##</span> Step 3: Positional Embeddings</h4> 
Standard Transformers are *permutation-invariant*, which means that they treat input elements as an unordered set, not a sequence. In other words, they process patches in parallel, without any information on *where* the patch is. But for images, the location of a patch matters! Sky above a tree depicts a landscape, while sky below a tree suggests a lake and its reflection. We need to give the model information about each patch's original location. Instead of simply feeding in the raw index of a patch (like patch #1, #2, etc.), we use **positional embeddings**. These are learned vectors, one for each possible patch position. Think of them as encoding the "address" of each patch into a vector, allowing the model to understand the image's structure. The vector has the same dimensionality of the patch embeddings, so that they can be added together.

**Why don't we simply use the raw index?** Using learnable embeddings (`nnx.Param`) gives the model maximum flexibility, because these embeddings can learn the optimal way to represent position for the specific task and dataset. They can learn and represent complex spatial relationships: for example, patches that are close together might have more similar embeddings.

<pre class="code-block-pastel"><code class="language-python">
# From transformer.py
class PositionalEmbedding(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        # in_features = number of patches + 1 (for CLS token, see next step)
        # out_features = embedding dimension
        self.in_features = in_features
        self.out_features = out_features
        key = rngs.params()
        # Learnable parameter for positional embeddings
        self.pos_embedding = nnx.Param(
            # Shape: (NumPatches + 1, EmbeddingDim)
            jax.random.normal(key=key, shape=(self.in_features, self.out_features)) * 0.02
        )

    def __call__(self) -> Float[Array, "1 M D"]: # M = NumPatches + 1
        # Simply returns the learned embedding table
        return self.pos_embedding
    </code>
</pre> 

We create a learnable `nnx.Param` tensor `pos_embedding`. Its shape is `(NumPatches + 1, EmbeddingDim)`. We simply add this to our patch embeddings. The + 1 is for the `[CLS]` classification token, which we'll see now.

<h4><span style="color: rgb(215, 215, 215);">##</span> Step 4: The CLS Token</h4> 

Inspired by [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), ViTs add an extra learnable embedding to the sequence at position 0, called the `[CLS]` (classification) token. 
Think of this token as a special "representative" embedding. 
This token has no direct input from the image pixels; it starts as a learned placeholder. 
The idea is that through the self-attention mechanism (coming next!), this CLS token acts like a 'sponge', aggregating information from all patch embeddings.
At the end, we can just use the output embedding corresponding to this CLS token to make the final image classification decision. 
It simplifies the architecture, providing a single vector ready for the classifier head.

<pre class="code-block-pastel"><code class="language-python">
# From transformer.py
# Inside VisionTransformer.__init__
self.cls_embedding = nnx.Param(
    # A single learnable embedding vector of dimension dim_emb
    jax.random.normal(key=self.rngs.params(), shape=(1, 1, self.dim_emb)) * 0.02
)

# Function to prepend the CLS token
def add_classification_token(
    cls_embedding: Float[Array, "1 1 D"], embeddings: Float[Array, "B N D"]
) -> Float[Array, "B M D"]: # M = N + 1
    # Repeat CLS token for each item in the batch
    cls_embedding_batch = jnp.tile(
        cls_embedding, (embeddings.shape[0], 1, 1) # Shape becomes (B, 1, D)
    )
    # Concatenate CLS token at the beginning of the patch sequence
    return jnp.concatenate([cls_embedding_batch, embeddings], axis=1) # Axis 1 is the sequence dim
    </code>
</pre> 

We define `cls_embedding` as another learnable `nnx.Param`. The `add_classification_token()` function duplicates it for the batch size and pre-appends it to the sequence of patch embeddings before adding the positional embeddings.

<div class="note">
    <strong>Recap so far:</strong><br>
        Image -> Patches (<code>patchify_image()</code>) <br> 
        Patches -> Patch Embeddings (<code>PatchEmbedding</code>)<br>
        Prepend CLS Token (<code>add_classification_token()</code>, <code>cls_embedding</code>)<br>
        Add Positional Info (<code>PositionalEmbedding</code>, <code>pos_embedding</code>)<br>
</div>

Now we have a sequence of embeddings ready for the Transformer Encoder! Shape: `(Batch, NumPatches + 1, EmbeddingDim)`.

<h4><span style="color: rgb(215, 215, 215);">##</span> Step 5: Attention and the Transformer Encoder</h4> 
<div style="text-align: center; margin: 10px 0;">
<video autoplay loop muted playsinline width="100%" height="400">
    <source src="../img/blog/vit/attention.mp4" type="video/mp4">
</video>
</div>
**Attention: this is the heart of the ViT**. Let's start with a strong intuition on the attention mechanism. Imagine each patch embedding (and the CLS token) wanting to update itself. To do this, each patch embedding first creates three vectors from itself: a Query (`Q`) representing its current state and what it's looking for, a Key (`K`) representing what it can offer, and a Value (`V`) representing the actual content it wants to share. 

<div class="note">
Attention as a conversation: patch embeddings talk to each other to understand their context. Q="What I need", K="What I have", V="What I'll share if you pay attention".
</div>

The attention procedures compares its Query with the Keys of all other tokens (including itself). High similarity (*dot product*) means high relevance. These similarities become **attention scores** (normalized via softmax). Finally, each patch embedding updates its representation by taking a weighted sum of all tokens' Values, weighted by the attention scores. It pays more attention to tokens relevant to its query. 

In other words, self-attention allows each token (patch embedding or CLS token) to "look" at all other tokens in the sequence and decide which ones are most relevant to it. It calculates attention scores, determining how much "focus" to place on other tokens when updating its own representation. 
In practice, we usually don't use a single set of `Q`, `K`, `V` matrices, but multiple sets. This is called **Multi-Head Self Attention**. *Multi-Head* means performing self-attention multiple times in parallel (*heads*), each with different learned `Q`, `K`, `V` projection matrices. As a result, we allow the model to capture different types of relationships simultaneously (e.g. one head might focus on texture, another on shapes, another on long-range dependencies). 

The attention magic happens in the *Transformer Encoder*. The Transformer Encoder is typically a stack of identical Encoder Blocks. Each block processes the sequence of embeddings, allowing different patches (and the CLS token) to interact and exchange information.

<div style="display: flex; align-items: center; gap: 20px;">
    <!-- Image on the left -->
    <div style="flex-shrink: 0;">
        <img src="https://theaisummer.com/static/aa65d942973255da238052d8cdfa4fcd/7d4ec/the-transformer-block-vit.png" 
             style="width: 200px;" 
             alt="Image of the ViT pipeline from the ViT paper. It shows the same steps outlined above"><br>
        <em>Encoder Block. <a href="https://theaisummer.com/">Source</a></em>
    </div>
    <!-- Text on the right -->
    <div>
        An Encoder Block usually contains:
        <ul>
            <li>Layer Normalization</li>
            <li>Multi-Head Self-Attention (MHSA)</li>
            <li>Skip Connection</li>
            <li>Layer Normalization</li>
            <li>Feed-Forward Network (MLP)</li>
            <li>Skip Connection</li>
        </ul>
    </div>
</div>
<br>

Before looking at the code, here are the key components that we need to develop:

- `MultiHeadAttention`: This module calculates the scaled dot-product attention across multiple heads. It uses linear layers (`nnx.Linear`) to create Query, Key, and Value vectors from the input embeddings. The split_attention_heads function reshapes these for parallel head processing. The core calculation `jnp.einsum('bhnd,bhmd->bhnm', q, k)` computes the dot products between queries and keys for all heads and batch items efficiently.
    Specirfically, the `MultiHeadAttention` block does the following:  
    - Takes input `x` of shape `(B, N, D)`
    - Projects `x` into Query (`Q`), Key (`K`), Value (`V`) vectors using Linear layers.
    - Splits `Q`, `K`, `V` into multiple heads: `(B, N, D) -> (B, num_heads, N, D_head)`
    - Calculates attention scores: `softmax(Q @ K^T / sqrt(d_k))`
    - Applies scores to `V`: `Attention = scores @ V`
    - Concatenates head outputs and projects back to `dim_emb` using another Linear layer.
- `LayerNorm`: Normalizes the activations within each layer, stabilizing training.
- `MLP`: A simple two-layer feed-forward network applied independently to each token's embedding. It helps transform the features learned by the attention mechanism.
- `Skip Connections`: Adding the input (`embeddings_in` or `x_skip1`) back to the output of a sub-layer. Crucial for training deep networks by preventing vanishing gradients.

Let's break down the EncoderBlock:
<pre class="code-block-pastel">
    <code class="language-python">
    # From transformer.py
    class EncoderBlock(nnx.Module):
        def __init__(self, num_heads: int, dim_emb: int, rngs: nnx.Rngs):
            self.num_heads = num_heads
            self.dim_emb = dim_emb
            # First Layer Norm
            self.layer_norm1 = LayerNorm()
            # Multi-Head Self-Attention
            self.multi_head_attention = MultiHeadAttention(
                num_heads=self.num_heads,
                dim_emb=self.dim_emb, # Input/Output dimension
                dim_qkv=self.dim_emb, # Dimension for Q, K, V (can differ)
                dim_out=self.dim_emb, # Final output dimension after attention
                rngs=rngs
            )
            # Second Layer Norm
            self.layer_norm2 = LayerNorm()
            # Feed-Forward Network
            self.mlp = nnx.Sequential(
                nnx.Linear(in_features=self.dim_emb, out_features=self.dim_emb * 4, rngs=nnx.Rngs(params=rngs.params())),
                nnx.gelu, # Activation function
                nnx.Linear(in_features=self.dim_emb * 4, out_features=self.dim_emb, rngs=nnx.Rngs(params=rngs.params()))
            )

        def __call__(self, embeddings_in: Float[Array, "B N D"]) -> Float[Array, "B N D"]:
            # Attention path
            x_norm1 = self.layer_norm1(embeddings_in)
            attention_out = self.multi_head_attention(x_norm1)
            # Add and Norm (Skip connection 1)
            x_skip1 = attention_out + embeddings_in # Residual connection
            # MLP path
            x_norm2 = self.layer_norm2(x_skip1)
            mlp_out = self.mlp(x_norm2)
            # Add and Norm (Skip connection 2)    
            x_skip2 = mlp_out + x_skip1 # Residual connection

            return x_skip2
    </code>
</pre>


The full VisionTransformer stacks multiple `EncoderBlock` instances using `nnx.Sequential`.

<div class="quiz">
    <strong>Quiz</strong><br>
    <p>Why are skip connections important in deep networks like Transformers?</p>
    <details>
    <summary>
    Click for the answer
    </summary>
        <p><i>They help gradients flow better during backpropagation, preventing them from vanishing and enabling the training of deeper models.</i></p>   
    </details>
</div>

<h4><span style="color: rgb(215, 215, 215);">##</span> Step 6: Classification Head</h4> 

After passing the sequence through all the Encoder Blocks, we need to get our final classification. It's time to shine for the CLS token we added in Step 3.

The output embedding corresponding to the first token (our CLS token) is assumed to have aggregated the necessary information from the entire image via the self-attention process. We pass this single embedding through a final Linear layer to get scores for each possible class.

<pre class="code-block-pastel">
    <code class="language-python">
    # From transformer.py
    # Inside VisionTransformer.__init__
    self.final_head = nnx.Linear(
        in_features=self.dim_emb,
        out_features=self.num_classes, # e.g., 10 for CIFAR-10, 1000 for ImageNet
        rngs=self.rngs
    )

    # Inside VisionTransformer.__call__
    def __call__(self, x: Float[Array, "B H W C"]) -> tuple[Float[Array, "B C"], Float[Array, "B N D"]]: # C = num_classes
        # ... (patching, embedding, positional, CLS token logic) ...
        patch_embeddings = self.patch_embedding_fn(x)
        embeddings = add_classification_token(self.cls_embedding, patch_embeddings)
        embeddings = embeddings + self.pos_embeddings # Positional applied AFTER CLS token

        # Pass through all encoder blocks
        vision_embeddings = self.encoder_blocks(embeddings) # (B, NumPatches + 1, D)

        # Select the output of the CLS token
        cls_output = vision_embeddings[:, 0, :] # Shape: (B, D)

        # Pass CLS output through the final classification head
        output_logits = self.final_head(cls_output) # Shape: (B, num_classes)

        # Return logits and optionally the full sequence output for other uses
        return output_logits, vision_embeddings
    </code>
</pre>
We select the output corresponding to the first token (index 0, `vision_embeddings[:, 0, :]`) and pass it to a final `nnx.Linear` layer (`self.final_head`) whose output size is the number of classes. This final linear layer maps the aggregated representation from the CLS token (`dim_emb`) to the number of output classes (`num_classes`), producing the raw prediction scores (logits) for each class.

<h3><span style="color: rgb(215, 215, 215);">#</span> Training the ViT</h3> 

We have the model architecture, now it's time to train it. We'll use the Hugging Face dataset [snacks](https://huggingface.co/datasets/Matthijs/snacks).

We train the network via supervised learning. We show the model many labeled images of snacks. For each image, the model makes a prediction. We compare this prediction to the true label using a loss function (cross-entropy for classification). This tells us how "wrong" the model was. We then use an optimizer (`Adam`) to slightly adjust all the model's learnable parameters (weights in Linear layers and embeddings) in a direction that reduces the loss. Repeat this thousands of times, and the model learns!

The code used for training is in `train.py` and `dataset.py`.  Here are the main components of our training procedure:

- Dataset Handling (`dataset.py`): The code uses the datasets library to load and preprocess data (e.g., resizing images). It defines a `HF_Dataset` class to manage loading data (from Hugging Face Hub or from disk) and applying transformations like resizing. It converts the data to JAX format (`.with_format("jax")`) for compatibility.
- Loss Function (`train.py`): Standard cross-entropy loss is used to compare the model's output logits with the one-hot encoded true labels. It measures the difference between the predicted probability distribution over classes and the true distribution (one-hot encoded label).e ("uses the gradients calculated via backpropagation to iteratively adjust model parameters to minimize the loss")
- Training Step (`train.py`): NNX provides `nnx.value_and_grad`, the core of NNX/JAX differentiation. It takes the loss function and returns a new function that, when called with the model and batch, computes both the loss value and the gradients of the loss with respect to the model's parameters (those marked with `nnx.Param`). `nnx.Optimizer` (`optax.adam`) uses the gradients calculated via backpropagation to iteratively adjust model parameters to minimize the loss. `optimizer.update(grads)` applies the computed gradients to update the model parameters. `nnx.jit` compiles the `train_step()` for significant speedup on GPUs/TPUs accelerators, as we saw earlier.

<pre class="code-block-pastel">
    <code class="language-python">
# From train.py
import optax
from flax import nnx
import jax

# ... (cross_entropy and loss_fn definitions) ...

@nnx.jit # Compile the training step
def train_step(
    model: transformer.VisionTransformer, # Model state (includes parameters)
    optimizer: nnx.Optimizer,          # Optimizer state (includes optimiser params like momentum)
    batch: dict[str, Array],
) -> tuple[transformer.VisionTransformer, nnx.Optimizer, Array]:
    """Perform a single training step."""

    # Calculate loss and gradients w.r.t. model parameters
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)

    # Update model parameters using the optimizer
    optimizer.update(grads)

    # Return updated model state, optimizer state, and the loss value
    return model, optimizer, loss

def run_training(
    config: my_types.ConfigFile,
    model: transformer.VisionTransformer,
    my_data: Dataset, # Training data iterator
) -> transformer.VisionTransformer:
    """Run full training."""

    # Initialize the optimizer (wraps model and optax optimizer)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=config["learning_rate"]))

    print("Starting training...")
    for epoch in range(config['num_epochs']):
        total_loss = 0.0
        num_batches = 0
        for batch in my_data.iter(batch_size=config['batch_size']): # Assumes data provides batches
            # Perform one training step
            model, optimizer, loss_value = train_step(model, optimizer, batch)
            total_loss += loss_value
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Average Loss: {avg_loss:.4f}")

    print("Training finished.")
    return model # Return the trained model state
    </code>
</pre>
The `run_training()` function manages the entire training procedure: it initializes the nnx.Optimizer, loops through epochs and batches, calls the train_step, and logs the progress.

- `jax.jit`: Just-In-Time compilation makes Python code run incredibly fast on GPUs/TPUs.
- `jax.grad` (via `nnx.value_and_grad`): Automatic differentiation, which simplifies gradient calculation.
- `NNX`: Provides a more familiar, stateful object-oriented way to manage model parameters and state compared to pure functional Flax/JAX, making the code potentially easier to follow for those used to PyTorch/Keras, while retaining JAX's power. It requires explicit handling of random number generators (rngs) and organizes parameters within modules (nnx.Param, nnx.Linear).

<h4><span style="color: rgb(215, 215, 215);">#</span> Conclusion</h4> 

⭐ We made it! We've walked through the entire process of building a Vision Transformer from scratch using JAX and NNX. 

<div class="note">
<strong>We've seen how to:</strong><br>
<ul>
<li>Patchify images into sequences. <br></li>
<li>Create learnable Patch and Positional Embeddings. <br></li>
<li>Use a CLS token for classification. <br></li>
<li>Leverage Multi-Head Self-Attention within Transformer Encoder blocks to find relationships between patches. <br></li>
<li>Set up a training loop with loss functions and optimizers to teach the model. <br></li>
</ul>
</div>

While we've built a functional ViT, this is a baseline and many improvements have been made since 2021. However, the core concepts outlined here remain. Modern improvements include: using different patchifying strategies (e.g., overlapping patches), more efficient attention mechanisms, different positional encoding schemes (learned 2D, relative), and integrating convolutional layers early on (Hybrid ViTs). New architectures also focus on "increasing" the information content of the patch embeddings, making them more robust and *3D spatially aware*. 

Moreover, ViTs are powerful backbones for downstream tasks like object detection, segmentation, and even generative models like **Diffusion Transformers**. I'll cover these topics in future posts.


<br><br>

