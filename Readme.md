Transformer Block for Tensorflow
This repository contains the implementation of a Transformer Block in Tensorflow. A Transformer Block is a building block of the Transformer architecture, which is a type of deep neural network architecture used for processing sequential data, such as natural language processing (NLP).

Classes
- MultiHeadSelfAttention
- PositionWiseFeedForward
- TransformerBlock

MultiHeadSelfAttention
- This class implements the Multi-Head Self Attention mechanism of the Transformer architecture. It takes in a query, key, value and mask and performs a dot product of the query and key and applies a softmax to obtain the attention weights. The context is then computed as the weighted sum of the values.

PositionWiseFeedForward
- This class implements the Position-Wise Feed-Forward mechanism of the Transformer architecture. It takes in an input and passes it through two dense layers with a ReLU activation in between.

TransformerBlock
- This class combines the MultiHeadSelfAttention and PositionWiseFeedForward mechanisms to form a Transformer block. It applies the MultiHeadSelfAttention mechanism to the input and then passes it through the PositionWiseFeedForward. The output is then normalized and added to the input, before being passed through a dropout layer.

Model Summary
- The implementation includes a model definition, which takes in an input of shape (None,) and outputs the result of passing the input through the TransformerBlock. The model is compiled with the Adam optimizer and binary cross-entropy loss. Running model.summary() displays a summary of the model architecture.

To expand upon this code, one could consider the following possibilities:

- Implement the Encoder and Decoder blocks: The given code implements only a single block of the Transformer. To create a full Transformer model, you would need to stack multiple blocks in the Encoder and Decoder components of the model.
- Experiment with different hyperparameters: The code uses specific values for the number of heads (8), the model dimensionality (512), the feed-forward dimensionality (2048), and the dropout rate (0.1). One could experiment with different values to see the impact on the model's performance.
- Add the positional encoding: The Transformer architecture uses positional encoding to incorporate the position of each input element in the sequence. This can be implemented as an addition to the input sequence before it is fed into the Transformer.
