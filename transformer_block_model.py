#  import and class definition - The code imports TensorFlow library as tf and defines a 
import tensorflow as tf 


#  MultiHeadSelfAttention class as a subclass of tf.keras.layers.Layer.
class MultiHeadSelfAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads): #  init method - The init method initializes the number of heads, the size of each 
    super(MultiHeadSelfAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads #  head (depth), and the dense layers for query, key, value, and context.
    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)
    self.dense = tf.keras.layers.Dense(units=d_model)

#  split_heads method - The split_heads method reshapes the inputs into a tensor with 
#  shape (batch_size, -1, self.num_heads, self.depth) and then transposes it to have 
#  the shape (batch_size, self.num_heads, -1, self.depth).

  def split_heads(self, inputs, batch_size): 
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

#  call method - The call method of the MultiHeadSelfAttention class performs the 
#  self-attention mechanism. It takes in query, key, value, and mask inputs and computes 
#  their dot product and then applies a softmax activation to obtain the attention scores. 
#  The context is then computed by multiplying the attention scores with the values. The 
#  context is then reshaped and passed through a dense layer to obtain the final output.
  
  def call(self, inputs):                          
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)
    scaled_attention = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(
        tf.cast(self.depth, tf.float32))
    if mask is not None:
      scaled_attention += (mask * -1e9)
    attention = tf.nn.softmax(scaled_attention, axis=-1)
    context = tf.matmul(attention, value)
    context = tf.transpose(context, perm=[0, 2, 1, 3])
    context = tf.reshape( 
        context, shape=(batch_size, -1, self.d_model))    
    outputs = self.dense(context)
    return outputs

#  PositionWiseFeedForward class - The PositionWiseFeedForward class is another subclass of 
#  tf.keras.layers.Layer that implements the feed-forward network. It has two dense layers 
#  feed_forward_1 and feed_forward_2, with the first dense layer having a ReLU activation.

class PositionWiseFeedForward(tf.keras.layers.Layer): 
  def __init__(self, d_model, ff_dim):
    super(PositionWiseFeedForward, self).__init__()
    self.d_model = d_model
    self.feed_forward_1 = tf.keras.layers.Dense(units=ff_dim, activation='relu')
    self.feed_forward_2 = tf.keras.layers.Dense(units=d_model)
  def call(self, inputs):
    x = self.feed_forward_1(inputs)
    x = self.feed_forward_2(x)
    return x

#  TransformerBlock class - The TransformerBlock class is a subclass of tf.keras.layers.Layer 
#  that implements a Transformer block. It has two sublayers: MultiHeadSelfAttention and 
#  PositionWiseFeedForward. It also contains layer normalization and dropout layers. The call 
#  method of this class applies the self-attention mechanism, feed-forward network, layer 
#  normalization, and dropout.

class TransformerBlock(tf.keras.layers.Layer):  
  def __init__(self, d_model, num_heads, ff_dim):
    super(TransformerBlock, self).__init__()
    self.att = MultiHeadSelfAttention(d_model, num_heads)
    self.ffn = PositionWiseFeedForward(d_model, ff_dim)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(0.1)

  def call(self, inputs):
    attn_output = self.att({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': None
    })
    
	#  Input and transformer - An input layer with shape (None,) is created and a TransformerBlock 
	#  object is created with d_model=512, num_heads=8, and ff_dim=2048.
	
	x = inputs + self.dropout(attn_output)       
    x = self.layernorm1(x)
    ffn_output = self.ffn(x)
    x = x + self.dropout(ffn_output)
    x = self.layernorm2(x)
    return x

#  model creation and compilation - A Keras model is created with the input layer and the transformer 
#  as its output. The model is compiled with Adam optimizer and binary cross-entropy loss.

inputs = tf.keras.layers.Input(shape=(None,))
transformer = TransformerBlock(d_model=512, num_heads=8, ff_dim=2048)
outputs = transformer(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

