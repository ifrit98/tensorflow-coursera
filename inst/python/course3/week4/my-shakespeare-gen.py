#! /usr/bin/python

# Character RNN for Text Generation
import tensorflow as tf

import numpy as np
import pandas as pd
import os
import json
import time


json_data = '/mnt/ipahome/georgej/external/leaf/data/shakespeare/data/all_data/all_data.json'
csv_data  = '/mnt/ipahome/georgej/shakespeare-data/Shakespeare_data.csv'
all_lines = '/mnt/ipahome/georgej/shakespeare-data/alllines.txt'

# Think this is what we want
text = open(all_lines, 'rb').read().decode(encoding='utf-8')
text_csv = pd.read_csv(csv_data)
# text_json = pd.read_json(json_data)

with open(json_data) as f:
  text_json = json.load(f)

local_data_drive = '/mnt/md0/georgej' # os.path.expanduser("~/")

path_to_file = os.path.join(local_data_drive, "shakespeare-data", "alllines.txt")

print(local_data_drive)
print(path_to_file)

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))


# Take a look at the first 250 characters in text
print(text[:250])


# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))



# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])


print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# Show how the first 13 characters from the text are mapped to integers
print ('{} --- characters mapped to int ---> {}'.format(repr(text[:13]), text_as_int[:13]))


# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)


# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(20):
  print(idx2char[i.numpy()])


# The batch method lets us easily convert these individual characters to 
# sequences of the desired size.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))




# For each sequence, duplicate and shift it to form the input and target 
# text by using the map method to apply a simple function to each batch:

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)



# Print the first examples input and target values:
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))



# Each index of these vectors are processed as one time step. 
# For the input at time step 0, the model receives the index for "F" 
# and trys to predict the index for "i" as the next character. At the 
# next timestep, it does the same thing but the RNN considers the 
# previous step context in addition to the current input character.
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    
    
# Create training batches

# We used tf.data to split the text into manageable sequences. 
# But before feeding this data into the model, we need to shuffle 
# the data and pack it into batches.



# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 1000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset


# Build The Model
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
  

model.summary()

  
# Try the model
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# To get actual predictions from the model we need to sample from the 
# output distribution, to get actual character indices. This distribution 
# is defined by the logits over the character vocabulary.

## Note: It is important to sample from this distribution as taking the 
## argmax of the distribution can easily get the model stuck in a loop.

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

sampled_indices


# Decode to see the text predicted by the untrained model:
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))




# Train the model

# Attach an optimizer and a loss function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

# Compile model with custom loss function
model.compile(optimizer='adam', loss=loss)


# Configure checkpoints
# Directory where the checkpoints will be saved
checkpoint_dir = './shakespeare_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# Execute training
EPOCHS=150

hist = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# Restore latest checkpoint
tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


"""
Prediction loop

The following code block generates the text:

    It Starts by choosing a start string, initializing the RNN state and 
    setting the number of characters to generate.

    Get the prediction distribution of the next character using the start 
    string and the RNN state.

    Then, use a categorical distribution to calculate the index of the 
    predicted character. Use this predicted character as our next input to
    the model.

    The RNN state returned by the model is fed back into the model so 
    that it now has more context, instead than only one character. After 
    predicting the next character, the modified RNN states are again fed 
    back into the model, which is how it learns as it gets more context 
    from the previously predicted characters.
"""

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)
  # Number of characters to generate
  num_generate = 1000
  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  # Empty string to store our results
  text_generated = []
  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0
  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)
      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"ROMEO: "))



"""
Advanced: Customized Training

The above training procedure is simple, but does not give you much control.

So now that you've seen how to run the model manually let's unpack the 
training loop, and implement it ourselves. This gives a starting point if, 
for example, to implement curriculum learning to help stabilize the model's 
open-loop output.

(https://www.tensorflow.org/api_docs/python/tf/GradientTape)
We will use tf.GradientTape to track the gradients. You can learn more about 
this approach by reading the eager execution guide. (https://www.tensorflow.org/guide/eager)

The procedure works as follows:

    First, initialize the RNN state. We do this by calling the 
    tf.keras.Model.reset_states method.

    Next, iterate over the dataset (batch by batch) and calculate the 
    predictions associated with each.

    Open a tf.GradientTape, and calculate the predictions and loss in that 
    context.

    Calculate the gradients of the loss with respect to the model variables 
    using the tf.GradientTape.grads method.

    Finally, take a step downwards by using the optimizer's 
    tf.train.Optimizer.apply_gradients method.
    """

"""
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss


# Training step
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  # initializing the hidden state at the start of every epoch
  # initally hidden is None
  hidden = model.reset_states()

  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  # saving (checkpoint) the model every 5 epochs
  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))
"""
