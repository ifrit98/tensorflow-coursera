data = "But, lo! from forth a copse that neighbours by,\n \
A breeding jennet, lusty, young, and proud, \n \
Adonis' trampling courser doth espy, \n \
And forth she rushes, snorts and neighs aloud; \n \
     The strong-neck'd steed, being tied unto a tree, \n \
     Breaketh his rein, and to her straight goes he. \n \
Imperiously he leaps, he neighs, he bounds, \n \
And now his woven girths he breaks asunder; \n \
The bearing earth with his hard hoof he wounds, \n \
Whose hollow womb resounds like heaven's thunder; \n \
     The iron bit he crushes 'tween his teeth \n \
     Controlling what he was controlled with. \n \
His ears up-prick'd; his braided hanging mane \n \
Upon his compass'd crest now stand on end; \n \
His nostrils drink the air, and forth again, \n \
As from a furnace, vapours doth he send: \n \
     His eye, which scornfully glisters like fire, \n \
     Shows his hot courage and his high desire. \n \
Sometime her trots, as if he told the steps, \n \
With gentle majesty and modest pride; \n \
Anon he rears upright, curvets and leaps, \n \
As who should say, Lo! thus my strength is tried; \n \
     And this I do to captivate the eye \n \
     Of the fair breeder that is standing by. \n \
What recketh he his rider's angry stir, \n \
His flattering Holla, or his Stand, I say? \n \
What cares he now for curb of pricking spur? \n \
For rich caparisons or trapping gay? \n \
     He sees his love, and nothing else he sees, \n \
     Nor nothing else with his proud sight agrees. \n \
Look, when a painter would surpass the life, \n \
In limning out a well-proportion'd steed, \n \
His art with nature's workmanship at strife, \n \
As if the dead the living should exceed; \n \
     So did this horse excel a common one, \n \
     In shape, in courage, colour, pace and bone \n \
Round-hoof'd, short-jointed, fetlocks shag and long, \n \
Broad breast, full eye, small head, and nostril wide, \n \
High crest, short ears, straight legs and passing strong, \n \
Thin mane, thick tail, broad buttock, tender hide: \n \
     Look, what a horse should have he did not lack, \n \
     Save a proud rider on so proud a back. \n \
Sometimes he scuds far off, and there he stares; \n \
Anon he starts at stirring of a feather;\n \
To bid the wind a race he now prepares,\n \
And where he run or fly they know not whether;\n \
     For through his mane and tail the high wind sings,\n \
     Fanning the hairs, who wave like feathered wings.\n \
He looks upon his love, and neighs unto her;\n \
She answers him as if she knew his mind;\n \
Being proud, as females are, to see him woo her,\n \
She puts on outward strangeness, seems unkind,\n \
     Spurns at his love and scorns the heat he feels,\n \
     Beating his kind embracements with her heels.\n \
Then, like a melancholy malcontent,\n \
He vails his tail that, like a falling plume\n \
Cool shadow to his melting buttock lent:\n \
He stamps, and bites the poor flies in his fume.\n \
     His love, perceiving how he is enraged,\n \
     Grew kinder, and his fury was assuaged.\n \
His testy master goeth about to take him;\n \
When lo! the unback'd breeder, full of fear,\n \
Jealous of catching, swiftly doth forsake him,\n \
With her the horse, and left Adonis there.\n \
     As they were mad, unto the wood they hie them,\n \
     Out-stripping crows that strive to over-fly them.\n \
     I prophesy they death, my living sorrow,\n \
     If thou encounter with the boar to-morrow.\n \
But if thou needs wilt hunt, be rul'd by me;\n \
Uncouple at the timorous flying hare,\n \
Or at the fox which lives by subtlety,\n \
Or at the roe which no encounter dare:\n \
     Pursue these fearful creatures o'er the downs,\n \
     And on they well-breath'd horse keep with they hounds.\n \
And when thou hast on food the purblind hare,\n \
Mark the poor wretch, to overshoot his troubles\n \
How he outruns with winds, and with what care\n \
He cranks and crosses with a thousand doubles:\n \
     The many musits through the which he goes\n \
     Are like a labyrinth to amaze his foes.\n \
Sometime he runs among a flock of sheep,\n \
To make the cunning hounds mistake their smell,\n \
And sometime where earth-delving conies keep,\n \
To stop the loud pursuers in their yell,\n \
     And sometime sorteth with a herd of deer;\n \
     Danger deviseth shifts; wit waits on fear:\n \
For there his smell with other being mingled,\n \
The hot scent-snuffing hounds are driven to doubt,\n \
Ceasing their clamorous cry till they have singled \n \
With much ado the cold fault cleanly out;\n \
     Then do they spend their mouths: Echo replies,\n \
     As if another chase were in the skies.\n \
By this, poor Wat, far off upon a hill,\n \
Stands on his hinder legs with listening ear,\n \
To hearken if his foes pursue him still:\n \
Anon their loud alarums he doth hear;\n \
     And now his grief may be compared well\n \
     To one sore sick that hears the passing-bell.\n \
Then shalt thou see the dew-bedabbled wretch\n \
Turn, and return, indenting with the way;\n \
Each envious briar his weary legs doth scratch,\n \
Each shadow makes him stop, each murmur stay:\n \
     For misery is trodden on by many,\n \
     And being low never reliev'd by any.\n \
Lie quietly, and hear a little more;\n \
Nay, do not struggle, for thou shalt not rise:\n \
To make thee hate the hunting of the boar,\n \
Unlike myself thou hear'st me moralize,\n \
     Applying this to that, and so to so;\n \
     For love can comment upon every woe.\n \
"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import numpy as np

corpus = data.lower().split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1


input_seqs = []
for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_seq = token_list[:i+1]
    input_seqs.append(n_gram_seq)

max_seq_len = max([len(x) for x in input_seqs])

input_sequences = np.array(
  pad_sequences(input_seqs, maxlen=max_seq_len, padding='pre')
)


# split sequences into xs and ys
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]
ys = to_categorical(labels, num_classes=total_words)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_seq_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(xs, ys, epochs=500, verbose=1)



import matplotlib.pyplot as plt

def plot_graphs(hist, string='acc'):
  plt.plot(hist.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()
  plt.clf()
  

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')



seed_text = "Laurence went to dublin"
next_words = 100

for _ in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
  predicted = model.predict_classes(token_list, verbose=0)
  output_word = ""
  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break
  seed_text += " " + output_word

print(seed_text)




# Tweaks to improve model and larger corpus


data = open('inst/python/course3/week4/laurence-songs').read()

from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Embedding(total_words, 128, input_length=max_seq_len-1))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(total_words, activation='softmax'))

adam = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

history = model.fit(xs, ys, epochs=100, verbose=2)

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
