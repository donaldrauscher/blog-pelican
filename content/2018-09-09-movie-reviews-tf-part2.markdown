Title: Building and Deploying a Deep Learning Model Part 2: Building the Custom Estimator
Date: 2018-09-09
Tags: tensorflow, nlp, glove, transfer-learning
Slug: movie-reviews-tf-part2

This is part 2 in a 3-part series ([part 1](./movie-reviews-tf-part1.html), [part 3](./movie-reviews-tf-part3.html)) on building and deploying a deep learning model for the popular [ACL 2011 IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/).  In this part, I build a custom estimator in Tensorflow.

===

A few details on the model itself:

* I used cosine annealing to reduce the learning rate throughout training
* I used dropout to counteract overfitting and batch normalization before each activation layer
* I used leaky ReLU rather than regular ReLU to mitigate the "dying ReLU" problem where neurons get stuck in negative states
* I leveraged transfer learning, using [Glove](https://nlp.stanford.edu/projects/glove/) to initialize my word embedding
* Rather than using bag-of-words which ignores the structure of sentences, I used a 1D convolution layer to model the interaction between words and their neighbors

<br/>

### Initialize word embeddings with [GloVe](https://nlp.stanford.edu/projects/glove/)


```python
# get vocabulary
vocab = tft_output.vocabulary_by_name('vocab')
vocab_size = len(vocab)
```


```python
# load glove embeddings
embedding_size = 200
glove_embeddings = {}

with open('glove/glove.twitter.27B.{}d.txt'.format(embedding_size), mode='r') as f:  
    for line in f:
        values = line.strip().split()
        w = values[0]
        vectors = np.asarray(values[1:], dtype='float32')
        glove_embeddings[w] = vectors
```


```python
# create initialized embedding matrix
embedding_matrix = truncnorm.rvs(a=-2, b=2, size=(vocab_size+1, embedding_size))

glove_np = pd.DataFrame(glove_embeddings).values
glove_mu, glove_std = np.mean(glove_np), np.std(glove_np)
        
for i, w in enumerate(vocab):
    try:
        embedding_matrix[i] = np.clip((glove_embeddings[w] - glove_mu)/glove_std, -2, 2)
    except KeyError:
        pass

embedding_matrix = embedding_matrix / math.sqrt(embedding_size)
    
def embedding_initializer(shape=None, dtype=tf.float32, partition_info=None):  
    assert dtype is tf.float32
    return embedding_matrix
```

### Build classifier


```python
# input function
def input_fn(input_file_pattern, num_epochs=None, batch_size=25, shuffle=True, prefetch=1):  
    input_file_names = glob.glob(input_file_pattern)
    
    ds = tf.data.TFRecordDataset(input_file_names)
    ds = ds.cache()

    if shuffle:
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000, count=num_epochs))
    else:
        ds = ds.repeat(num_epochs)

    ds = ds.apply(tf.contrib.data.map_and_batch(
        map_func=lambda x: tf.parse_single_example(x, feature_spec), 
        batch_size=batch_size,
        num_parallel_calls=multiprocessing.cpu_count()
    ))
    
    if prefetch > 0:
        ds = ds.prefetch(prefetch)
    
    features = ds.make_one_shot_iterator().get_next()
    labels = features.pop('label')
    return features, labels

train_input_fn = functools.partial(input_fn,
                                   input_file_pattern=wildcard(TRAIN_TRANSFORMED_PATH),
                                   num_epochs=1)

test_input_fn = functools.partial(input_fn,
                                  input_file_pattern=wildcard(TEST_TRANSFORMED_PATH),
                                  num_epochs=1)
```


```python
# create estimator spec
def make_model(features, labels, mode, params, config):

    # hyperparameters
    dropout = params['dropout']
    conv_filters = params['conv_filters']
    dense_units = params['dense_units']
    learning_rate_start = params['learning_rate_start']
    learning_rate_steps = params['learning_rate_steps']
    
    # flag if training
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    # set up feature columns
    terms = features['terms_indices']
    
    terms_shape = terms.dense_shape
    terms_shape = tf.stack([terms_shape[0], tf.where(terms_shape[1] < 3, tf.constant(3, dtype=tf.int64), terms_shape[1])], axis=0)

    terms = tf.sparse_to_dense(terms.indices, terms_shape, terms.values, default_value=vocab_size)
    terms_embed_seq = tf.contrib.layers.embed_sequence(terms, vocab_size=vocab_size+1, embed_dim=embedding_size, initializer=embedding_initializer)
    
    # build graph
    net = terms_embed_seq
    net = tf.layers.dropout(net, rate=dropout, training=is_training)
    net = tf.layers.conv1d(inputs=net, filters=conv_filters, kernel_size=3, strides=1, activation=tf.nn.leaky_relu)
    net = tf.reduce_max(input_tensor=net, axis=1)      
    net = tf.layers.dropout(net, rate=dropout, training=is_training)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.layers.dense(net, units=dense_units, activation=tf.nn.leaky_relu)
    logits = tf.layers.dense(net, 2)
    
    # compute predictions
    predicted_classes = tf.argmax(logits, 1)
    predicted_probs = tf.nn.softmax(logits)
    
    # generate predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': predicted_probs
        }
        
        export_outputs = {
          'predict': tf.estimator.export.PredictOutput(outputs=predictions)
        }
        
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # create training op with cosine annealing for learning rate
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        
        learning_rate = tf.train.cosine_decay(learning_rate=learning_rate_start, global_step=global_step, 
                                              alpha=0.05, decay_steps=learning_rate_steps)
        
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
        
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # compute evaluation metrics
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_classes),
        'auc': tf.metrics.auc(labels=labels, predictions=predicted_probs[:, 1])
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

### Train classifier


```python
# build classifier
!rm -Rf $MODEL_LOG

epoch_size = 25000
num_epochs = 5
batch_size = 10
num_steps = epoch_size * num_epochs / batch_size // 1000 * 1000

params = dict(
    dropout=0.2,
    conv_filters=500,
    dense_units=100,
    learning_rate_start=0.1,
    learning_rate_steps=num_steps
)

ckpt_config = tf.estimator.RunConfig(keep_checkpoint_max=num_epochs)

classifier = tf.estimator.Estimator(model_fn=make_model,
                                    params=params,
                                    model_dir=MODEL_LOG,
                                    config=ckpt_config)
```


```python
# train classifier
train_stats = []
for i in range(num_epochs):
    print("Starting epoch {}/{}...".format(i+1, num_epochs))
    classifier.train(input_fn=lambda: train_input_fn(batch_size=batch_size))
    ckpt = classifier.latest_checkpoint()
    train_auc = classifier.evaluate(input_fn=lambda: train_input_fn())['auc']
    test_auc = classifier.evaluate(input_fn=lambda: test_input_fn())['auc']
    train_stats.append((ckpt, train_auc, test_auc))

train_stats = pd.DataFrame(train_stats, columns=['ckpt', 'train_auc', 'test_auc'])
```

    Starting epoch 1/5...
    Starting epoch 2/5...
    Starting epoch 3/5...
    Starting epoch 4/5...
    Starting epoch 5/5...


### Evaluate classifier


```python
# plot train stats
ind = np.arange(len(train_stats)) + 1
width = 0.35

fig, ax = plt.subplots()
train_bar = ax.bar(ind - width/2, train_stats['train_auc'].round(4), width, color='SkyBlue', label='Train')
test_bar = ax.bar(ind + width/2, train_stats['test_auc'].round(4), width,  color='IndianRed', label='Test')

# adds labels to a bar chart series
def autolabel(ax, rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')

autolabel(ax, train_bar, "center")
autolabel(ax, test_bar, "center")

ax.set_ylabel('AUC')
ax.set_xlabel('Epochs')
ax.set_xticks(ind)
ax.legend()
ax.set_ylim(0.8, 1.1)

plt.show()
```


![png](/images/movie-review-auc.png)



```python
# overall stats
best_ckpt = train_stats.sort_values(by=['test_auc'], ascending=False)['ckpt'].values[0]

train_stats = classifier.evaluate(input_fn=train_input_fn, checkpoint_path=best_ckpt)
test_stats = classifier.evaluate(input_fn=test_input_fn, checkpoint_path=best_ckpt)

train_stats = pd.DataFrame.from_dict(train_stats, orient='index', columns=['train'])
test_stats = pd.DataFrame.from_dict(test_stats, orient='index', columns=['test'])
stats = train_stats.join(test_stats)
stats
```


<table class="pretty">
  <thead>
    <tr>
      <th></th>
      <th>train</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loss</th>
      <td>0.088654</td>
      <td>0.230451</td>
    </tr>
    <tr>
      <th>auc</th>
      <td>0.997005</td>
      <td>0.969034</td>
    </tr>
    <tr>
      <th>global_step</th>
      <td>12500.000000</td>
      <td>12500.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.973600</td>
      <td>0.911200</td>
    </tr>
  </tbody>
</table>


### Export


```python
def serving_input_fn():
    review = tf.placeholder(dtype=tf.string)
    label = tf.zeros(dtype=tf.int64, shape=[1, 1]) # just a placeholder
    
    transformed_features = tft_output.transform_raw_features({'review': review, 'label': label})
    
    return tf.estimator.export.ServingInputReceiver(transformed_features, {'review': review})


export_path = classifier.export_savedmodel(export_dir_base='exports',
                                           serving_input_receiver_fn=serving_input_fn,
                                           checkpoint_path=best_ckpt)

export_path = export_path.decode('utf-8')
```

Link to all code: [https://github.com/donaldrauscher/movie-reviews-tf](https://github.com/donaldrauscher/movie-reviews-tf)
