Title: Using tf.Transform For Input Pipelines
Date: 2018-09-09
Tags: tensorflow, tf-transform, apache-beam, cloud-ml
Slug: tf-transform

When initially building [my movie classification model](movie-reviews-tf.html), I used a version of the dataset that had already been preprocessed into TFRecords.  Though convenient, this created a problem when deploying the model; I wasn't able to replicate the preprocessing in my serving environment leading to training-serving skew.  My solution: [tf.Transform](https://github.com/tensorflow/transform).  

You can use tf.Transform to construct preprocessing pipelines that can be run as part of a Tensorflow graph.  tf.Transform prevents skew by ensuring that the data seen during serving is consistent with the data seen during training.  Furthermore, you can execute tf.Transform pipelines at scale with Apache Beam, a huge advantage when preparing large datasets for training.

<img src="{filename}images/tf-transform.png" width="600px" style="display:block; margin-left:auto; margin-right:auto;">

Source: [https://ai.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html](https://ai.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html)

Here is the code that I used to preprocess my data:

```python
# load data into TFRecords
def load_data(g, out):
    inputs = glob.glob(g)
    np.random.shuffle(inputs)
    with tf.python_io.TFRecordWriter(out) as writer:
        for i in inputs:
            label = 1 if i.split('/')[2] == 'pos' else 0
            with open(i, 'r') as f:
                review = f.read()
            
            example = tf.train.Example()
            example.features.feature['review'].bytes_list.value.append(review)
            example.features.feature['label'].int64_list.value.append(label)
                                
            writer.write(example.SerializeToString())
    
load_data('aclImdb/train/[posneg]*/*.txt', 'data/train.tfrecord')
load_data('aclImdb/test/[posneg]*/*.txt', 'data/test.tfrecord')
```

```python
# schema for raw data
RAW_DATA_FEATURE = {
    'review': tf.FixedLenFeature(shape=[1], dtype=tf.string),
    'label': tf.FixedLenFeature(shape=[1], dtype=tf.int64)
}

RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec(RAW_DATA_FEATURE))
```

```python
# train our tft transformer
with beam.Pipeline() as pipeline:
    with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
        coder = tft.coders.ExampleProtoCoder(RAW_DATA_METADATA.schema)

        train_data = (
            pipeline
            | 'ReadTrain' >> tfrecordio.ReadFromTFRecord('data/train.tfrecord')
            | 'DecodeTrain' >> beam.Map(coder.decode))

        test_data = (
            pipeline
            | 'ReadTest' >> tfrecordio.ReadFromTFRecord('data/test.tfrecord')
            | 'DecodeTest' >> beam.Map(coder.decode))

        
        # remove links, tags, quotes, apostraphes, and number commas
        # then lowercase, split by punctuation, and remove low frequency words
        def preprocessing_fn(inputs):
            remove = ["https?:\/\/(www\.)?([^\s]*)", "<([^>]+)>", "\'", "\""]
            remove = '|'.join(remove)
            
            reviews = tf.reshape(inputs['review'], [-1])
            reviews = tf.regex_replace(reviews, remove, '')
            reviews = tf.regex_replace(reviews, r"([0-9]),([0-9])", '\\1\\2')
            
            for letter in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                reviews = tf.regex_replace(reviews, letter, letter.lower())
                
            terms = tf.string_split(reviews, '.,!?() ')
            terms_indices = tft.compute_and_apply_vocabulary(terms, top_k=VOCAB_SIZE, default_value=VOCAB_SIZE, vocab_filename='vocab')
            
            return {
                'terms': terms_indices,
                'label': inputs['label']
            }

        
        (transformed_train_data, transformed_metadata), transform_fn = (
            (train_data, RAW_DATA_METADATA)
            | 'AnalyzeAndTransform' >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

        transformed_test_data, _ = (
            ((test_data, RAW_DATA_METADATA), transform_fn)
            | 'Transform' >> beam_impl.TransformDataset())
        
        transformed_data_coder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)

        _ = (
            transformed_train_data
            | 'EncodeTrain' >> beam.Map(transformed_data_coder.encode)
            | 'WriteTrain' >> tfrecordio.WriteToTFRecord('data/train_transformed.tfrecord'))

        _ = (
            transformed_test_data
            | 'EncodeTest' >> beam.Map(transformed_data_coder.encode)
            | 'WriteTest' >> tfrecordio.WriteToTFRecord('data/test_transformed.tfrecord'))
        
        _ = (
            transform_fn
            | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn('tft_output'))
```

And here we attach the tf.Transform preprocessing function (exported previously) to the trained classifier and export both for serving: 

```python
tf_transform_output = tft.TFTransformOutput('tft_output')

def serving_input_fn():
    review = tf.placeholder(dtype=tf.string)
    label = tf.zeros(dtype=tf.int64, shape=[1, 1]) # just a placeholder
    
    transformed_features = tf_transform_output.transform_raw_features({'review': review, 'label': label})
    
    return tf.estimator.export.ServingInputReceiver(transformed_features, {'review': review})


classifier.export_savedmodel(export_dir_base='exports',
                             serving_input_receiver_fn=serving_input_fn)
```

NOTE: My preprocessing function requires a 'label' input, which we obviously don't have for inference requests.  I impute an arbitrary tensor here to avoid an error.

While I have found tf.Transform super-useful, I am still constrained by preprocessing that can be done with native TF ops!  [`tf.py_func`](https://www.tensorflow.org/api_docs/python/tf/py_func) lets you insert a Python function as a TF op.  However, a documented limitation is that it is *not* serialized in the GraphDef, so it cannot be used for serving, which requires serializing the model and restoring in a different environment.  This has prevended me from doing more complicated text preprocessing steps like Porter stemming.

Nevertheless, I still love tf.Transform, an unsung hero of the TF ecosystem!  Here's a [link](https://github.com/donaldrauscher/movie-reviews-tf) to all the code for the model build.  Cheers!
 