Title: Building and Deploying a Deep Learning Model Part 1: Using tf.Transform For Input Pipelines
Date: 2018-09-02
Tags: tensorflow, tf-transform, apache-beam, cloud-ml
Slug: movie-reviews-tf-part1

This is part 1 in a 3-part series ([part 2](./movie-reviews-tf-part2.html), [part 3](./movie-reviews-tf-part3.html)) on building and deploying a deep learning model for the popular [ACL 2011 IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/).  In this part, I tackle data preprocessing.

===

The `sklearn.preprocessing` module has some great utility functions and transformer classes (e.g. scaling, encoding categorical features) for converting raw data into a numeric representation that can be modelled.  How do we do this in the context of Tensorflow?  And how do we ensure serving-time preprocessing transformations are exactly as those performed during training?  The solution: [tf.Transform](https://github.com/tensorflow/transform).  

<img src="{filename}images/tf-transform.png" width="600px" style="display:block; margin-left:auto; margin-right:auto;">

Source: [https://ai.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html](https://ai.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html)

You can use tf.Transform to construct preprocessing pipelines that can be run as part of a Tensorflow graph. tf.Transform prevents skew by ensuring that the data seen during serving is consistent with the data seen during training.  Furthermore, you can execute tf.Transform pipelines at scale with Apache Beam, a huge advantage when preparing large datasets for training.  Currently, you can only use tf.Transform in Python 2 since [Apache Beam doesn't yet have Python 3 support](https://jira.apache.org/jira/browse/BEAM-1251).

Here is the code that I used to preprocess my data.  I start by converting raw data into TFRecords files, then I transform those TFRecords files with tf.Transform.

```python
# this pulls out our proper nouns and treats them as single words
def proper_preprocessing(review):
    proper = r"([A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z]([a-z]+|\.)(?:\s+[0-9]+)?)"
    space_between_brackets = r"[\.\s]+(?=[^\[\]]*]])"
    brackets = r"(?:[\[]{2})(.*?)(?:[\]]{2})"
    
    review = re.sub(proper, '[[\\1]]', review)
    review = re.sub(space_between_brackets, '~', review)
    review = re.sub(brackets, '\\1', review)
    return review
```

```python
# load into TFRecords
def load_data(g, out):
    inputs = glob.glob(g)
    np.random.shuffle(inputs)
    with tf.python_io.TFRecordWriter(out) as writer:
        for i in inputs:
            label = 1 if i.split('/')[2] == 'pos' else 0
            with open(i, 'r') as f:
                review = f.read()
            
            example = tf.train.Example()
            example.features.feature['review'].bytes_list.value.append(proper_preprocessing(review))
            example.features.feature['label'].int64_list.value.append(label)
                                
            writer.write(example.SerializeToString())


load_data('aclImdb/train/[posneg]*/*.txt', TRAIN_PATH)
load_data('aclImdb/test/[posneg]*/*.txt', TEST_PATH)
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
            | 'ReadTrain' >> tfrecordio.ReadFromTFRecord(TRAIN_PATH)
            | 'DecodeTrain' >> beam.Map(coder.decode))

        test_data = (
            pipeline
            | 'ReadTest' >> tfrecordio.ReadFromTFRecord(TEST_PATH)
            | 'DecodeTest' >> beam.Map(coder.decode))


        # remove links, tags, quotes, apostraphes
        # bracketize proper nouns, names, and numbers
        # then lowercase, split by punctuation, and remove low frequency words
        def preprocessing_fn(inputs):
            remove = '|'.join(["https?:\/\/(www\.)?([^\s]*)", "<([^>]+)>", "\'", "\""])
            punctuation = r"([.,;!?\(\)\/])+"
            number_commas = r"([0-9]),([0-9])"

            reviews = tf.reshape(inputs['review'], [-1])

            reviews = tf.regex_replace(reviews, remove, '')
            reviews = tf.regex_replace(tf.regex_replace(reviews, punctuation, ' \\1 '), r"\s+", ' ')
            reviews = tf.regex_replace(reviews, number_commas, '\\1\\2')

            for letter in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                reviews = tf.regex_replace(reviews, letter, letter.lower())

            terms = tf.string_split(reviews, ' ')
            terms_indices = tft.compute_and_apply_vocabulary(terms, frequency_threshold=5, num_oov_buckets=1, vocab_filename='vocab')

            return {
                'terms': terms,
                'terms_indices': terms_indices,
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
            | 'WriteTrain' >> tfrecordio.WriteToTFRecord(TRAIN_TRANSFORMED_PATH))

        _ = (
            transformed_test_data
            | 'EncodeTest' >> beam.Map(transformed_data_coder.encode)
            | 'WriteTest' >> tfrecordio.WriteToTFRecord(TEST_TRANSFORMED_PATH))

        _ = (
            transform_fn
            | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(TFT_OUT_PATH))
```

NOTE: [RE2](https://github.com/google/re2) does not support constructs for which [*only* backtracking solutions are known to exist](https://github.com/google/re2/wiki/WhyRE2). Thus, backreferences and look-around assertions are not supported! As a result, I can't put my logic for identifying movie names / proper nouns into tf.regex_replace(...).

While I have found tf.Transform super-useful, we are still constrained by preprocessing that can be done with native TF ops!  [`tf.py_func`](https://www.tensorflow.org/api_docs/python/tf/py_func) lets you insert a Python function as a TF op.  However, a documented limitation is that it is *not* serialized in the GraphDef, so it cannot be used for serving, which requires serializing the model and restoring in a different environment.  This has prevended me from doing more complicated text preprocessing steps like Porter stemming.  Nevertheless, I still love tf.Transform, an unsung hero of the TF ecosystem!  

Link to all code: [https://github.com/donaldrauscher/movie-reviews-tf](https://github.com/donaldrauscher/movie-reviews-tf)
