import logging
import tensorflow as tf
import numpy as np

from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_single_tag_keys, get_choice, is_skipped

logger = logging.getLogger(__name__)

class DummyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(DummyModel, self).__init__(**kwargs)
        
        # pre-initialize your variables here
        from_name, schema = list(self.parsed_label_config.items())[0]
        #self.from_name = from_name
        #self.to_name = schema['to_name'][0]
        #self.labels = schema['labels']
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(self.parsed_label_config, 'Choices', 'Image')
        self.labels = tf.convert_to_tensor(sorted(self.labels_in_config))
        print(self.labels)
        self.image_width, self.image_height = 70, 70
        self.batch_size = 3
        self.epochs = 3
        self.model = tf.keras.models.load_model("ConvModel.keras")

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: 
            model returns the list of predictions based on input list of tasks
            
            :param tasks: Label Studio tasks in JSON format
        """
        print("predict!---------------------------------------------")
        print(tasks)
        print("predict!---------------------------------------------")

        image_path = get_image_local_path(tasks[0]['data'][self.value])

        #image = Image.open(image_path).resize((self.image_width, self.image_height))
        #image = np.array(image) / 255.0
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [self.image_height, self.image_width])
        image = image / 255.0
        result = self.model.predict(image[np.newaxis, ...])
        predicted_label_idx = np.argmax(result[0], axis=-1)
        predicted_label_score = result[0][predicted_label_idx]
        predicted_label = self.labels[predicted_label_idx]
        return [{
            'result': [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [str(predicted_label.numpy(), 'utf-8')]}
            }],
            'score': float(predicted_label_score)
        }]

    def fit(self, completions, workdir=None, **kwargs):
        """ This is where training happens: train your model given list of completions,
            then returns dict with created links and resources
            :param completions: aka annotations, the labeling results from Label Studio 
            :param workdir: current working directory for ML backend
        """
        # save some training outputs to the job result

        annotations = []
        for completion in completions:
            print("one completion")
            if is_skipped(completion):
                print("skipped!")
                continue
            image_path = get_image_local_path(completion['data'][self.value])
            image_label = get_choice(completion)
            annotations.append((image_path, image_label))
        
        print("annotations len:", len(annotations))
        if len(annotations)==0:
            return {'model_file': "ConvModel.keras", "0 annotations": True}

        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices(annotations)

        def prepare_item(item):
            print(item)
            label = tf.argmax(item[1] == self.labels)
            img = tf.io.read_file(item[0])
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.resize(img, [self.image_height, self.image_width])
            img = img / 255.0
            #img = tf.image.rgb_to_grayscale(img)
            #img = img[np.newaxis, ...]
            return img, label
        
        ds = ds.map(prepare_item, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache().shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.fit(ds, epochs=self.epochs)

        self.model.save("ConvModel.keras")
        return {'model_file': "ConvModel.keras"}