import io
import os
import platform
import typing as t
import numpy as np
import tensorflow as tf

import matplotlib
if platform.system() == 'Darwin':
  matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Summary():
    """ Helper class to write TensorBoard summaries """
    def __init__(self, output_dir: str):
        self.dpi = 120
        plt.style.use('seaborn-deep')
        
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(output_dir, 'train'))
        self.validate_summary_writer = tf.summary.create_file_writer(os.path.join(output_dir, 'validate'))
    
    def scalar(self, tag, value, epoch, training):
        if training:
            with self.train_summary_writer.as_default():
                tf.summary.scalar(tag, value, step=epoch)
        else:
            with self.validate_summary_writer.as_default():
                tf.summary.scalar(tag, value, step=epoch)
      
    def losses(self, results):
        for key, value in results.items():
            value = tf.math.reduce_mean(value)
            print('%s = %.4f  ' %(key, value.numpy()), end='')
        print('\n')

    def image(self, tag, values, step: int = 0, training: bool = False):
        writer = self.get_writer(training)
        with writer.as_default():
          tf.summary.image(tag, data=values, step=step, max_outputs=len(values))
    
    def figure(self,
               tag,
               figure,
               step: int = 0,
               training: bool = False,
               close: bool = True):
        """ Write matplotlib figure to summary
        Args:
          tag: data identifier

          figure: matplotlib figure or a list of figures
          step: global step value to record
          training: training summary or test summary
          close: flag to close figure
        """
        buffer = io.BytesIO()
        figure.savefig(buffer, dpi=self.dpi, format='png', bbox_inches='tight')
        buffer.seek(0)
        image = tf.image.decode_png(buffer.getvalue(), channels=4)
        self.image(tag, tf.expand_dims(image, 0), step=step, training=training)
        if close:
          plt.close(figure)
    
    def image_cycle(self,
                    tag: str,
                    images: t.List[np.ndarray],
                    labels: t.List[str],
                    step: int = 0,
                    training: bool = False):
        """ Plot image cycle to TensorBoard 
        Args:
          tags: data identifier
          images: list of np.ndarray where len(images) == 3 and each array has 
                  shape (N,H,W,C)
          labels: list of string where len(labels) == 3
          step: global step value to record
          training: training summary or test summary
        """
        assert len(images) == len(labels) == 3
        for sample in range(len(images[0])):
          figure, axes = plt.subplots(nrows=1,
                                      ncols=3,
                                      figsize=(9, 3.25),
                                      dpi=self.dpi)
          axes[0].imshow(images[0][sample, ...], interpolation='none')
          axes[0].set_title(labels[0])
          
          axes[1].imshow(images[1][sample, ...], interpolation='none')
          axes[1].set_title(labels[1])
          
          axes[2].imshow(images[2][sample, ...], interpolation='none')
          axes[2].set_title(labels[2])
          
          plt.setp(axes, xticks=[], yticks=[])
          plt.tight_layout()
          figure.subplots_adjust(wspace=0.02, hspace=0.02)
          self.figure(tag=f'{tag}/sample_#{sample:03d}',
                      figure=figure,
                      step=step,
                      training=training,
                      close=True)


def append_dict(dict1: dict, dict2: dict, replace: bool = False):
    """ append items in dict2 to dict1 """
    for key, value in dict2.items():
      if replace:
        dict1[key] = value
      else:
        if key not in dict1:
          dict1[key] = []
        dict1[key].append(value)


def plot_cycle(plot_ds, gan, summary, epoch: int):
    samples = {}
    
    for x, y in plot_ds:
      fake_x, fake_y, cycle_x, cycle_y = gan.cycle_step(x, y, training=False)
      append_dict(dict1=samples,
                  dict2={
                      'x': x,
                      'y': y,
                      'fake_x': fake_x,
                      'fake_y': fake_y,
                      'cycle_x': cycle_x,
                      'cycle_y': cycle_y
                  })
    
    for key, images in samples.items():
      # scale images back to [0, 255]
      images = tf.concat(images, axis=0).numpy()
      images = ((images + 1) * 127.5).astype(np.uint8)
      samples[key] = images
    
    summary.image_cycle(
        tag=f'X_cycle',
        images=[samples['x'], samples['fake_y'], samples['cycle_x']],
        labels=['X', 'G(X)', 'F(G(X))'],
        step=epoch,
        training=False)
    summary.image_cycle(
        tag=f'Y_cycle',
        images=[samples['y'], samples['fake_x'], samples['cycle_y']],
        labels=['Y', 'F(Y)', 'G(F(Y))'],
        step=epoch,
        training=False)
