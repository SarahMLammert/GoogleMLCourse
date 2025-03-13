import keras
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import plotly.express as px
import dataclasses

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

rice_dataset_raw = pd.read_csv("Rice_Cammeo_Osmancik.csv")
rice_dataset = rice_dataset_raw[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class',
    ]]
#print(rice_dataset.describe())
min_length = rice_dataset['Major_Axis_Length'].min()
max_length = rice_dataset['Major_Axis_Length'].max()
#print("{:.1f} px is the minimum length of the rice grains".format(min_length))
#print("{:.1f} px is the maximum length of the rice grains".format(max_length))
min_area = rice_dataset['Area'].min()
max_area = rice_dataset['Area'].max()
#print("{:.1f} px is the minimum area".format(min_area))
#print("{:.1f} px is the maximum area".format(max_area))
#print("{:.1f} px is the range".format(max_area-min_area))
max_perimeter = rice_dataset['Perimeter'].max()
#print("{:.1f} px is the maximum perimeter".format(max_perimeter))
sd_perimeter = rice_dataset['Perimeter'].std()
#print("{:.1f} px is the standard deviation for perimeter".format(sd_perimeter))
mean_perimeter = rice_dataset['Perimeter'].mean()
#print("{:.1f} px is the mean perimeter".format(mean_perimeter))
sd_max_perimeter = (max_perimeter-mean_perimeter)/sd_perimeter
#print("{:.1f} px is the number of standard deviations from the mean for the largest perimeter".format(sd_max_perimeter))

'''
for x_axis_data, y_axis_data in [
    ('Area', 'Eccentricity'),
    ('Convex_Area', 'Perimeter'),
    ('Major_Axis_Length', 'Minor_Axis_Length'),
    ('Perimeter', 'Extent'),
    ('Eccentricity', 'Major_Axis_Length'), ]:
  px.scatter(rice_dataset, x=x_axis_data, y=y_axis_data, color='Class').show()

x_axis_data = 'Area'
y_axis_data = 'Extent'
z_axis_data = 'Major_Axis_Length'

px.scatter_3d(rice_dataset,
              x=x_axis_data,
              y=y_axis_data,
              z=z_axis_data,
              color='Class',).show()
'''

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std
normalized_dataset['Class'] = rice_dataset['Class']
#print(normalized_dataset.head())

keras.utils.set_random_seed(42)
normalized_dataset['Class_Bool'] = (normalized_dataset['Class'] == 'Cammeo').astype(int)
normalized_dataset.sample(10)

number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]
#print(test_data.head())

label_columns = ['Class', 'Class_Bool']
train_features = train_data.drop(columns=label_columns)
train_labels = train_data['Class_Bool'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()


input_features = ['Eccentricity',
                  'Major_Axis_Length',
                  'Area',]


@dataclasses.dataclass()
class ExperimentSettings:
  learning_rate: float
  number_epochs: int
  batch_size: int
  classification_threshold: float
  input_features: list[str]

@dataclasses.dataclass()
class Experiment:
  name: str
  settings: ExperimentSettings
  model: keras.Model
  epochs: np.ndarray
  metrics_history: keras.callbacks.History

  def get_final_metric_value(self, metric_name: str) -> float:
    if metric_name not in self.metrics_history:
      raise ValueError(f'Unknown metric {metric_name}: available metrics are'
                       f' {list(self.metrics_history.columns)}')
    return self.metrics_history[metric_name].iloc[-1]


def create_model(settings: ExperimentSettings,
                 metrics: list[keras.metrics.Metric],) -> keras.Model:
  model_inputs = [keras.Input(name=feature, shape=(1,))
                  for feature in settings.input_features]

  concatenated_inputs = keras.layers.Concatenate()(model_inputs)
  dense = keras.layers.Dense(
      units=1, input_shape=(1,), name='dense_layer', activation=keras.activations.sigmoid)
  
  model_output = dense(concatenated_inputs)
  model = keras.Model(inputs=model_inputs, outputs=model_output)
  model.compile(optimizer=keras.optimizers.RMSprop(settings.learning_rate),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics,)
  return model


def train_model(experiment_name: str,
                model: keras.Model,
                dataset: pd.DataFrame,
                labels: np.ndarray,
                settings: ExperimentSettings,) -> Experiment:
  features = {feature_name: np.array(dataset[feature_name])
              for feature_name in settings.input_features}

  history = model.fit(x=features,
                      y=labels,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs,)

  return Experiment(name=experiment_name,
                    settings=settings,
                    model=model,
                    epochs=history.epoch,
                    metrics_history=pd.DataFrame(history.history),)

def plot_experiment_metrics(experiment: Experiment, metrics: list[str]):
  plt.figure(figsize=(12, 8))

  for metric in metrics:
    plt.plot(experiment.epochs, experiment.metrics_history[metric], label=metric)

  plt.xlabel("Epoch")
  plt.ylabel("Metric value")
  plt.grid()
  plt.legend()

settings = ExperimentSettings(learning_rate=0.001,
                              number_epochs=60,
                              batch_size=100,
                              classification_threshold=0.35,
                              input_features=input_features,)

metrics = [keras.metrics.BinaryAccuracy(name='accuracy', threshold=settings.classification_threshold),
           keras.metrics.Precision(name='precision', thresholds=settings.classification_threshold),
           keras.metrics.Recall(name='recall', thresholds=settings.classification_threshold),
           keras.metrics.AUC(num_thresholds=100, name='auc'),]

model = create_model(settings, metrics)
experiment = train_model('baseline', model, train_features, train_labels, settings)

plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
plot_experiment_metrics(experiment, ['auc'])