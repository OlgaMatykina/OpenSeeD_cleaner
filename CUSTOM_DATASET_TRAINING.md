# Quickstart

Если новый датасет использует формат COCO аннотаций и нейминг переменных не очень важен, то в дальнейшей инструкции достаточно изменить количество и имена классов в задействованных файлах. Затем адаптировать размер изображения и количество итераций и периоды чекпоинтов/валидации в конфиге.

# Preparing
OpenSeed работает с оригинальными размерами изображений и их аннотаций. Если датасет имеет картинки с большими размерами (напрмиер 3000х2250 пикс.), следует уменьшить их размеры и аннотации. Для этого можно использовать скрипт [scripts/resize_images.py](scripts/resize_images.py). Для его запуска необходимо запустить команду:

```
python scripts/resize_images.py \
--annotation_path datasets/valid_cleaner/train.json \
--images_path datasets/valid_cleaner/train
```

Здесь нужно передать:
- --annotation_path = путь к аннотационному файлу .json в формате **COCO**
- --images_path = путь к папке с изображениями (не добавляйте в конце слеш "/", могут возникнуть проблемы)

Также, в файле [scripts/resize_images.py](scripts/resize_images.py) после всех импортов необходимо указать **размеры выходных изображений**:
- MAX_WIDTH - новая ширина для входных изображений
- MAX_HEIGHT - новая высота для входных изображений

По дефолту эти константы имеют значения 512, выходные изображения будут квадратных размеров 512х512. 
Скрипт сформирует новую папку ресайзнутых изображения **images_path + "_resized"**. Там же будет лежать новая аннотация с именем **annotations.json**   

# Config
Основные части параметров, которые нужно изменить в конфигурационном файле следующие:

```
OUTPUT_DIR: # сюда сохраняются чекпоинты и результаты инференса
WEIGHTS: # path to checkpoint # с этих весом начинается обучение модели
```

```
SOLVER
```
От количества изображений и размера батча зависит максимальное количество итераций и период чекпоинта.

```
TEST
```
От количества изображений и размера батча зависит период запуска валидации (в итерациях).

Также необходимо создать конфиг для своего датасета, например:

```
ROSBAG:
  INPUT:
    MIN_SIZE_TEST: 480
    MAX_SIZE_TEST: 1333
    IMAGE_SIZE: 480
    MIN_SCALE: 0.1
    MAX_SCALE: 2.0
    DATASET_MAPPER_NAME: "rosbag_instance"
    IGNORE_VALUE: 255
    COLOR_AUG_SSD: False
    SIZE_DIVISIBILITY: 32
    RANDOM_FLIP: "horizontal"
    MASK_FORMAT: "bitmask"
    FORMAT: "RGB"
    CROP:
      ENABLED: True
  DATASET:
    DATASET: 'rosbag'
  TRAIN:
    ASPECT_RATIO_GROUPING: true
    BATCH_SIZE_TOTAL: 2
    BATCH_SIZE_PER_GPU: 2
    SHUFFLE: true
  TEST:
    DETECTIONS_PER_IMAGE: 100
    NAME: coco_eval
    IOU_TYPE: ['bbox', 'segm']
    USE_MULTISCALE: false
    BATCH_SIZE_TOTAL: 1
    MODEL_FILE: ''
    AUG:
      ENABLED: False
  DATALOADER:
    FILTER_EMPTY_ANNOTATIONS: False
    NUM_WORKERS: 1
    LOAD_PROPOSALS: False
    SAMPLER_TRAIN: "TrainingSampler"
    ASPECT_RATIO_GROUPING: True
```

Параметры INPUT чаще всего будут использовать Dataset Mapper, имя датасета будет использовать для формирования соответствующих eval и trainer dataloader.

TEST.NAME отвечает за тип evaluator, при использовании датасетов с COCO аннотацией лучше не изменять его.

# Dataset registration

Необходимо зарегистрировать датасет в detectron2. Для этого необходимо создать в [datasets/registration](datasets/registration) модуль, который будет регистрировать датасет. Например, [datasets/registration/register_rosbag.py](datasets/registration/register_rosbag.py)

Здесь основными параметрами являются количество классов и их названия, которые затем будут использовать в eval dataloader, а также пути к файлам для загрузки аннотаций и изображений. 

Наконец, необходимо добавить импорт созданного модуля в __init__.py [datasets/registration](datasets/registration) для регистрации пакета. 

# Dataset mapper

Dataset mapper используется для работы DataLoader во время обучения и валидации, позволяя формировать из оригинальных изображений и аннотаций датасета тензоры и объекты detectron2, использующиеся для работы модели. 

Для этого необходимо создать в [datasets/dataset_mappers](datasets/dataset_mappers) модуль, который будет регистрировать датасет. Например, [datasets/dataset_mappers/rosbag_instance_dataset_mapper.py](datasets/dataset_mappers/rosbag_instance_dataset_mapper.py). Dataset Mapper определяет обработку аннотаций и предобработку изображений. Если датасет соответствует формату COCO, то основными параметрами для изменения будут являться конкретные преобразования, применяемые во время обучения/инференса.

Наконец, необходимо добавить импорт созданного модуля в __init__.py [datasets/dataset_mappers](datasets/dataset_mappers) для регистрации пакета. 

# Build datasets

В файле [datasets/build.py](datasets/build.py) необходимо добавить имя нового датасета (пункт <YOUR DATASET NAME>.DATASET.DATASET) в следующие функции:
- get_config_from_name
- build_eval_dataloader
- build_train_dataloader

При этом для build_eval_dataloader и build_train_dataloader необходимо указать, какой Dataset Mapper использовать, а для get_config_from_name - какую часть общего конфигурационного файла.

# Utils

Необходимо указать в [utils/constants.py](utils/constants.py) список названий категорий из нового датасета. Эти имена будут использоваться для формирования текстовых запросов для модели, поэтому критически важно, чтобы их порядок совпадал с тем, что подразумевается при регистрации датасета.

Добавить загрузку категорий в [openseed/utils/misc.py](openseed/utils/misc.py) в функцию get_class_names.

Также, необходимо в файле [utils/misc.py](utils/misc.py) в функции hook_switcher добавить dataset switcher. Например, следующими строчками:

```
elif name in ['rosbag_v1_train', 'rosbag_v1_val']:
  mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True}
```

# Model 

Необходимо обновить файл [openseed/architectures/openseed_model.py](openseed/architectures/openseed_model.py)

1. При инициализации класса модели необходимо добавить логику создания списка названий категорий при обучении, например: 

```
if rosbag_on:
    task = 'seg'
    if not coco_mask_on:
        task = 'det'
    self.train_class_names[task] = get_class_names(train_dataset_name[0], background=background)
    self.train_class_names[task] = [a.replace("-merged", "").replace("-other", "").replace("-stuff", "") for a
                                        in self.train_class_names[task]]
    train_class_names = []
    for name in self.train_class_names[task]:
        names = name.split('-')
        if len(names) > 1:
            assert len(names) == 2
            train_class_names.append(names[1] + ' ' + names[0])
        else:
            train_class_names.append(name)
    self.train_class_names[task] = train_class_names
```
Данный флаг передаётся при инициализицаии в качестве параметра. Включение данного флага будет зависеть от значение следующего параметра в конфигурационном файле.

```
MODEL:
  DECODER:
    ROSBAG: True
```
Можно добавить дополнительные датасеты в конфигурацию декодера модели, обновив функцию from_config в [openseed/architectures/openseed_model.py](openseed/architectures/openseed_model.py), и неподсредственно конфигурационный файл.

2. Затем необходимо обновить функцию forward, добавив логику вычисления функции потерь.

```
if self.task_switch['rosbag']:
    #self.criterion.num_classes = 133 if 'pano' in self.train_dataset_name[0] else 80
    self.criterion.num_classes = 61
    task = 'seg'
    if not self.coco_mask_on:
        task = 'det'
    # import ipdb; ipdb.set_trace()
    losses_coco = self.forward_seg(batched_inputs['rosbag'], task=task)
    new_losses_coco = {}
    for key, value in losses_coco.items():
        new_losses_coco['coco.'+str(key)] = losses_coco[key]
    losses.update(new_losses_coco)
    losses.update({
        "Total loss": sum([loss for k, loss in losses.items()])
    })
    storage = get_event_storage()
    wandb.log(losses, step=storage.iter)

```
Параметр self.task_switch в норме зависит от rosbag_on и автоматически инициализируется во время инициализации модели. По сути, необходимо поменять только количество классов. 

Также при желании можно изменить тип вывода, например, ограничив его только инстанс сегментацией: 

```
if self.task_switch['rosbag']:
    inference_task = 'inst_seg'
```

Подробности того, на что влияют различные inference_task можно посмотреть в функции forward_seg. Соответственно, как можем заметить, параметры, связанные с типом сегментации, в конфигурационном файле ни на что не влияют. 

# Training

При использовании чекпоинта, загруженного из репозитория OpenSeeD:

```
python train_net.py --original_load --num-gpus 1 --config-file configs/openseed/openseed_swint_lang_rosbag.yaml
```

При использовании чекпоинтов, полученных в результате обучения с использованием данного кода:

```
python train_net.py --num-gpus 1 --config-file configs/openseed/openseed_swint_lang_rosbag.yaml
```

Перед запуском обучения рекомендуется проверить адекватную загрузку названий категорий с помощью запуска валидации: 

```
python train_net.py --original_load --eval_only --num-gpus 1 --config-file configs/openseed/openseed_swint_lang_rosbag.yaml
```

или

```
python train_net.py --eval_only --num-gpus 1 --config-file configs/openseed/openseed_swint_lang_rosbag.yaml
```

Т.к. OpenSeeD обладает хорошей способностью к Open Vocabulary Segmentation, вероятность того, что метрики будут около 0 для всех категорий низкая и такая ситуация скорее всего свидетельствует о том, что нарушено соответствие между названиями категорий внутри модуля OpenSeeD и внутри модуля, использующегося для валидации.

# Inference

Для запуска инференса необходимо запустить следующую команду:

```
python scripts/inference_instseg.py evaluate \
--conf_files configs/openseed/openseed_swint_lang_cleaner.yaml \
--annotation_path datasets/valid_cleaner/val.json \
--images_path datasets/valid_cleaner/val \
--output_root output/ \
--overrides WEIGHT Weights/chekpoints/model_final.pth
```

Здесь нужно передать:
- --conf_files = путь к конфигу датасета
- --annotation_path = путь к аннотационному файлу .json в формате **COCO**
- --images_path = путь к папке с изображениями (не добавляйте в конце слеш "/", могут возникнуть проблемы)
- --output_root = путь к папке с результатами инференса
- --overrides WEIGHT = путь к весам модели

Также, в файле [scripts/inference_instseg.py](scripts/inference_instseg.py) после всех импортов необходимо указать **константы**:
- REQUEST_LIST - желаемые запросы в списке
- TIME_NUMBER_START - номер изображения, после которого начнется замер среднего времени инференса
- MAX_WIDTH - максимальная ширина для входных изображений
- MAX_HEIGHT - максимальная высота для входных изображений
- THRESHOLD - порог по скору для визуализации и аннотаций

После окончания работы скрипта в папке --output_root будут лежать фотографии с размеченными масками и DT файл annotation.json с предсказанными боксами и масками в формате COCO.

# Metrics

Чтобы посчитать метрики на GT и DT файлах, необходимпо запустить следующую команду:

```
python scripts/calculate_metrics.py \
path/to/GT_file.json \
path/to/DT_file.json 
```

Здесь нужно передать:
- path/to/GT_file.json - путь к GT файлу в формате COCO
- path/to/DT_file.json - путь к DT файлу в формате COCO
