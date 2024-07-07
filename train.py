from huggingface_hub import notebook_login
from huggingface_hub import hf_hub_download
from transformers.utils import send_example_telemetry
from transformers import VideoClassificationProcessor, VideoClassificationModel, Trainer, TrainingArguments
import pytorchvideo.data
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, RandomShortSideScale, UniformTemporalSubsample
import torchvision.transforms as transforms
import os
import pathlib
from itertools import chain
import imageio
import numpy as np
from IPython.display import Image

model_ckpt = "MCG-NJU/videomae-base"  # pre-trained model from which to fine-tune
batch_size = 8  # batch size for training and evaluation

notebook_login()

send_example_telemetry("video_classification_notebook", framework="pytorch")

hf_dataset_identifier = "MANMEET75/VideoAnalytics"
filename = "PullUpPushUps.zip"
file_path = hf_hub_download(
    repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"
)

# !unzip {file_path}

dataset_root_path = os.getcwd()
dataset_root_path = pathlib.Path(dataset_root_path)

video_count_train_avi = len(list(dataset_root_path.glob("train/*/*.avi")))
video_count_val_avi = len(list(dataset_root_path.glob("val/*/*.avi")))
video_count_test_avi = len(list(dataset_root_path.glob("test/*/*.avi")))

video_count_train_mp4 = len(list(dataset_root_path.glob("train/*/*.mp4")))
video_count_val_mp4 = len(list(dataset_root_path.glob("val/*/*.mp4")))
video_count_test_mp4 = len(list(dataset_root_path.glob("test/*/*.mp4")))

video_count_train = video_count_train_avi + video_count_train_mp4
video_count_val = video_count_val_avi + video_count_val_mp4
video_count_test = video_count_test_avi + video_count_test_mp4

video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

all_video_file_paths = list(
    chain(
        dataset_root_path.glob("train/*/*.avi"),
        dataset_root_path.glob("val/*/*.avi"),
        dataset_root_path.glob("test/*/*.avi"),
        dataset_root_path.glob("train/*/*.mp4"),
        dataset_root_path.glob("val/*/*.mp4"),
        dataset_root_path.glob("test/*/*.mp4"),
    )
)

all_video_file_paths[:5]

class_labels = sorted({str(path).split("/")[5] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Unique classes: {list(label2id.keys())}.")

processor = VideoClassificationProcessor.from_pretrained(model_ckpt)
model = VideoClassificationModel.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

mean = processor.feature_extractor.mean
std = processor.feature_extractor.std
input_size = processor.feature_extractor.frame_size

num_frames_to_sample = model.config.max_video_length
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps

train_transform = transforms.Compose([
    ApplyTransformToKey(
        key="video",
        transform=transforms.Compose([
            UniformTemporalSubsample(num_frames=num_frames_to_sample),
            Normalize(mean=mean, std=std),
            RandomShortSideScale(min_size=input_size[0], max_size=input_size[1]),
        ])
    ),
])

train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.RandomClipSampler(clip_duration),
    decode_audio=False,
    transform=train_transform,
)

val_transform = transforms.Compose([
    ApplyTransformToKey(
        key="video",
        transform=transforms.Compose([
            UniformTemporalSubsample(num_frames=num_frames_to_sample),
            Normalize(mean=mean, std=std),
            transforms.Resize(size=input_size),
        ])
    ),
])

val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.UniformClipSampler(clip_duration),
    decode_audio=False,
    transform=val_transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.UniformClipSampler(clip_duration),
    decode_audio=False,
    transform=val_transform,
)

sample_video = next(iter(train_dataset))
sample_video.keys()

def investigate_video(sample_video):
    """Utility to investigate the keys present in a single video sample."""
    for k in sample_video:
        if k == "video":
            print(k, sample_video["video"].shape)
        else:
            print(k, sample_video[k])

    print(f"Video label: {id2label[sample_video['label'].item()]}")

investigate_video(sample_video)

model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-HumanActivityRecognition"
num_epochs = 2

training_args = TrainingArguments(
    output_dir=new_model_name,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(p):
    """Computes accuracy on a batch of predictions."""
    predictions, labels = p
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate(test_dataset)
