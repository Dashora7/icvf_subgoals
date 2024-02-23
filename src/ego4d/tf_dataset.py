import tensorflow as tf
from functools import partial
from tf_augmentations import augment


PROTO_TYPE_SPEC = {"images": tf.string, "text": tf.string}


def get_ego4d_dataloader(
    path,
    *,
    batch_size,
    augment_kwargs,
    shuffle_buffer_size=25000,
    cache=True,
):
    # get the tfrecord files
    paths = tf.io.gfile.glob(tf.io.gfile.join(path, "*.tfrecord"))

    # split into train/val
    train_paths = paths[: int(0.9 * len(paths))]
    val_paths = paths[int(0.9 * len(paths)) :]

    train_dataset = _construct_dataset(
        train_paths,
        shuffle_buffer_size=shuffle_buffer_size,
        cache=cache,
    )
    val_dataset = _construct_dataset(
        val_paths,
        shuffle_buffer_size=shuffle_buffer_size,
        cache=cache,
    )

    # augment the training data
    train_dataset = train_dataset.enumerate()
    train_dataset = train_dataset.map(
        partial(_augment, augment_kwargs=augment_kwargs),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # batch the dataset
    train_dataset = train_dataset.batch(
        batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True
    )
    val_dataset = val_dataset.batch(
        batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True
    )

    # always prefetch last
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


def _construct_dataset(paths, *, shuffle_buffer_size, cache):
    # read them
    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.AUTOTUNE)

    # decode the examples (yields videos)
    dataset = dataset.map(_decode_example, num_parallel_calls=tf.data.AUTOTUNE)

    # cache all the dataloading
    if cache:
        dataset = dataset.cache()

    # add goals (yields videos)
    dataset = dataset.map(
        partial(_add_goals),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # unbatch to get individual frames
    dataset = dataset.unbatch()

    # process each frame
    dataset = dataset.map(_process_frame, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.repeat()

    # shuffle the dataset
    dataset = dataset.shuffle(shuffle_buffer_size)

    return dataset


def _decode_example(example_proto):
    # decode the example proto according to PROTO_TYPE_SPEC
    features = {
        key: tf.io.FixedLenFeature([], tf.string) for key in PROTO_TYPE_SPEC.keys()
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], dtype)
        for key, dtype in PROTO_TYPE_SPEC.items()
    }

    return parsed_tensors


def _add_goals(video):
    # video is a dict with keys "images" and "text"
    # "images" is a tensor of shape [n_frames, 224, 224, 3]
    # "text" is a tensor of shape [n_frames]

    # for now: for frame i, select a goal uniformly from the range [i, n_frames)
    num_frames = tf.shape(video["images"])[0]
    rand = tf.random.uniform(shape=[num_frames], minval=0, maxval=1, dtype=tf.float32)
    offsets = tf.cast(
        tf.floor(rand * tf.cast(tf.range(num_frames)[::-1], tf.float32)), tf.int32
    )
    indices = tf.range(num_frames) + offsets
    video["goals"] = tf.gather(video["images"], indices)

    # for now: just get rid of text
    del video["text"]

    return video


def _process_frame(frame):
    for key in ["images", "goals"]:
        frame[key] = tf.io.decode_jpeg(frame[key])
        # this will throw an error if any images aren't 224x224x3
        frame[key] = tf.ensure_shape(frame[key], [224, 224, 3])
        # may want to think more carefully about the resize method
        frame[key] = tf.image.resize(frame[key], [128, 128], method="lanczos3")
        # normalize images to [-1, 1]
        frame[key] = tf.cast(frame[key], tf.float32) / 127.5 - 1.0

    return frame


def _augment(seed, image, *, augment_kwargs):
    for key in ["images", "goals"]:
        image[key] = augment(image[key], [seed, seed], **augment_kwargs)

    return image


if __name__ == "__main__":
    import tqdm
    import matplotlib.pyplot as plt

    dataset, _ = get_ego4d_dataloader(
        "gs://rail-tpus-kevin/ego4d-tfrecord",
        batch_size=8,
        shuffle_buffer_size=1,
        augment_kwargs={
            "random_resized_crop": {"scale": [0.8, 1.0], "ratio": [0.9, 1.1]},
            "random_brightness": [0.05],
            "random_contrast": [0.95, 1.05],
            "random_saturation": [0.95, 1.05],
            "random_hue": [0.05],
            "augment_order": [
                "random_resized_crop",
                "random_flip_left_right",
                "random_flip_up_down",
                "random_rot90",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        },
    )
    with tqdm.tqdm() as pbar:
        for batch in dataset:
            pbar.update(len(batch["images"]))
            for image in batch["images"]:
                plt.imshow(image / 2 + 0.5)
                plt.savefig("test.png")
            assert False