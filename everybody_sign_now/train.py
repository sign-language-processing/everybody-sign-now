import datetime
import os
import time

import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from everybody_sign_now.config import BATCH_SIZE
from everybody_sign_now.data import project_dataset, batch_dataset, test_dataset
from everybody_sign_now.model import discriminator, discriminator_loss, discriminator_optimizer, generator, \
    generator_loss, \
    generator_optimizer

os.makedirs("progress", exist_ok=True)

p_dataset = project_dataset({"Amit": (255, 200, 200), "Maayan_1": (200, 200, 255), "Maayan_2": (200, 200, 255)})
train_dataset = batch_dataset(p_dataset, BATCH_SIZE)
test_dataset = batch_dataset(test_dataset(), 1)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def generate_images(model, test_input, tar, step):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 5))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig("progress/" + str(step) + ".png")


def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)


def fit(train_ds, test_ds, steps):
    example_input, example_target = next(test_ds)

    for step, (input_image, target_image) in enumerate(tqdm(train_ds)):
        if step % (1000 // BATCH_SIZE) == 0:
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target, int(step))
            print(f"Step: {step // 1000}k")

        train_step(input_image, target_image, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % (1000 // BATCH_SIZE) == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save("generator.h5")


# Restore saved training if failed, etc
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint.restore(latest_checkpoint)

fit(train_dataset, test_dataset, steps=50001)

# Fix nan weight
problematic_weight = generator.weights[35]
problematic_weight.assign(tf.zeros(problematic_weight.shape))

generator.save("generator.h5")
