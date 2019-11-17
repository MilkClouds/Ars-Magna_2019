import keras, os
from keras import layers
from keras.preprocessing import image
import numpy as np
import random

class class_GAN:
    def __init__(self, latent_dim, height, width, channels):
        self.latent_dim = latent_dim
        self.height = height
        self.width = width
        self.channels = channels
        self.build_generator()
        self.build_discriminator()

        # 생성자의 훈련 중 판별자가 훈련되지 않도록 gan network에서 discriminator 층의 trainble을 false로 만든다.
        self.discriminator.trainable = False
        
        gan_input = keras.Input(shape=(latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = keras.models.Model(gan_input, gan_output)
        
        gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
        
        self.gan = gan


    def build_generator(self):
        generator_input = keras.Input(shape=(self.latent_dim,))
        
        # 입력을 16 × 16 크기의 128개 채널을 가진 특성 맵으로 변환
        x = layers.Dense(128 * 16 * 16)(generator_input)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((16, 16, 128))(x)
        
        # 합성곱 층 추가
        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)
        
        # 32 × 32 크기로 업샘플링
        x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)
        
        # 32 × 32 크기의 1개 채널을 가진 특성 맵 생성
        x = layers.Conv2D(self.channels, 7, activation='tanh', padding='same')(x)
        generator = keras.models.Model(generator_input, x)
        generator.summary()
        
        self.generator = generator
    
    def build_discriminator(self):
        discriminator_input = layers.Input(shape=(self.height, self.width, self.channels))
        x = layers.Conv2D(128, 3)(discriminator_input)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Flatten()(x)
        
        x = layers.Dropout(0.4)(x)
        
        # 분류 층
        x = layers.Dense(1, activation='sigmoid')(x)
        
        discriminator = keras.models.Model(discriminator_input, x)
        discriminator.summary()
        
        discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
        discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    
        self.discriminator = discriminator

latent_dim = 32
height = 32
width = 32
channels = 3

GAN = class_GAN(latent_dim, height, width, channels)

GAN.gan.load_weights('gan.h5')

# CIFAR10 데이터 로드
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()


x_train = x_train[y_train.flatten() == 3]

# 데이터 정규화
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.


iterations = 10000
batch_size = 100
save_dir = './datasets/gan_images/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# 훈련 반복 시작
start = 0
for step in range(6301, iterations + 1):
    # 잠재 공간에서 무작위로 포인트 샘플링(노이즈 생성)
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # 가짜 이미지 디코딩
    generated_images = GAN.generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])

    labels += 0.05 * np.random.random(labels.shape)

    # discriminator 훈련
    d_loss = GAN.discriminator.train_on_batch(combined_images, labels)

    # 잠재 공간에서 무작위로 포인트를 샘플링(노이즈 생성)
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    misleading_targets = np.zeros((batch_size, 1))

    # generator 훈련(gan 모델에서 discriminator의 가중치는 동결)
    a_loss = GAN.gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
      start = 0

    if step % 100 == 0:
        GAN.gan.save_weights('gan.h5')

        print('스텝 %s에서 판별자 손실: %s' % (step, d_loss))
        print('스텝 %s에서 적대적 손실: %s' % (step, a_loss))

        # 가짜 이미지
        img = image.array_to_img(generated_images[random.randint(0,len(generated_images)-1)] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_' + str(step) + '.png'))

        # 진짜 이미지
        img = image.array_to_img(real_images[random.randint(0,len(real_images)-1)] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_' + str(step) + '.png'))