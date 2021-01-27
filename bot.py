import telebot
import io

from torch import no_grad
from torchvision import transforms
from torch import load
from PIL import Image
from math import sqrt

import dataset
import models

SIZE = 512

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

model = models.Generator(dim=32, dropout=0.5)
checkpoint = load('generator_weights')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
for param in model.parameters():
    param.requires_grad = False

TOKEN = "1545167241:AAE10GplOEUHSFVc3Ko7CZxgyEl4-pM7AE0"
bot = telebot.TeleBot(TOKEN)
user_dict = dict()

keyboard = telebot.types.ReplyKeyboardMarkup(True)
keyboard.row('Задать коэффициент расстяжения')
hideboard = telebot.types.ReplyKeyboardRemove() 


@bot.message_handler(commands=['start'])
def send_initial(message):
    text = 'Это трансформер фото в изображения jojo bizzare adventures (так по крайней мере задумывалось).\n'+\
    'Задайте коэффициент расстяжения или сразу отправьте фото и получите преобразование'
    bot.send_message(message.from_user.id, text, reply_markup=keyboard)


@bot.message_handler(commands=['help'])
def send_initial(message):
    bot.send_message(message.from_user.id,  'Преобразование осуществляется с помощью CycleGAN\n\n' +
        'Использовать параметр растяжения следует следующим образом:\n' +
        'изображение слишком размыто - увеличьте коэффициент, слишком детализированно - уменьшите\n\n' +
        'Физический смысл - во сколько раз увеличивается фото перед трансформацией. Стандартное значение = 1\n\n' +
        f'Примечание: на площадь фотографии установлено ограничение в {SIZE}x{SIZE} пикселей. ' +
        'Ваше изображение будет дополнительно уменьшено, если оно превысит допустимый размер\n\n'+ \
        'Вы можете в любой момент отправить фотографию и получить ее преобразование')


@bot.message_handler(content_types=['text'])
def send_messages(message):
    if message.text.lower() == 'задать коэффициент расстяжения':
        bot.send_message(message.from_user.id, 'Отправьте положительное число' +
                         '(во сколько раз увеличить размер фото). Дробная часть - через точку',
                         reply_markup=hideboard)
        bot.register_next_step_handler(message, set_coef)


@bot.message_handler(content_types=['photo'])
def convert_photo(message):
    try:
        fileID = message.photo[-1].file_id
    except:
        bot.send_message(message.from_user.id, 'Фото некорректно. Повторите попытку')
        return 0
    file = bot.get_file(fileID)
    image = bot.download_file(file.file_path)
    image = Image.open(io.BytesIO(image))
    if message.from_user.id in user_dict and user_dict[message.from_user.id] != 1:
        size = (int(user_dict[message.from_user.id] * image.size[1]),
                int(user_dict[message.from_user.id] * image.size[0]))
    else:
        size = (image.size[1], image.size[0])
    s = size[0] * size[1]
    if s > SIZE * SIZE:
        size = (int(size[0] * SIZE / sqrt(s)), int(size[1] * SIZE / sqrt(s)))
        bot.send_message(message.from_user.id,
                         f'Площадь фотографии более {SIZE}x{SIZE}, размеры были уменьшены')
    image = transforms.Resize(size)(image)
    image = transform(image).unsqueeze(0)
    bot.send_message(message.from_user.id, 'Обработка займет некоторое время')

    with no_grad():
        image = transforms.ToPILImage()(dataset.de_norm(model(image).detach()[0]))
    bio = io.BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    bot.send_photo(message.from_user.id, photo=bio)
    bot.send_message(message.from_user.id, 'Можете отправить следующее фото')

    del image
    del bio

def set_coef(message):
    try:
        coef = float(message.text)
        if coef <= 0:
            raise ValueError
        user_dict[message.from_user.id] = coef
        bot.send_message(message.from_user.id, 'Коэффициент расстяжения задан на {}'.format(coef), reply_markup=keyboard)
    except ValueError:
        bot.send_message(message.from_user.id, 'Коэффициент введен некорректно. Повторите попытку')
        bot.register_next_step_handler(message, set_coef)


bot.polling(none_stop=True)

