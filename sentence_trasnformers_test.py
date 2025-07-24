import sentence_transformers

model = sentence_transformers.SentenceTransformer('jinaai/jina-embeddings-v2-base-en',trust_remote_code=True)
electronics_products = [
    "Смартфон Samsung Galaxy S23",
    "Смартфон iPhone 15 Pro",
    "Ноутбук ASUS ROG Zephyrus G14",
    "Ноутбук MacBook Pro 14 M3",
    "Планшет iPad Pro 12.9",
    "Планшет Samsung Galaxy Tab S9",
    "Умные часы Apple Watch Series 9",
    "Фитнес-браслет Xiaomi Mi Band 8",
    "Наушники AirPods Pro 2",
    "Наушники Sony WH-1000XM5",
    "Беспроводная колонка JBL Charge 5",
    "Умная колонка Яндекс Станция 2",
    "Игровая консоль PlayStation 5",
    "Игровая консоль Xbox Series X",
    "Видеокарта NVIDIA RTX 4090",
    "Процессор AMD Ryzen 9 7950X",
    "Материнская плата ASUS ROG Strix Z790",
    "Оперативная память Kingston Fury 32GB DDR5",
    "SSD-накопитель Samsung 980 Pro 1TB",
    "Монитор LG UltraFine 4K 27\"",
    "Монитор игровой ASUS TUF VG27AQ",
    "Клавиатура механическая Logitech G Pro X",
    "Мышь игровая Razer DeathAdder V3",
    "Коврик для мыши SteelSeries QcK Heavy",
    "Внешний жесткий диск WD Black 2TB",
    "Роутер ASUS RT-AX86U",
    "Веб-камера Logitech C920",
    "Микрофон HyperX QuadCast",
    "Графический планшет Wacom Intuos Pro",
    "Принтер лазерный HP LaserJet Pro",
    "3D-принтер Creality Ender 3",
    "Блок питания Corsair RM850x",
    "Корпус для ПК NZXT H7 Elite",
    "Система охлаждения Noctua NH-D15",
    "Игровой руль Logitech G29",
    "Джойстик Xbox Wireless Controller",
    "VR-шлем Meta Quest 3",
    "Электронная книга PocketBook 740",
    "Power Bank Xiaomi 20000 mAh",
    "Зарядное устройство Anker 65W GaN",
    "Кабель USB-C Thunderbolt 4",
    "Док-станция для ноутбука Dell WD19",
    "Сетевой фильтр APC SurgeArrest",
    "Стабилизатор напряжения Ресанта 1000Вт",
    "Умная лампочка Xiaomi Yeelight",
    "Робот-пылесос Roborock S8",
    "Умный замок Aqara Smart Lock",
    "Дрон DJI Mavic 3",
    "Экшн-камера GoPro Hero 12",
    "Фотоаппарат Sony Alpha A7 IV"
]

import datetime
print(datetime.datetime.now())
text_embeddings = model.encode(sentences=electronics_products[:10],task="text-matching")
print(datetime.datetime.now())

list_of_embeddings = text_embeddings.tolist()
print(len(list_of_embeddings))
print(len(list_of_embeddings[0]))