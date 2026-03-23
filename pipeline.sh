#!/bin/bash

echo "Запуск конвейера машинного обучения"

echo "Генерируем данные"
python data_creation.py

echo "Производим предобработку данных"
python model_preprocessing.py

echo "Обучаем модели"
python model_preparation.py

echo "Проводим тестирование"
python model_testing.py

echo "Работа конвейера завершена!"
read -p "Нажмите Enter для выхода..."