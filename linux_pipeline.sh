#!/bin/bash

echo "Запуск конвейера машинного обучения"

echo "Генерируем данные"
python3 data_creation.py

echo "Производим предобработку данных"
python3 model_preprocessing.py

echo "Обучаем модели"
python3 model_preparation.py

echo "Проводим тестирование"
python3 model_testing.py

echo "Работа конвейера завершена!"
read -p "Нажмите Enter для выхода..."