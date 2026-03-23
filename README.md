# AMO

## Состав группы
Киселёв Алексей Александрович - BremeCanell - DevOps Engineer
Кравец Виктория Дмитриевна - Rainbow-Ray - ML Engineer
Насибуллин Артур Ильнурович - Frogog / 123home - Data Scientist

## Очистка данных
Датасеты X и y в папках train и test очищены и масштабированы.
Данные взяты из train0.csv и test0.csv

## Загрузка данных
Загрузка тестовых данных: 

    x = pd.read_csv('test/X.csv').to_numpy()
    y = pd.read_csv('test/y.csv').to_numpy().ravel()

## Загрузка модели 
    
    filename = "model/model.pkl"
    with open(filename, 'rb') as f:
        loaded_model = pickle.load(f)
