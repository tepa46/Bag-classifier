# Bag-classifier


## Классификатор изображений пакетов.

### Датасет

[Plastic - Paper - Garbage Bag Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images?resource=download)


### Классифицируемые виды пакетов:

1) Мусорный
2) Пластиковый
3) Бумажный

### Гипотезы для классификации

1) Мусорные пакеты имеют темные цвета
2) Бумажные пакеты имеют более насыщенные цвета
3) Пластиковые пакеты и мусорные пакеты часто содержат яркие блики
4) На изображениях с бумажными пакетами много длинных отрезков
5) Бумажные пакеты часто имеют светлокоричневый цвет
6) Бумажные пакеты - матовые
7) Пластиковые пакеты имеют яркие цвета (желтый, синий, белый)
8) Из-за сильно выраженных складок на мусорных пакетах, найденные контуры мусорных пакетов по площади будут меньше, чем контуры пластиковых и бумажных пакетов
9) Контуры бумажных и пластиковых пакетов имеют меньше углов
10) Пластиковые пакеты из-за своей прозрачности могут иметь участки ненасыщенного цвета

### Проверка гипотез

Проверка всех гипотез осуществляется с использованием статистических тестов не требующих нормального распределения данных:

- Mann–Whitney U test
- Kolmogorov–Smirnov test


### Иллюстрация работы инструмента

<img width="394" alt="image" src="https://github.com/user-attachments/assets/20e93bac-6b21-4355-a45f-3dbd2b9fef27" />

<img width="395" alt="image" src="https://github.com/user-attachments/assets/8f15fb9d-31c2-4a20-883b-20714104b1f0" />


### Документация

Добавьте в зависимости проекта `pdoc` - инструмент для автоматической генерации документации в Python

```
rye add pdoc
```

Выполните команду

```
rye run pdoc src/app
```
