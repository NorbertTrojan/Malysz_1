# Analiza ryzyka kredytowego



Projekt zawiera analizę danych finansowych oraz modele przewidujące
prawdopodobieństwo niewypłacalności klienta (PD).

## Wymagania

- Python **3.14.5**
- Pakiety z pliku [`requirements.txt`](requirements.txt)

Instalacja zależności:

```bash
pip install -r requirements.txt
```

## Zawartość projektu

| Plik lub katalog | Opis |
|---|---|
| [`zbior_1.csv`](zbior_1.csv) | Zbiór danych używany w analizie i modelach. |
| [`ekspoloracja_danych.ipynb`](ekspoloracja_danych.ipynb) | Eksploracja danych, sprawdzanie braków i podstawowych zależności. |
| [`Uczenie regresji/przykladowa_regresja_logistyczna.ipynb`](Uczenie%20regresji/przykladowa_regresja_logistyczna.ipynb) | Przykładowy model regresji logistycznej. |
| [`Uczenie regresji/regresja_logistyczna_pipeline.ipynb`](Uczenie%20regresji/regresja_logistyczna_pipeline.ipynb) | Optymalizacja regresji logistycznej |
| [`ucenie black_box/black__box.ipynb`](ucenie%20black_box/black__box.ipynb) | Optymalizacja modelu black-box.|
| [`finalne_modele.ipynb`](finalne_modele.ipynb) | Końcowe modele, ich ocena i test n nich. |
| [`funkcje/funkcje.py`](funkcje/funkcje.py) | Funkcje przygotowania i transformacji cech. |
| [`funkcje/do_pipeline.py`](funkcje/do_pipeline.py) | Klasy i elementy pipeline dla modeli. |
| [`funkcje/wskazniki_kredytowe.py`](funkcje/wskazniki_kredytowe.py) | Obliczanie wskaźników finansowych i kredytowych. |
| [`MODEL_CARD_regresja_logistyczna.md`](MODEL_CARD_regresja_logistyczna.md) | Model card dla regresji logistycznej |
| [`MODEL_CARD_black_box.md`](MODEL_CARD_black_box.md) | Model card dla modelu black-box |

Notebooki można wywoływać w dowolnej kolejności, nie zależą od siebie.

![Malysz](malysz.webp)
