
# MODEL CARD

## 1. Opis modelu

### Cel modelu:

Model służy do przewidywania prawdopodobieństwa niewypłacalności (Probability of Default, PD) klientów na podstawie dostępnych danych dotyczących ich charakterystyki i historii kredytowej.

### Zastosowanie:
Model ma wspierać ocenę ryzyka kredytowego i proces podejmowania decyzji dotyczących udzielenia kredytu.

### Typ problemu:
Klasyfikacja binarna - przewidywanie niewypłacalności.

### Wykorzystane modele:
* interpretowalny model bazowy - regresja logistyczna,
* model typu black-box - XGBoost.

## 2. Opis danych
* liczba zmiennych
* liczba obserwacji
* zmienna docelowa
* rozkład zmiennej docelowej
* wstępne przetwarzanie danych
* podział danych

## 3. Budowa modelu
* model bazowy: regresja logistyczna
* model typu black-box: XGBoost
* procedura trenowania: (sposób trenowania i dobór hiperparametrów)
* miary oceny:

## 4. Kalibracja
* cel kalibracji: Dostosowanie prognozowanych wartości PD do docelowego prawdopodobieństwa niewypłacalności w portfelu, wynoszącego 4%.
* metoda kalibracji:...
* docelowa średnia PD: 4%
* średnia prognozowana PD przed kalibracją:...
* średnia prognozowana PD po kalibracji:...
* zaobserwowany odsetek niewypłacalności: (w zbiorze walidacyjnym/testowym)
* ocena kalibracji:?

## 5. Jakość modelu
| Miara | Zbiór walidacyjny | Zbiór testowy |
|---|---|---|
| ROC AUC | | - |
| Gini | | 0 |
| F1-score | | 0|
| Precision | | 0 |
| Recall | | 0 |
| Brier Score |  | 0 |



## 6. Interpretowalność i wyjaśnialność

### 6.1. Wyjaśnienia globalne
- Zastosowane metody: [metody dla modelu interpretowalnego i black-box].
- Najważniejsze zmienne wpływające na predykcje: [lista zmiennych].
- Główne wnioski z analizy globalnej: [krótki opis].

### 6.2. Wyjaśnienia lokalne
- Zastosowane metody: [metody dla obu modeli].
- Analizowane przypadki: [krótki opis wybranych obserwacji].
- Główne wnioski: [krótki opis].

Szczegółowe wyniki, wykresy oraz eksperymenty dotyczące wierności i stabilności wyjaśnień przedstawiono w raporcie technicznym.

## 7. Ograniczenia
* (wynikające z danych)
* (jakość i reprezentatywność próby)
* (ograniczenia kalibracji)
* (ograniczenia interpretowalności)
* brak gwarancji, że zależności wykryte przez model mają charakter przyczynowo-skutkowy

## 8. Ryzyka
* Obciążenie danych (Data Bias): ryzyko wykorzystania obciążonych lub niereprezentatywnych danych treningowych
* Ryzyko związane z modelem (Model Risk): Ryzyko wystąpienia błędnych predykcji i przeuczenia modelu
* Ryzyko kalibracji: prognozowane wartości PD mogą przestać być zgodne z rzeczywistym odsetkiem niewypłacalności
* Zmiana rozkładu danych: zmiany w populacji kredytobiorców mogą obniżyć jakość działania modelu
* Ryzyko decyzyjne: predykcje modelu nie powinny być traktowane jako pełny zamiennik oceny ryzyka kredytowego

## 9. Plan monitorowania
| Obszar | Miara/metoda | Częstotliwość monitorowania |
|---|---|---|
| Jakość predykcji | 0.85 | 0.83 |
| Kalibracja| | 0.66 |
| Jakość kalibracji| 0.72 | 0.69 |
|Data Drift| 0.75 | 0.71 |
| Prediction Drift | 0.69 | 0.67 |
| Stabilność modelu| 0.12 | 0.14 |


### Działania w ramach monitorowania:
* Analiza istotnych odchyleń średniej PD od docelowego poziomu 4%
* Analiza pogorszenia jakości predykcji
* Ponowna ocena kalibracji w przypadku zmiany zaoserwowanego odsetka niewypłacalności
* Ponowne trenowanie lub kalibracja modelu w razie potrzeby

## 10. Przeznaczenie i ograniczenia stosowania
Model jest przeznaczony do wspierania analizy ryzyka kredytowego. Jego predykcje powinny być interpretowane z uwzględnieniem dostępnych danych, założeń dotyczących kalibracji oraz ograniczeń modelu. Model powinien być regularnie monitorowany i nie należy traktować jego wyników jako bezbłędnych.



