
# MODEL CARD

## 1. Opis modelu

### Cel modelu:

Model służy do przewidywania prawdopodobieństwa niewypłacalności (Probability of Default, PD) klientów na podstawie dostępnych danych dotyczących ich charakterystyki i historii kredytowej.

### Zastosowanie:
Model ma wspierać ocenę ryzyka kredytowego i proces podejmowania decyzji dotyczących udzielenia kredytu.

### Typ problemu:
Klasyfikacja binarna - przewidywanie niewypłacalności.

### Wykorzystane modele:
* model typu black-box - XGBoost.

## 2. Opis danych
* liczba zmiennych
* liczba obserwacji
* zmienna docelowa
* rozkład zmiennej docelowej
* wstępne przetwarzanie danych
* podział danych

## 3. Budowa modelu
* model typu black-box: XGBoost
* procedura trenowania: zbiór danych został podzielony na trzy części: treningową, walidacyjną i testową. Początkowo siatka hiperparametrów była zawężana trenując na zbiorze treningowym i oceniając na walidacyjnym. Aby uniknąć przeuczenia ostateczne hiperparametry zostały wybrane przy pomocy kroswalidacji z5 foldami na samym zbiorze treningowym.
* miary oceny: AUC

## 4. Kalibracja
* cel kalibracji: dostosowanie prognozowanych wartości PD do docelowego prawdopodobieństwa niewypłacalności w portfelu, wynoszącego 4%.
* metoda kalibracji: kalibracja kształtu okazała się zbyteczna, więc jedynie została zastosowana kalibracja poziomu.
* docelowa średnia PD: 4%
* średnia prognozowana PD przed kalibracją: 0.088
* średnia prognozowana PD po kalibracji: 0.043
* zaobserwowany odsetek niewypłacalności: (w zbiorze walidacyjnym/testowym)
* ocena kalibracji:?

## 5. Jakość modelu
| Miara | Zbiór walidacyjny | Zbiór testowy |
|---|---|---|
| ROC AUC | 0.8397| 0.7958 |
| Gini |0.6794	 | 0.5916 |
| Brier Score | 0.0732 |0.0688 |
|PR-AUC|0.4225|0.4888|
|KS|0.5396|0.5029|
|log-loss|0.2327|0.2333|



## 6. Interpretowalność i wyjaśnialność

### 6.1. Wyjaśnienia globalne
- Zastosowane metody: analiza SHAP.
- Najważniejsze zmienne wpływające na predykcje: wsk_index_skr_zaw, wsk_index_nowe, Aktywa, wsk_zadluzenia.
- Główne wnioski z analizy globalnej: [krótki opis].

### 6.2. Wyjaśnienia lokalne
- Zastosowane metody: analiza SHAP.
- Analizowane przypadki: obserwacje ze zbioru testowego znajdujące się tuż pod progiem przyznania kredytu.
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
* Ponowna ocena kalibracji w przypadku zmiany zaobserwowanego odsetka niewypłacalności
* Ponowne trenowanie lub kalibracja modelu w razie potrzeby

## 10. Przeznaczenie i ograniczenia stosowania
Model jest przeznaczony do wspierania analizy ryzyka kredytowego. Jego predykcje powinny być interpretowane z uwzględnieniem dostępnych danych, założeń dotyczących kalibracji oraz ograniczeń modelu. Model powinien być regularnie monitorowany i nie należy traktować jego wyników jako bezbłędnych.



