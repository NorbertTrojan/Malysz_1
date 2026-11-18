# MODEL CARD — regresja logistyczna

## 1. Opis modelu

### Cel modelu

Model służy do szacowania prawdopodobieństwa niewykonania zobowiązania (*Probability of Default*, PD) przez przedsiębiorstwa na podstawie ich danych finansowych i wskaźników ekonomicznych.

### Zastosowanie

Model wspiera ocenę ryzyka kredytowego, porządkowanie przedsiębiorstw według przewidywanego ryzyka oraz przypisywanie do klas ratingowych. Wynik modelu nie powinien stanowić jedynej podstawy decyzji kredytowej.

### Typ problemu

Klasyfikacja binarna: `default = 1` oznacza wystąpienie defaultu, a `default = 0` jego brak.

### Wykorzystany model

Interpretowalny model bazowy — **regresja logistyczna z regularyzacją L1 (LASSO)** i dodatkowym zerowaniem dodatnich współczynników po dopasowaniu.

## 2. Opis danych

- **Źródło:** `zbior_1.csv`, zbiór udostępniony na potrzeby projektu.
- **Liczba obserwacji:** 3000.
- **Liczba pierwotnych zmiennych objaśniających:** 222 (przed tworzeniem i selekcją cech).
- **Zmienna docelowa:** `default` (0/1).
- **Rozkład targetu:** 262 defaulty (8,73%) oraz 2738 obserwacji bez defaultu (91,27%).
- **Charakter danych:** przekrojowy; brak wymiaru czasowego.
- **Liczba cech z niezerowymi współczynnikami w końcowym modelu:** 15.

### Wstępne przetwarzanie danych

1. Utworzenie dodatkowych wskaźników finansowych z danych źródłowych.
2. Zastąpienie wartości nieskończonych brakami (`NaN`).
3. Odrzucenie cech numerycznych z udziałem braków powyżej 35% lub udziałem zer powyżej 30%.
4. Redukcja nadmiarowych cech — dla par o bezwzględnej korelacji większej niż 0,7 zachowywano zmienną o większej liczbie wartości unikalnych.
5. Wykrywanie wybranych niemonotonicznych zależności i zastępowanie takich cech funkcjami zawiasowymi (10 kwantyli, próg odchylenia stosunku DR/PD równy 0,1, co najmniej dwa kolejne kwantyle).
6. Selekcja cech według jednowymiarowego AUC (minimalna wartość po uwzględnieniu kierunku: 0,55).
7. Transformacja `log1p` dla nieujemnych cech o skośności większej niż 2 oraz odwracanie znaków wybranych cech w celu ujednolicenia kierunku wpływu.
8. Uzupełnianie braków medianą, dodawanie wskaźników braków (`SimpleImputer(add_indicator=True)`) oraz standaryzacja.

Statystyki i reguły selekcji zależne od danych wyznaczano w pipeline na części treningowej odpowiedniego podziału. Końcowy transformer regresji logistycznej wykorzystywał cechy numeryczne; wstępnie analizowane kodowanie kategorii metodą WoE nie było elementem końcowej konfiguracji.

### Podział danych

| Zbiór | Udział | Liczba obserwacji |
|---|---:|---:|
| Treningowy | 70% | 2100 |
| Walidacyjny | 15% | 450 |
| Testowy | 15% | 450 |

Podział był stratyfikowany względem zmiennej `default` (`random_state=2137`). Na zbiorze treningowym zastosowano dodatkowo pięciokrotną stratyfikowaną walidację krzyżową.

## 3. Budowa modelu

### Algorytm i konfiguracja

Zastosowano pipeline `PrzygotowanieCech → SimpleImputer → StandardScaler → LogisticRegressionUjemne`.

| Parametr | Wartość |
|---|---:|
| Regularyzacja | L1 (LASSO) |
| `solver` | `liblinear` |
| `C` | 0,1 |
| `l1_ratio` | 1 |
| `max_iter` | 1000 |
| `random_state` | 2137 |
| Liczba niezerowych współczynników | 15 |

Klasa `LogisticRegressionUjemne` po standardowym dopasowaniu zeruje dodatnie współczynniki. **Nie jest to równoważne optymalizacji regresji logistycznej z ograniczeniami na znaki współczynników.** Ujemny znak pozostawionych współczynników należy interpretować z uwzględnieniem przekształceń i odwracania znaków cech.

### Procedura trenowania

Początkowo zawężano zakres hiperparametrów przez dopasowanie modelu na zbiorze treningowym i porównanie wyników na zbiorze walidacyjnym. Następnie ostateczne parametry wybierano za pomocą pięciokrotnej stratyfikowanej walidacji krzyżowej na zbiorze treningowym. Zbiór testowy pozostawiono do końcowej oceny modelu i kalibracji.

### Miary oceny

ROC AUC, Gini, PR-AUC, KS, Brier score i log-loss. W pięciokrotnej walidacji krzyżowej średnie ROC AUC wyniosło **0,7771**, a odchylenie standardowe **0,0209**.

## 4. Kalibracja

### Cel kalibracji

Dostosowanie średniego prognozowanego PD do zewnętrznego założenia biznesowego dotyczącego długoterminowego poziomu ryzyka portfela: **4%**.

### Metoda kalibracji

Porównano brak dodatkowej kalibracji kształtu z metodami Platta, regresją izotoniczną i beta calibration. Żadna z dodatkowych metod nie poprawiła log-loss na zbiorze walidacyjnym. W dalszej analizie zachowano kształt predykcji modelu, wprowadzając jedynie **korektę poziomu PD** poprzez dodanie stałej do logitu.

Korektę wyznaczono na zbiorze walidacyjnym tak, aby jego średnie prognozowane PD wynosiło 4%; następnie zastosowano niezmieniony parametr do zbioru testowego. Przesunięcie logitu dla regresji logistycznej wyniosło **−1,045**.

### Wyniki kalibracji

| Miara | Wartość |
|---|---:|
| Docelowa średnia PD | 4,00% |
| Średnia PD przed korektą — walidacja | 9,51% |
| Średnia PD przed korektą — test | 9,88% |
| Średnia PD po korekcie — test | 4,11% |
| Obserwowany DR — walidacja | 8,89% |
| Obserwowany DR — test | 8,67% |
| Brier score przed korektą — test | 0,0721 |
| Brier score po korekcie — test | 0,0762 |
| Log-loss przed korektą — test | 0,2567 |
| Log-loss po korekcie — test | 0,2780 |

### Ocena kalibracji

Po korekcie średnie PD w zbiorze testowym wyniosło 4,11%, czyli było zbliżone do docelowego poziomu 4%. Jednocześnie empiryczny odsetek defaultów wynosił 8,67%. Korekta **pogorszyła Brier score i log-loss** na zbiorze testowym, co wskazuje, że realizacja zewnętrznego założenia o tendencji centralnej nie oznacza poprawy kalibracji statystycznej na tej próbie.

Przesunięcie logitu jest ściśle monotoniczną transformacją prawdopodobieństw, dlatego nie zmienia kolejności obserwacji ani ROC AUC. Ocena sensowności docelowego poziomu 4% wymaga odniesienia do reprezentatywnego portfela docelowego.

## 5. Jakość modelu

Poniżej podano **metryki przed korektą do 4%**, aby wszystkie wiersze tabeli odnosiły się do tej samej wersji predykcji.

| Miara | Zbiór walidacyjny | Zbiór testowy |
|---|---:|---:|
| ROC AUC | 0,8061 | 0,7806 |
| Gini | 0,6122 | 0,5613 |
| PR-AUC | 0,3290 | 0,2815 |
| KS | 0,4829 | 0,5396 |
| Log-loss | 0,2482 | 0,2567 |
| Brier score | 0,0704 | 0,0721 |

Na zbiorze testowym ROC AUC wyniosło **0,7806** (bootstrapowy 95% przedział: **[0,6965; 0,8549]**). Jest ono niższe niż na walidacji, lecz zbliżone do średniego AUC z walidacji krzyżowej (0,7771). Wyniki sugerują użyteczną zdolność dyskryminacyjną modelu, przy niepewności wynikającej z ograniczonej liczby przypadków defaultu.

## 6. Interpretowalność i wyjaśnialność

### 6.1. Interpretowalność globalna

**Zastosowane metody:** analiza znaków i wartości współczynników regresji logistycznej po standaryzacji oraz interpretacja ekonomiczna istotnych zmiennych.

Końcowy model miał **15 niezerowych współczynników**. Największe wartości bezwzględne dotyczyły m.in. następujących cech (nazwy zgodne z przekształconymi cechami modelu):

| Cecha | Współczynnik |
|---|---:|
| `-Aktywa_spadek` | −0,3793 |
| `-wsk_zadluzenia_wzrost` | −0,3748 |
| `-wsk_index_skr_zaw` | −0,3387 |
| `-wsk_index_nowe` | −0,2956 |
| `-wsk_ebitda_zobowiazan_odsetki_4_spadek` | −0,2813 |

Wyniki wskazują, że istotną rolę w prognozie odgrywają cechy opisujące zmianę aktywów, zadłużenie, otoczenie branżowe i zdolność pokrywania zobowiązań. Jest to ekonomicznie uzasadnione, jednak część cech była odwracana, logarytmowana lub zastępowana funkcjami zawiasowymi, dlatego kierunek wpływu należy zawsze odczytywać w kontekście konkretnej transformacji.

Znormalizowane współczynniki obrazują wpływ na **logarytm szans defaultu**, nie bezpośrednio na PD. Wartość bezwzględna współczynnika nie jest uniwersalną miarą przyrostu skuteczności predykcyjnej.

**Istotne ograniczenie:** wszystkie pozostawione współczynniki są ujemne także ze względu na zastosowane zerowanie dodatnich wag. Sam ten wynik nie potwierdza niezależnie monotoniczności ekonomicznej.

### 6.2. Interpretowalność lokalna

**Zastosowana metoda:** bezpośrednia dekompozycja logitu na wyraz wolny i wkłady poszczególnych przetworzonych cech: `β_j × x_ij` (gdzie `x_ij` oznacza wartość cechy w reprezentacji podanej regresji).

Przeanalizowano dwie obserwacje ze zbioru testowego: **238** i **249**.

- **Obserwacja 238:** istotne wkłady dotyczyły wskaźników branżowych, zmian aktywów i miar EBITDA w relacji do zobowiązań oraz odsetek. Wskaźniki branżowe są w ograniczonym stopniu kontrolowane przez pojedyncze przedsiębiorstwo, natomiast struktura aktywów i pokrycie zobowiązań mogą stanowić przedmiot pogłębionej analizy finansowej.
- **Obserwacja 249:** największy wkład wiązał się ze zmianą zadłużenia; znaczenie miały także indeksy branżowe i zmiana aktywów. Sugeruje to potrzebę szczegółowej oceny struktury finansowania przedsiębiorstwa.

W analizie wykorzystano wskazany na zajęciach próg decyzyjny **18,6%**, bez wyznaczania go na podstawie rzeczywistej funkcji kosztów. Lokalna dekompozycja wyjaśnia obliczenie predykcji modelu, lecz nie dowodzi, że zmiana wskazanej cechy spowoduje odpowiednią zmianę rzeczywistego ryzyka.

### 6.3. Stabilność i wiarygodność interpretacji

W pięciokrotnej walidacji krzyżowej odchylenie standardowe ROC AUC wyniosło **0,0209**, co opisuje stabilność **jakości modelu**, a nie samoistnie stabilność współczynników lub wyjaśnień. Analizowana dekompozycja logitu jest dokładna dla ustalonego modelu i tej samej reprezentacji wejścia; nie przeprowadzono osobnego ilościowego eksperymentu stabilności rankingów współczynników pomiędzy ponownymi dopasowaniami.

Szczegółowe wykresy i studia przypadków przedstawiono w raporcie technicznym.

## 7. Ograniczenia

- **Liczebność próby:** jedynie 262 defaulty w całym zbiorze; zbiór testowy obejmuje 450 obserwacji, w tym 39 defaultów.
- **Brak wymiaru czasowego:** nie przeprowadzono oceny stabilności w czasie ani walidacji out-of-time.
- **Niepotwierdzona reprezentatywność:** brak gwarancji, że struktura danych odpowiada docelowemu portfelowi kredytowemu.
- **Założenie o liniowości logitu:** zależności modelowane są liniowo względem przekształconych cech, co może nie oddawać wszystkich interakcji i nieliniowości.
- **Selekcja i przekształcenia cech:** mimo bezpośredniej interpretowalności regresji, funkcje zawiasowe, logarytmowanie i odwracanie znaków utrudniają interpretację względem cech źródłowych.
- **Ograniczenie znaków współczynników:** dodatnie współczynniki zerowano po dopasowaniu, bez optymalizacji modelu z narzuconymi ograniczeniami.
- **Korelacja cech:** selekcja nie usuwa całej współzależności i może wpływać na stabilność oszacowanych współczynników.
- **Kalibracja:** narzucony poziom 4% różni się od częstości defaultów w badanej próbie, co pogorszyło Brier score i log-loss.
- **Brak interpretacji przyczynowej:** model ujawnia zależności statystyczne, a nie skutki interwencji w parametry przedsiębiorstwa.

## 8. Ryzyka

- **Data Bias:** próba może być obciążona lub niereprezentatywna względem rzeczywistych wnioskodawców.
- **Model Risk:** możliwe błędne predykcje, przeuczenie i niewykryte zależności nieliniowe.
- **Ryzyko kalibracji:** PD dostosowane do 4% może zaniżać rzeczywistą częstość defaultów w populacji podobnej do testowej.
- **Data Drift:** zmiany charakterystyki finansowej przedsiębiorstw i otoczenia gospodarczego mogą obniżyć skuteczność modelu.
- **Prediction Drift:** rozkład PD i udział poszczególnych ratingów mogą zmieniać się w czasie.
- **Ryzyko interpretacyjne:** znak współczynnika dla przekształconej cechy może być błędnie interpretowany jako wpływ jej pierwotnej wartości; znaczenie może mieć także współliniowość.
- **Ryzyko decyzyjne:** automatyczne odrzucanie lub akceptowanie wniosków wyłącznie na podstawie predykcji może prowadzić do kosztownych błędów.

## 9. Plan monitorowania

Poniższa tabela przedstawia **proponowany**, a nie już wdrożony, plan monitorowania. Metryki wymagające znajomości faktycznego defaultu można wyznaczyć dopiero po upływie odpowiedniego horyzontu obserwacji.

| Obszar | Miara/metoda | Proponowana częstotliwość |
|---|---|---|
| Jakość predykcji | ROC AUC, PR-AUC, KS | Kwartalnie, gdy dostępne są etykiety |
| Kalibracja poziomu | Średnie PD względem celu 4%; średnie PD względem DR | Miesięcznie dla PD, po uzyskaniu etykiet dla DR |
| Jakość kalibracji | Brier score, log-loss, krzywe reliability | Kwartalnie, gdy dostępne są etykiety |
| Data Drift | PSI i zmiany rozkładów kluczowych wskaźników | Miesięcznie |
| Prediction Drift | PSI rozkładu PD, udziały klas ratingowych | Miesięcznie |
| Stabilność modelu | Stabilność współczynników, liczba aktywnych cech, zmienność AUC | Kwartalnie lub przy ponownym trenowaniu |

### Działania w ramach monitorowania

1. Weryfikowanie istotnych odchyleń średniego PD od docelowego poziomu 4% oraz od zaobserwowanego DR.
2. Analizowanie pogorszenia jakości dyskryminacyjnej względem walidacji i wyniku testowego.
3. Kontrola zmian rozkładów wejściowych wskaźników, predykcji i udziałów klas ratingowych.
4. Weryfikowanie znaków i stabilności współczynników po ponownym dopasowaniu modelu.
5. Przegląd zasadności korekty do 4% przy zmianach portfela i założeń biznesowych.
6. Ponowne trenowanie lub rekalkibracja modelu dopiero po analizie przyczyn pogorszenia wyników i ocenie na niezależnej próbie.

## 10. Przeznaczenie i ograniczenia stosowania

Model przeznaczony jest do **wspomagania** oceny ryzyka kredytowego przedsiębiorstw. Jego główną zaletą jest możliwość bezpośredniej analizy współczynników i rozłożenia wyniku pojedynczego przedsiębiorstwa na wkłady cech.

W badaniu regresję logistyczną wybrano jako model końcowy ze względu na lepszą bezpośrednią interpretowalność, mimo nieco niższego ROC AUC względem XGBoost. Nie oznacza to jednak, że model jest gotowy do samodzielnego podejmowania decyzji kredytowych.

Przed wdrożeniem wymagane są w szczególności: walidacja na reprezentatywnej populacji docelowej, ocena stabilności w czasie, weryfikacja adekwatności poziomu PD równego 4% oraz określenie kosztów błędnych decyzji i zasad nadzoru nad modelem.
