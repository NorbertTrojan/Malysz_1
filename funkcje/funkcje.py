from __future__ import annotations

import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin, clone




def usuwanie_najbardziej_scorrelowanych(df, prog, numeryczne_kolumny=None, bezpieczne_kolumny=None, czy_wyswietlic=False):
    """
    Wylicza korelacje między kolumnami w zbiorze danych i usuwa te, które mają korelację większą niż podany próg. 
    Jeżeli nie chcemy usnąć niektórych kolumn, możemy je podać w liście bezpieczne_kolumny.
    Usuwamy kolumny o mniejszej różności (mniejsza liczba unikalnych wartości).
    Funkcja zwraca DataFrame z usuniętymi kolumnami.

    Parametry:
    - df: DataFrame zawierający dane wejściowe
    - prog: próg korelacji, powyżej którego kolumny zostaną usunięte
    - numeryczne_kolumny: lista nazw kolumn numerycznych, które mają być uwzględnione w obliczeniach korelacji. Jeśli None, wszystkie kolumny numeryczne zostaną użyte.
    - bezpieczne_kolumny: lista nazw kolumn, które nie powinny zostać usunięte, nawet jeśli mają wysoką korelację.
    """
    if numeryczne_kolumny is not None:
        X = df[numeryczne_kolumny]
    else:
        X = df.select_dtypes(include=[np.number])

    korelacje = X.corr()

    # Pobranie tylko górnego trójkąta macierzy, bez przekątnej
    maska = np.triu(np.ones(korelacje.shape), k=1).astype(bool)

    pary_korelacji = (
        korelacje.where(maska)
        .stack()
        .reset_index()
    )

    pary_korelacji.columns = ["cecha_1", "cecha_2", "korelacja"]

    # Najsilniejsze zależności: dodatnie i ujemne
    pary_korelacji["abs_korelacja"] = pary_korelacji["korelacja"].abs()

    pary_korelacji=pary_korelacji[pary_korelacji["abs_korelacja"] > prog].sort_values(by="abs_korelacja", ascending=False).reset_index(drop=True)

    do_usuniecia = set()

    for index, row in pary_korelacji.iterrows():
        c_1 = row["cecha_1"]
        c_2 = row["cecha_2"]
        if c_1 not in do_usuniecia:
            if c_2 not in do_usuniecia:
                if bezpieczne_kolumny is not None:
                    c1_bezpieczna = c_1 in bezpieczne_kolumny
                    c2_bezpieczna = c_2 in bezpieczne_kolumny

                    if c1_bezpieczna and c2_bezpieczna:
                        raise ValueError(
                            f"Kolumny {c_1} i {c_2} są silnie skorelowane, "
                            "ale obie są oznaczone jako bezpieczne."
                        )
                    elif c1_bezpieczna:
                        do_usuniecia.add(c_2)
                    elif c2_bezpieczna:
                        do_usuniecia.add(c_1)
                    else:
                        if df[c_1].nunique() > df[c_2].nunique():
                            do_usuniecia.add(c_2)
                        else:
                            do_usuniecia.add(c_1)
                else:
                    if df[c_1].nunique() > df[c_2].nunique():
                        do_usuniecia.add(c_2)
                    else:
                        do_usuniecia.add(c_1)

    if czy_wyswietlic:
        print(f"Usunięto {len(do_usuniecia)} kolumn.")
    return df.drop(columns=list(do_usuniecia))




def wykryj_niemonotoniczne(
    df: pd.DataFrame,
    y: pd.Series,
    liczba_kwantyli: int = 10,
    liczba_kwantyli_z_rzedu: int = 2,
    prog: float = 0.1,
) -> tuple[list[str], list[list[float | str]]]:
    """Wykrywa cechy o kształcie U lub odwróconego U.

    Dla każdej cechy dopasowywany jest jeden model regresji logistycznej na
    całej ramce danych. Następnie dla każdego kwantyla cechy liczony jest
    iloraz ``DR / PD``, gdzie ``DR`` jest średnią wartością ``y``, a ``PD``
    średnią predykcją modelu.

    Cecha jest oznaczana tylko wtedy, gdy wymagany odchył od 1 występuje na
    jednym końcu rozkładu, ale nie na obu. Progi w wyniku są wartościami
    cechy rozdzielającymi kwantyle skrajne od pozostałej części rozkładu.

    Zwraca
    -------
    tuple[list[str], list[list[float | str]]]
        Pierwsza lista zawiera nazwy wykrytych cech. Druga zawiera elementy
        w postaci ``[typ, prog_dolny, prog_gorny]``, w tej samej kolejności.
        ``typ`` przyjmuje wartość ``"zwykle_u"`` albo ``"odwrocone_u"``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df musi być obiektem pandas.DataFrame.")
    if df.empty:
        raise ValueError("df nie może być pusta.")
    if not df.columns.is_unique:
        raise ValueError("Nazwy kolumn w df muszą być unikalne.")
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise TypeError("Wszystkie kolumny df muszą być numeryczne.")
    if not liczba_kwantyli % 1 == 0 or liczba_kwantyli < 2:
        raise ValueError("liczba_kwantyli musi być całkowita i wynosić co najmniej 2.")
    if (
        not liczba_kwantyli_z_rzedu % 1 == 0
        or liczba_kwantyli_z_rzedu < 1
        or liczba_kwantyli_z_rzedu * 2 > liczba_kwantyli
    ):
        raise ValueError(
            "liczba_kwantyli_z_rzedu musi być dodatnia i nie może "
            "przekraczać połowy liczby kwantyli."
        )
    if not np.isfinite(prog) or prog < 0:
        raise ValueError("prog musi być skończoną liczbą nieujemną.")

    y = pd.Series(y, index=df.index)
    if len(y) != len(df):
        raise ValueError("df i y muszą mieć tę samą liczbę obserwacji.")
    y_array = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_array).all() or not np.isin(y_array, [0, 1]).all():
        raise ValueError("y musi być zmienną binarną (0/1) bez braków danych.")
    if np.unique(y_array).size != 2:
        raise ValueError("y musi zawierać obie klasy: 0 i 1.")

    dane_modelowe = df.replace([np.inf, -np.inf], np.nan)
    mediany = dane_modelowe.median()
    if mediany.isna().any():
        puste_kolumny = mediany[mediany.isna()].index.tolist()
        raise ValueError(
            "Nie można uzupełnić medianą kolumn całkowicie pustych: "
            f"{puste_kolumny}."
        )
    x_array = dane_modelowe.fillna(mediany).to_numpy(dtype=float)
    x_array = StandardScaler().fit_transform(x_array)
    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    model.fit(x_array, y_array)
    pd_array = model.predict_proba(x_array)[:, 1]

    oznaczone_kolumny = []
    informacje = []
    gorna_granica = 1 + prog
    dolna_granica = 1 - prog

    for numer_kolumny, kolumna in enumerate(df.columns):
        wartosci = df.iloc[:, numer_kolumny]
        maska_obserwacji = np.isfinite(wartosci.to_numpy(dtype=float))
        wartosci_do_analizy = wartosci.loc[maska_obserwacji]
        if wartosci_do_analizy.nunique() < liczba_kwantyli:
            continue

        try:
            grupy, krawedzie = pd.qcut(
                wartosci_do_analizy,
                q=liczba_kwantyli,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
        except ValueError:
            continue

        liczba_grup = len(krawedzie) - 1
        if liczba_grup != liczba_kwantyli:
            continue

        tabela = pd.DataFrame(
            {
                "grupa": grupy.to_numpy(),
                "y": y_array[maska_obserwacji],
                "pd": pd_array[maska_obserwacji],
            }
        ).groupby("grupa", sort=True)
        dr = tabela["y"].mean().to_numpy()
        pd_sr = tabela["pd"].mean().to_numpy()
        stosunek = np.divide(
            dr,
            pd_sr,
            out=np.full_like(dr, np.nan, dtype=float),
            where=pd_sr != 0,
        )
        if not np.isfinite(stosunek).all():
            continue

        pierwsze = stosunek[:liczba_kwantyli_z_rzedu]
        ostatnie = stosunek[-liczba_kwantyli_z_rzedu:]
        wysoki_dol = np.all(pierwsze > gorna_granica)
        wysoki_gora = np.all(ostatnie > gorna_granica)
        niski_dol = np.all(pierwsze < dolna_granica)
        niski_gora = np.all(ostatnie < dolna_granica)

        if wysoki_dol ^ wysoki_gora:
            typ = "zwykle_u"
            lewy_skrajny = wysoki_dol
            ekstremum = int(np.argmin(stosunek))
            warunek = stosunek > gorna_granica
        elif niski_dol ^ niski_gora:
            typ = "odwrocone_u"
            lewy_skrajny = niski_dol
            ekstremum = int(np.argmax(stosunek))
            warunek = stosunek < dolna_granica
        else:
            continue

        if lewy_skrajny:
            indeks = 0
            while indeks + 1 < liczba_grup and warunek[indeks + 1]:
                indeks += 1
            prog_dolny = krawedzie[indeks + 1]
            prog_gorny = krawedzie[min(ekstremum + 1, liczba_grup)]
        else:
            indeks = liczba_grup - 1
            while indeks > 0 and warunek[indeks - 1]:
                indeks -= 1
            prog_dolny = krawedzie[max(ekstremum, 0)]
            prog_gorny = krawedzie[indeks]

        oznaczone_kolumny.append(kolumna)
        informacje.append([typ, float(prog_dolny), float(prog_gorny)])

    return oznaczone_kolumny, informacje


def dodaj_cechy_zawiasowe(
    df: pd.DataFrame,
    dane_niemonotoniczne: tuple[list[str], list[list[float | str]]],
) -> pd.DataFrame:
    """Zastępuje wykryte cechy U dwiema cechami zawiasowymi.

    ``dane_niemonotoniczne`` powinno być wynikiem funkcji
    :func:`wykryj_niemonotoniczne`. Braki danych w przekształcanej kolumnie
    pozostają brakami również w obu nowych kolumnach.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df musi być obiektem pandas.DataFrame.")
    if not isinstance(dane_niemonotoniczne, (tuple, list)):
        raise TypeError(
            "dane_niemonotoniczne musi być krotką "
            "(kolumny, informacje)."
        )
    if len(dane_niemonotoniczne) != 2:
        raise ValueError(
            "dane_niemonotoniczne musi zawierać kolumny i informacje."
        )

    kolumny, informacje = dane_niemonotoniczne
    if len(kolumny) != len(informacje):
        raise ValueError(
            "Liczba kolumn musi odpowiadać liczbie zestawów informacji."
        )
    if len(set(kolumny)) != len(kolumny):
        raise ValueError("Kolumny w danych wejściowych muszą być unikalne.")

    brakujace = sorted(set(kolumny) - set(df.columns))
    if brakujace:
        raise KeyError(f"Brak kolumn w df: {brakujace}")

    wynik = df.copy()
    for kolumna, informacja in zip(kolumny, informacje):
        if not isinstance(informacja, (list, tuple)) or len(informacja) != 3:
            raise ValueError(
                f"Informacja dla kolumny '{kolumna}' musi mieć format "
                "[typ, prog_dolny, prog_gorny]."
            )

        typ, prog_dolny, prog_gorny = informacja
        if typ not in {"zwykle_u", "odwrocone_u"}:
            raise ValueError(
                f"Nieznany typ '{typ}' dla kolumny '{kolumna}'."
            )
        if not np.isfinite(prog_dolny) or not np.isfinite(prog_gorny):
            raise ValueError(
                f"Progi dla kolumny '{kolumna}' muszą być skończone."
            )
        if prog_dolny > prog_gorny:
            raise ValueError(
                f"Prog_dolny nie może być większy od prog_gorny "
                f"dla kolumny '{kolumna}'."
            )

        dynamika = pd.to_numeric(wynik[kolumna], errors="coerce")
        if typ == "zwykle_u":
            spadek = (prog_dolny - dynamika).clip(lower=0)
            wzrost = (dynamika - prog_gorny).clip(lower=0)
        else:
            wzrost = (dynamika - prog_dolny).clip(lower=0)
            spadek = (prog_gorny - dynamika).clip(lower=0)

        wynik[f"{kolumna}_spadek"] = spadek
        wynik[f"{kolumna}_wzrost"] = wzrost
        wynik = wynik.drop(columns=kolumna)

    return wynik
