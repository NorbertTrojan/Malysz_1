import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin, clone

from funkcje.wskazniki_kredytowe import oblicz_wskazniki

def dodaj_cechy_zawiasowe(
    df: pd.DataFrame,
    y: pd.Series | None = None,
    kolumny_niemonotoniczne: list[str] | None = None,
    progi: dict[str, dict[str, float]] | None = None,
    liczba_kandydatow: int = 15,
    cv: int = 5
    
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Dodaje cechy zawiasowe, zastępując nimi oryginalne kolumny.

    Tryb treningowy:
        Jeśli progi=None, wyznacza je na podstawie df i y,
        minimalizując log-loss regresji logistycznej w CV.

    Tryb transformacji:
        Jeśli podano progi, tworzy cechy zawiasowe bez używania y.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df musi być obiektem pandas.DataFrame.")

    if kolumny_niemonotoniczne is None:
        kolumny_niemonotoniczne = list(progi) if progi is not None else []

    if not kolumny_niemonotoniczne:
        raise ValueError("Nie wskazano kolumn niemonotonicznych.")

    if progi is None:
        if y is None:
            raise ValueError(
                "Do wyznaczenia progów wymagane jest y."
            )
        if not isinstance(y, pd.Series):
            raise TypeError("y musi być obiektem pandas.Series.")
        if len(df) != len(y):
            raise ValueError("df i y muszą mieć tę samą liczbę obserwacji.")
        if liczba_kandydatow < 2:
            raise ValueError(
                "liczba_kandydatow musi wynosić co najmniej 2."
            )

    brakujace = sorted(set(kolumny_niemonotoniczne) - set(df.columns))
    if brakujace:
        raise KeyError(f"Brak kolumn w df: {brakujace}")

    wynik = df.copy()
    wyznaczone_progi = {}

    for kolumna in kolumny_niemonotoniczne:
        wartosci = pd.to_numeric(df[kolumna], errors="coerce")

        if progi is not None:
            if kolumna not in progi:
                raise KeyError(f"Brak progów dla kolumny '{kolumna}'.")

            prog_spadek = progi[kolumna]["prog_spadek"]
            prog_wzrost = progi[kolumna]["prog_wzrost"]

        else:
            x_array = wartosci.to_numpy(dtype=float)
            y_array = pd.to_numeric(y, errors="coerce").to_numpy(
                dtype=float
            )

            maska = np.isfinite(x_array) & np.isfinite(y_array)
            x = x_array[maska]
            target = y_array[maska]

            if not np.isin(target, [0, 1]).all():
                raise ValueError("y musi być zmienną binarną (0/1).")

            klasy, liczebnosci = np.unique(
                target, return_counts=True
            )

            if len(klasy) != 2 or liczebnosci.min() < cv:
                raise ValueError(
                    f"Każda klasa musi mieć przynajmniej {cv} obserwacji."
                )

            kandydaci = np.unique(
                np.quantile(
                    x,
                    np.linspace(0.1, 0.9, liczba_kandydatow)
                )
            )

            if len(kandydaci) < 2:
                raise ValueError(
                    f"Za mało różnych wartości w '{kolumna}'."
                )

            najlepszy_wynik = np.inf
            najlepsze_progi = None

            kfold = StratifiedKFold(
                n_splits=cv,
                shuffle=True,
                random_state=2137
            )

            for dolny in kandydaci:
                for gorny in kandydaci:
                    if dolny >= gorny:
                        continue

                    X = np.column_stack([
                        np.maximum(dolny - x, 0),
                        np.maximum(x - gorny, 0)
                    ])

                    wyniki = []

                    for train_idx, valid_idx in kfold.split(X, target):
                        model = LogisticRegression(max_iter=1000)
                        model.fit(X[train_idx], target[train_idx])

                        pred = model.predict_proba(
                            X[valid_idx]
                        )[:, 1]

                        wyniki.append(
                            log_loss(
                                target[valid_idx],
                                pred,
                                labels=[0, 1]
                            )
                        )

                    sredni_log_loss = np.mean(wyniki)

                    if sredni_log_loss < najlepszy_wynik:
                        najlepszy_wynik = sredni_log_loss
                        najlepsze_progi = (dolny, gorny)

            prog_spadek, prog_wzrost = najlepsze_progi

        wynik[f"{kolumna}_spadek"] = (
            prog_spadek - wartosci
        ).clip(lower=0)

        wynik[f"{kolumna}_wzrost"] = (
            wartosci - prog_wzrost
        ).clip(lower=0)

        wyznaczone_progi[kolumna] = {
            "prog_spadek": float(prog_spadek),
            "prog_wzrost": float(prog_wzrost)
        }

    wynik = wynik.drop(columns=kolumny_niemonotoniczne)


    return wynik, wyznaczone_progi

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
    df,
    y,
    liczba_kwantyli=10,
    min_obserwacji=50,
    min_zmiana_dr=0.05,
    min_zmian_kierunku=1,
    min_zmian_w_gore=0,
    min_zmian_w_dol=0,
    min_rozstep_dr=0.03,
    czy_wyswietlic=False
):

    """
    Wykrywa numeryczne cechy o potencjalnie niemonotonicznej zależności
    ze zmienną objaśnianą na podstawie współczynnika DR (default rate)
    obliczanego w kolejnych decylach cechy.

    Dla każdej cechy numerycznej:
    1. Dzieli obserwacje na decyle według wartości cechy.
    2. Oblicza DR (średnią wartość zmiennej objaśnianej) w każdym decylu.
    3. Analizuje zmiany DR pomiędzy sąsiednimi decylami.
    4. Ignoruje zmiany DR mniejsze niż określony próg.
    5. Sprawdza liczbę wzrostów, spadków i zmian kierunku DR.
    6. Weryfikuje minimalną różnicę między maksymalnym a minimalnym DR.

    Cecha zostaje oznaczona jako niemonotoniczna, jeśli jednocześnie
    spełnia wszystkie określone kryteria restrykcyjności.

    Parametry
    ----------
    df : pandas.DataFrame
        Ramka danych zawierająca cechy objaśniające oraz zmienną objaśnianą.

    y : pandas.Series lub array-like
        Zmienna objaśniana, np. binarna 0/1.

    liczba_kwantyli : int, default=10
        Docelowa liczba kwantyli wykorzystywanych do analizy zależności.

    min_obserwacji : int, default=50
        Minimalna liczba obserwacji wymagana w każdym decylu.

    min_zmiana_dr : float, default=0.05
        Minimalna bezwzględna zmiana DR między sąsiednimi decylami,
        aby została uznana za istotny wzrost lub spadek.

    min_zmian_kierunku : int, default=1
        Minimalna liczba zmian kierunku przebiegu DR.

    min_zmian_w_gore : int, default=0
        Minimalna liczba istotnych wzrostów DR między decylami.

    min_zmian_w_dol : int, default=0
        Minimalna liczba istotnych spadków DR między decylami.

    min_rozstep_dr : float, default=0.03
        Minimalna różnica między maksymalnym a minimalnym DR
        w analizowanych decylach.

    Zwraca
    -------
    list
        Lista nazw cech oznaczonych jako potencjalnie niemonotoniczne.
        Lista jest uporządkowana malejąco według liczby zmian kierunku
        oraz rozstępu DR.

    Uwagi
    -----
    - Analizowane są wyłącznie cechy numeryczne.
    - Braki danych są usuwane osobno dla każdej analizowanej cechy.
    - Cechy z niewystarczającą liczbą unikalnych wartości lub obserwacji
      w decylach są pomijane.
    """

    wyniki = []

    # Wyrównanie indeksów
    y = pd.Series(y, index=df.index)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        tmp = pd.DataFrame({
            "x": df[col],
            "y": y
        }).dropna()

        if tmp["x"].nunique() < liczba_kwantyli:
            continue

        try:
            tmp["decyl"] = pd.qcut(
                tmp["x"],
                liczba_kwantyli,
                labels=False,
                duplicates="drop"
            )

            diag = tmp.groupby("decyl").agg(
                DR=("y", "mean"),
                n=("y", "size")
            )

            if len(diag) < liczba_kwantyli:
                continue

            if diag["n"].min() < min_obserwacji:
                continue

            dr = diag["DR"].values
            roznice = np.diff(dr)

            kierunki = np.where(
                roznice > min_zmiana_dr, 1,
                np.where(roznice < -min_zmiana_dr, -1, 0)
            )

            kierunki = kierunki[kierunki != 0]

            wzrosty = (kierunki == 1).sum()
            spadki = (kierunki == -1).sum()

            zmiany_kierunku = (
                np.sum(kierunki[1:] != kierunki[:-1])
                if len(kierunki) > 1 else 0
            )

            rozstep_dr = dr.max() - dr.min()

            niemonotoniczna = (
                wzrosty >= min_zmian_w_gore
                and spadki >= min_zmian_w_dol
                and zmiany_kierunku >= min_zmian_kierunku
                and rozstep_dr >= min_rozstep_dr
            )

            wyniki.append({
                "cecha": col,
                "liczba_decyl": len(diag),
                "min_n": diag["n"].min(),
                "DR_min": dr.min(),
                "DR_max": dr.max(),
                "rozstep_DR": rozstep_dr,
                "wzrosty": wzrosty,
                "spadki": spadki,
                "zmiany_kierunku": zmiany_kierunku,
                "niemonotoniczna": niemonotoniczna
            })

        except ValueError:
            continue

    wyniki = pd.DataFrame(wyniki)

    if wyniki.empty:
        if czy_wyswietlic:
            print("Nie znaleziono cech spełniających warunki analizy.")
        return []

    wyniki_niemonotoniczne = wyniki[
        wyniki["niemonotoniczna"]
    ].sort_values(
        ["zmiany_kierunku", "rozstep_DR"],
        ascending=False
    )
    if czy_wyswietlic:
        print(f"Liczba przeanalizowanych cech: {len(wyniki)}")
        print(f"Liczba cech niemonotonicznych: {len(wyniki_niemonotoniczne)}")

        display(wyniki_niemonotoniczne.round(4))

    return wyniki_niemonotoniczne["cecha"].tolist()


class PrzygotowanieCech(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        kolumny_do_pominiecia=None,
        prog_korelacji=0.9,
        prog_auc=0.6,
        tylko_wskazniki=False,
        prog_nan=0.5,
        prog_zer=0.5,
    ):
        self.kolumny_do_pominiecia = kolumny_do_pominiecia
        self.prog_korelacji = prog_korelacji
        self.prog_auc = prog_auc
        self.tylko_wskazniki = tylko_wskazniki
        self.prog_nan = prog_nan
        self.prog_zer = prog_zer

    def _przygotuj_baze(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Dane wejściowe muszą być obiektem pandas.DataFrame.")

        pomin = self.kolumny_do_pominiecia or []
        brakujace = sorted(set(pomin) - set(X.columns))
        if brakujace:
            raise ValueError(f"Brak wymaganych kolumn do pominięcia: {brakujace}")

        baza = X.drop(columns=pomin).copy()
        kolumny_poczatkowe = set(baza.columns)

        baza = oblicz_wskazniki(baza)

        if self.tylko_wskazniki:
            nowe_kolumny = [
                kolumna for kolumna in baza.columns
                if kolumna not in kolumny_poczatkowe
            ]
            baza = baza[nowe_kolumny]

        return baza

    @staticmethod
    def _kolumny_numeryczne(X):
        kolumny = X.select_dtypes(include=["float"]).columns.tolist()
        if "wsk_liczba_dni_istnienia" in X.columns and "wsk_liczba_dni_istnienia" not in kolumny:
            kolumny.append("wsk_liczba_dni_istnienia")
        return kolumny

    def _odrzuc_kolumny_niskiego_auc(self, X, y):
        rows = []
        for kolumna in X.columns:
            maska = X[kolumna].notna()
            if maska.sum() == 0 or y.loc[maska].nunique() < 2:
                continue
            auc = roc_auc_score(y.loc[maska], X.loc[maska, kolumna])
            rows.append((kolumna, max(auc, 1 - auc), auc >= 0.5))

        if not rows:
            raise ValueError("Nie udało się policzyć AUC dla żadnej cechy.")

        tabela = pd.DataFrame(rows, columns=["cecha", "auc", "zgodny_kierunek"])
        do_usuniecia = tabela.loc[tabela["auc"] < self.prog_auc, "cecha"].tolist()
        do_odwrocenia = tabela.loc[
            (tabela["auc"] >= self.prog_auc) & tabela["zgodny_kierunek"], "cecha"
        ].tolist()
        return do_usuniecia, do_odwrocenia

    def fit(self, X, y):
        y = pd.Series(y, index=X.index if isinstance(X, pd.DataFrame) else None)
        baza = self._przygotuj_baze(X)
        self.kolumny_numeryczne_ = self._kolumny_numeryczne(baza)
        if not self.kolumny_numeryczne_:
            raise ValueError("Nie znaleziono cech numerycznych.")

        numeryczne = baza[self.kolumny_numeryczne_].copy()
        numeryczne = numeryczne.replace([np.inf, -np.inf], np.nan)

        maska_nan = numeryczne.isna().mean() <= self.prog_nan
        maska_zer = numeryczne.eq(0).mean() <= self.prog_zer
        maska_jakosci = maska_nan & maska_zer

        self.kolumny_zbyt_wielu_brakow_ = numeryczne.columns[~maska_jakosci].tolist()
        numeryczne = numeryczne.loc[:, maska_jakosci]

        po_korelacji = usuwanie_najbardziej_scorrelowanych(
            numeryczne, prog=self.prog_korelacji
        )
        self.kolumny_po_korelacji_ = po_korelacji.columns.tolist()
        nie_monotoniczne = wykryj_niemonotoniczne(po_korelacji, y)
        self.kolumny_niemonotoniczne_ = list(nie_monotoniczne)

        if self.kolumny_niemonotoniczne_:
            po_zawiasach, self.progi_ = dodaj_cechy_zawiasowe(
                po_korelacji, y, self.kolumny_niemonotoniczne_
            )
        else:
            po_zawiasach, self.progi_ = po_korelacji, {}

        self.kolumny_do_usuniecia_, self.kolumny_do_odwrocenia_ = (
            self._odrzuc_kolumny_niskiego_auc(po_zawiasach, y)
        )
        wynik = po_zawiasach.drop(columns=self.kolumny_do_usuniecia_)
        wynik[self.kolumny_do_odwrocenia_] = -wynik[self.kolumny_do_odwrocenia_]

        skosnosc = wynik.skew()
        self.kolumny_do_log_ = [
            kolumna for kolumna in wynik.columns
            if skosnosc[kolumna] > 2 and wynik[kolumna].min() >= 0
        ]
        wynik[self.kolumny_do_log_] = np.log1p(wynik[self.kolumny_do_log_])
        self.kolumny_wynikowe_ = wynik.columns.tolist()
        return self

    def transform(self, X):
        baza = self._przygotuj_baze(X)
        numeryczne = baza.reindex(columns=self.kolumny_numeryczne_)
        numeryczne = numeryczne.replace([np.inf, -np.inf], np.nan)
        numeryczne = numeryczne.drop(columns=self.kolumny_zbyt_wielu_brakow_)
        numeryczne = numeryczne.reindex(columns=self.kolumny_po_korelacji_)

        if self.kolumny_niemonotoniczne_:
            numeryczne = dodaj_cechy_zawiasowe(
                numeryczne,
                kolumny_niemonotoniczne=self.kolumny_niemonotoniczne_,
                progi=self.progi_,
            )[0]

        numeryczne = numeryczne.drop(columns=self.kolumny_do_usuniecia_)
        numeryczne[self.kolumny_do_odwrocenia_] = -numeryczne[self.kolumny_do_odwrocenia_]
        numeryczne[self.kolumny_do_log_]=numeryczne[self.kolumny_do_log_].clip(lower=0)
        numeryczne[self.kolumny_do_log_] = np.log1p(numeryczne[self.kolumny_do_log_])
        return numeryczne.reindex(columns=self.kolumny_wynikowe_)



class PrzygotowanieCech_xgb(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        kolumny_do_pominiecia=None,
        prog_korelacji=0.9,
        tylko_wskazniki=False,
        prog_nan=0.5,
        prog_zer=0.5,
        uwzglednij_kategoryczne=True,
    ):
        self.kolumny_do_pominiecia = kolumny_do_pominiecia
        self.prog_korelacji = prog_korelacji
        self.tylko_wskazniki = tylko_wskazniki
        self.prog_nan = prog_nan
        self.prog_zer = prog_zer
        self.uwzglednij_kategoryczne = uwzglednij_kategoryczne

    def _przygotuj_baze(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Dane wejściowe muszą być obiektem pandas.DataFrame.")

        pomin = self.kolumny_do_pominiecia or []
        brakujace = sorted(set(pomin) - set(X.columns))

        if brakujace:
            raise ValueError(
                f"Brak wymaganych kolumn do pominięcia: {brakujace}"
            )

        baza = X.drop(columns=pomin).copy()
        kolumny_poczatkowe = set(baza.columns)

        baza = oblicz_wskazniki(baza)

        if self.tylko_wskazniki:
            nowe_kolumny = [
                kolumna for kolumna in baza.columns
                if kolumna not in kolumny_poczatkowe
            ]
            baza = baza[nowe_kolumny]

        return baza

    @staticmethod
    def _kolumny_numeryczne(X):
        kolumny = X.select_dtypes(include=["float"]).columns.tolist()

        if (
            "wsk_liczba_dni_istnienia" in X.columns
            and "wsk_liczba_dni_istnienia" not in kolumny
        ):
            kolumny.append("wsk_liczba_dni_istnienia")

        return kolumny

    def fit(self, X, y=None):
        baza = self._przygotuj_baze(X)

        self.kolumny_numeryczne_ = self._kolumny_numeryczne(baza)

        self.kolumny_kategoryczne_ = (
            [
                col for col in baza.columns
                if col not in self.kolumny_numeryczne_
            ]
            if self.uwzglednij_kategoryczne
            else []
        )

        self.kategorie_ = {
            col: baza[col].dropna().unique().tolist()
            for col in self.kolumny_kategoryczne_
        }

        numeryczne = baza[self.kolumny_numeryczne_].copy()
        numeryczne = numeryczne.replace([np.inf, -np.inf], np.nan)

        maska_nan = numeryczne.isna().mean() <= self.prog_nan
        maska_zer = numeryczne.eq(0).mean() <= self.prog_zer
        maska_jakosci = maska_nan & maska_zer

        self.kolumny_zbyt_wielu_brakow_ = (
            numeryczne.columns[~maska_jakosci].tolist()
        )

        numeryczne = numeryczne.loc[:, maska_jakosci]

        if numeryczne.shape[1] > 0:
            po_korelacji = usuwanie_najbardziej_scorrelowanych(
                numeryczne,
                prog=self.prog_korelacji
            )
            self.kolumny_po_korelacji_ = po_korelacji.columns.tolist()
        else:
            self.kolumny_po_korelacji_ = []

        self.kolumny_wynikowe_ = (
            self.kolumny_po_korelacji_
            + self.kolumny_kategoryczne_
        )

        if not self.kolumny_wynikowe_:
            raise ValueError("Nie pozostały żadne cechy po filtrowaniu.")

        return self

    def transform(self, X):
        baza = self._przygotuj_baze(X)

        numeryczne = baza.reindex(
            columns=self.kolumny_po_korelacji_
        ).copy()

        numeryczne = numeryczne.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if not self.uwzglednij_kategoryczne:
            return numeryczne

        kategoryczne = pd.DataFrame(index=baza.index)

        for col in self.kolumny_kategoryczne_:
            wartosci = (
                baza[col]
                if col in baza.columns
                else pd.Series(np.nan, index=baza.index)
            )

            kategoryczne[col] = pd.Categorical(
                wartosci,
                categories=self.kategorie_[col]
            )

        wynik = pd.concat(
            [numeryczne, kategoryczne],
            axis=1
        )

        return wynik.reindex(
            columns=self.kolumny_wynikowe_
        )
