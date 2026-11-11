import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin, clone

from funkcje import usuwanie_najbardziej_scorrelowanych, wykryj_niemonotoniczne, dodaj_cechy_zawiasowe
from wskazniki_kredytowe import oblicz_wskazniki


class PrzygotowanieCech(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        kolumny_do_pominiecia=None,
        prog_korelacji=0.9,
        prog_auc=0.6,
        tylko_wskazniki=False,
        prog_nan=0.5,
        prog_zer=0.5,
        prog_monotonicznosci=0.1
    ):
        self.kolumny_do_pominiecia = kolumny_do_pominiecia
        self.prog_korelacji = prog_korelacji
        self.prog_auc = prog_auc
        self.tylko_wskazniki = tylko_wskazniki
        self.prog_nan = prog_nan
        self.prog_zer = prog_zer
        self.prog_monotonicznosci = prog_monotonicznosci

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
        nie_monotoniczne = wykryj_niemonotoniczne(po_korelacji, y, prog=self.prog_monotonicznosci)
        self.kolumny_niemonotoniczne_ = nie_monotoniczne

        if self.kolumny_niemonotoniczne_:
            po_zawiasach=dodaj_cechy_zawiasowe(po_korelacji, nie_monotoniczne)
        else:
            po_zawiasach= po_korelacji

        self.kolumny_do_usuniecia_, self.kolumny_do_odwrocenia_ = (
            self._odrzuc_kolumny_niskiego_auc(po_zawiasach, y)
        )
        wynik = po_zawiasach.drop(columns=self.kolumny_do_usuniecia_)
        #wynik[self.kolumny_do_odwrocenia_] = -wynik[self.kolumny_do_odwrocenia_]

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
                numeryczne, self.kolumny_niemonotoniczne_
            )

        numeryczne = numeryczne.drop(columns=self.kolumny_do_usuniecia_)
        #numeryczne[self.kolumny_do_odwrocenia_] = -numeryczne[self.kolumny_do_odwrocenia_]
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