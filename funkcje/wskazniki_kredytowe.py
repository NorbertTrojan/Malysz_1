"""Obliczanie wskaźników finansowych przydatnych w ocenie kredytowej.

Funkcja oblicza wskaźniki kredytowe na podstawie danych finansowych.


Wynikiem jest kopia danych wejściowych z dodatkowymi kolumnami wskaźników.
Brak danych albo dzielenie przez zero daje wartość NaN, a nie sztuczny wynik.
"""


import numpy as np
import pandas as pd


def _wartosc(df: pd.DataFrame, nazwa: str) -> pd.Series:
    """Zwraca kolumnę liczbową albo serię NaN, gdy kolumny nie ma w danych."""
    if nazwa not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[nazwa], errors="coerce")


def _suma(df: pd.DataFrame, *nazwy: str) -> pd.Series:
    """Sumuje dostępne kolumny; brakujące kolumny są traktowane jako brak danych."""
    serie = [_wartosc(df, nazwa) for nazwa in nazwy]
    return pd.concat(serie, axis=1).sum(axis=1, min_count=1)


def _iloraz(licznik: pd.Series, mianownik: pd.Series) -> pd.Series:
    """Dzieli bez generowania nieskończoności dla mianownika równego zero."""
    mianownik = mianownik.where(mianownik != 0)
    return licznik / mianownik


def oblicz_wskazniki(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca kopię ramki ``df`` z dodatkowymi wskaźnikami kredytowymi.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Argument df musi być ramką pandas.DataFrame, "
            f"otrzymano {type(df).__name__}."
        )

    aktywa = _wartosc(df, "Aktywa")
    aktywa_obrotowe = _wartosc(df, "Aktywa_obrotowe")
    aktywa_trwale = _wartosc(df, "Aktywa_trwale")
    kapital_wlasny = _wartosc(df, "Kapital_wlasny")
    zobowiazania_krotkie = _wartosc(df, "Zobowiazania_krotkoterminowe")
    zobowiazania_dlugie = _wartosc(df, "Zobowiazania_dlugoterminowe")
    zobowiazania = zobowiazania_krotkie + zobowiazania_dlugie
    zapasy = _wartosc(df, "Zapasy")
    srodki_pieniezne = _wartosc(df, "Srodki_pieniezne")
    inwestycje_krotkie = _wartosc(df, "Inwestycje_krotkoterminowe")
    naleznosci = _wartosc(df, "Naleznosci_krotkoterminowe")
    sprzedaz = _wartosc(df, "przychody_sprzedazy")
    zysk_operacyjny = _wartosc(df, "zysk_operacyjny")
    zysk_netto = _wartosc(df, "zysk_netto")
    koszty_odsetki = _wartosc(df, "koszty_odsetki")
    amortyzacja = _wartosc(df, "amortyzacja")
    przeplywy_operacyjne = _wartosc(df, "RP_przeplywy_operacyjne")
    kredyty = _wartosc(df, "Kredyty_pozyczki")

    ebitda = zysk_operacyjny + amortyzacja
    gotowka = srodki_pieniezne + inwestycje_krotkie
    kapital_obrotowy_netto = aktywa_obrotowe - zobowiazania_krotkie
    dlug_netto = kredyty - srodki_pieniezne

    wynik = df.copy()
    wynik["wsk_kredyt_plynnosc_biezaca"] = _iloraz(
        aktywa_obrotowe, zobowiazania_krotkie
    )
    wynik["wsk_kredyt_plynnosc_szybka"] = _iloraz(
        aktywa_obrotowe - zapasy, zobowiazania_krotkie
    )
    wynik["wsk_kredyt_plynnosc_gotowkowa"] = _iloraz(
        gotowka, zobowiazania_krotkie
    )
    wynik["wsk_kredyt_kapital_obrotowy_netto"] = kapital_obrotowy_netto
    wynik["wsk_kredyt_zadluzenie_ogolne"] = _iloraz(zobowiazania, aktywa)
    wynik["wsk_kredyt_zadluzenie_do_kapitalu"] = _iloraz(
        zobowiazania, kapital_wlasny
    )
    wynik["wsk_kredyt_udzial_kapitalu_wlasnego"] = _iloraz(
        kapital_wlasny, aktywa
    )
    wynik["wsk_kredyt_dlug_netto_ebitda"] = _iloraz(dlug_netto, ebitda)
    wynik["wsk_kredyt_zobowiazania_ebitda"] = _iloraz(zobowiazania, ebitda)
    wynik["wsk_kredyt_pokrycie_odsetek"] = _iloraz(
        zysk_operacyjny, koszty_odsetki
    )
    wynik["wsk_kredyt_ebitda_pokrycie_odsetek"] = _iloraz(
        ebitda, koszty_odsetki
    )
    wynik["wsk_kredyt_zadluzenie_gotowkowe"] = _iloraz(
        kredyty, przeplywy_operacyjne
    )
    wynik["wsk_kredyt_pokrycie_zobowiazan_gotowka"] = _iloraz(
        przeplywy_operacyjne, zobowiazania
    )
    wynik["wsk_kredyt_roa"] = _iloraz(zysk_netto, aktywa)
    wynik["wsk_kredyt_roe"] = _iloraz(zysk_netto, kapital_wlasny)
    wynik["wsk_kredyt_marza_ebitda"] = _iloraz(ebitda, sprzedaz)
    wynik["wsk_kredyt_marza_operacyjna"] = _iloraz(
        zysk_operacyjny, sprzedaz
    )
    wynik["wsk_kredyt_przeplywy_operacyjne_aktywa"] = _iloraz(
        przeplywy_operacyjne, aktywa
    )
    wynik["wsk_kredyt_rotacja_naleznosci"] = _iloraz(
        sprzedaz, naleznosci
    )
    wynik["wsk_kredyt_cykl_naleznosci_dni"] = _iloraz(
        naleznosci * 365, sprzedaz
    )
    wynik["wsk_kredyt_cykl_konwersji_gotowki"] = (
        wynik["wsk_kredyt_cykl_naleznosci_dni"]
        + _iloraz(zapasy * 365, _wartosc(df, "koszty_sprzedanych_produktow"))
        - _iloraz(
            _suma(
                df,
                "Zobowiazania_dostaw_uslug_12m",
                "Zobowiazania_dostaw_uslug_pow12m",
            )
            * 365,
            _wartosc(df, "koszty_sprzedanych_produktow"),
        )
    )
    return wynik


