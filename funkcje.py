import pandas as pd
import numpy as np
def usuwanie_najbardziej_scorrelowanych(df, prog, numeryczne_kolumny=None, bezpieczne_kolumny=None):
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

    print(f"Usunięto {len(do_usuniecia)} kolumn.")
    return df.drop(columns=list(do_usuniecia))