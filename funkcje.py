import pandas as pd
import numpy as np


def dodaj_cechy_zawiasowe(
    df: pd.DataFrame,
    y: pd.Series,
    kolumny_niemonotoniczne: list[str],
    dolny_kwantyl: float = 0.1,
    gorny_kwantyl: float = 0.9,
    progi: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Dodaje cechy zawiasowe opisujące skrajne wartości wskazanych kolumn.

    Dla każdej kolumny ``cecha`` tworzone są:

    ``cecha_spadek = max(prog_spadek - cecha, 0)``
        wartość rośnie, gdy cecha spada poniżej progu ``dolny_kwantyl``;

    ``cecha_wzrost = max(cecha - prog_wzrost, 0)``
        wartość rośnie, gdy cecha przekracza próg ``gorny_kwantyl``.

    Jeśli ``progi`` nie zostaną podane, są wyznaczane z kwantyli danych w
    ``df``. ``y`` służy do sprawdzenia zgodności obserwacji, ale nie jest
    używany do wyznaczania progów. Funkcję należy więc najpierw wywołać na
    zbiorze treningowym, a zapisane progi przekazać później dla walidacji
    i testu.

    Zwraca:
        Kopię ramki z nowymi kolumnami oraz słownik progów.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df musi być obiektem pandas.DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y musi być obiektem pandas.Series.")
    if len(df) != len(y):
        raise ValueError("df i y muszą mieć tę samą liczbę obserwacji.")
    if not 0 < dolny_kwantyl < gorny_kwantyl < 1:
        raise ValueError(
            "dolny_kwantyl i gorny_kwantyl muszą spełniać "
            "0 < dolny_kwantyl < gorny_kwantyl < 1."
        )

    brakujace = sorted(set(kolumny_niemonotoniczne) - set(df.columns))
    if brakujace:
        raise KeyError(f"Brak kolumn w df: {brakujace}")

    wynik = df.copy()
    ustalone_progi = progi
    wyznaczone_progi: dict[str, dict[str, float]] = {}

    for kolumna in kolumny_niemonotoniczne:
        wartosci = pd.to_numeric(wynik[kolumna], errors="coerce")
        if ustalone_progi is None:
            prog_spadek = wartosci.quantile(dolny_kwantyl)
            prog_wzrost = wartosci.quantile(gorny_kwantyl)
        else:
            if kolumna not in ustalone_progi:
                raise KeyError(f"Brak progów dla kolumny '{kolumna}'.")
            prog_spadek = ustalone_progi[kolumna]["prog_spadek"]
            prog_wzrost = ustalone_progi[kolumna]["prog_wzrost"]

        if pd.isna(prog_spadek) or pd.isna(prog_wzrost):
            raise ValueError(
                f"Kolumna '{kolumna}' nie zawiera żadnych wartości liczbowych."
            )

        wynik[f"{kolumna}_spadek"] = (
            prog_spadek - wartosci
        ).clip(lower=0)
        wynik[f"{kolumna}_wzrost"] = (
            wartosci - prog_wzrost
        ).clip(lower=0)
        wyznaczone_progi[kolumna] = {
            "prog_spadek": float(prog_spadek),
            "prog_wzrost": float(prog_wzrost),
        }

    return wynik, wyznaczone_progi
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

def wykryj_niemonotoniczne(
    df,
    y,
    liczba_kwantyli=10,
    min_obserwacji=50,
    min_zmiana_dr=0.05,
    min_zmian_kierunku=1,
    min_zmian_w_gore=0,
    min_zmian_w_dol=0,
    min_rozstep_dr=0.03
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
        print("Nie znaleziono cech spełniających warunki analizy.")
        return []

    wyniki_niemonotoniczne = wyniki[
        wyniki["niemonotoniczna"]
    ].sort_values(
        ["zmiany_kierunku", "rozstep_DR"],
        ascending=False
    )

    print(f"Liczba przeanalizowanych cech: {len(wyniki)}")
    print(f"Liczba cech niemonotonicznych: {len(wyniki_niemonotoniczne)}")

    display(wyniki_niemonotoniczne.round(4))

    return wyniki_niemonotoniczne["cecha"].tolist()