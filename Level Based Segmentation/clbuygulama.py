def clb():
    """
    Kural tabanlı müşteri sınıflandırması ile yeni müşterilerin getiri
    beklentilerini ve segmenlerini yazdırır.

    Returns
    -------
    None

    Example
    -------
    import pandas as pd
    clb()
    >>> Ülke: TUR - CAN - BRA - USA - GER - FRA
    fra
    >>> İşletim sistemi: IOS - ANDROID
    android
    >>> Cinsiyet: MALE - FEMALE
    female
    >>> Yaş:
    25

                     Beklenen getiri Segment
    Kullanıcı:                 45.43       A
    """
    import pandas as pd
    df = pd.read_csv('customers_level_based.csv')
    ctrs = ['TUR', 'CAN', 'BRA', 'USA', 'GER', 'FRA']
    srcs = ['IOS', 'ANDROID']
    sxs = ['MALE', 'FEMALE']
    ags = [i for i in range(0, 101)]
    age_cat = {"0_18": [i for i in range(19)],
               "19_23": [i for i in range(19, 24)],
               "24_30": [i for i in range(24, 31)],
               "31_40": [i for i in range(31, 41)],
               "41_66": [i for i in range(41, 67)],
               "67_100": [i for i in range(67, 101)]}

    while True:
        cnt_int = input(
            "Ülke: " + " - ".join(ctrs) + " ").upper().strip()
        if cnt_int not in ctrs:
            continue
        break

    while True:
        src_int = input("İşletim sistemi: " + " - ".join(
            srcs) + " ").upper().strip()
        if src_int not in srcs:
            continue
        break

    while True:
        sx_int = input(
            "Cinsiyet: " + " - ".join(sxs) + " ").upper().strip()
        if sx_int not in sxs:
            continue
        break

    while True:
        age_int = int(input('Yaş: '))
        if age_int not in ags:
            continue
        for k, v in age_cat.items():
            if age_int in v:
                ag_cat = k
        break

    new_user = "_".join([cnt_int, src_int, sx_int, ag_cat])
    user_df = df[df.customers_level_based == new_user][
        ['PRICE', 'SEGMENT']]
    try:
        print(pd.DataFrame({'Beklenen getiri': round(user_df.PRICE.values[0], 2),
                            'Segment': user_df.SEGMENT.values[0]},
                           index=["Kullanıcı: "]))
    except:
        print('Veri bulunamadı.')


if __name__ == "__main__":
    clb()
