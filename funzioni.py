import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def stampa_info(dataset):
    print('\n\nCampi descritti dalle colonne del dataset:\n', dataset.columns.str.strip()) 
    print('\nDimensioni del dataset:\t', dataset.shape)
    


def visualizza_distribuzione_conteggio(col, desc='Controllo distribuzione dei valori:'):
    plt.suptitle(desc)
    plt.subplots_adjust(wspace=0.6)
    sns.violinplot(x=col.iloc[:, 0], y=col.iloc[:, 1], hue=None, split=True, inner='quartile')
    plt.show()
    plt.close()

def scala_dati(col, dati, dati_test=None):
    norm = MinMaxScaler()
    stan = StandardScaler()

    if dati_test is None:
        dati_n = dati.copy()
        dati_s = dati.copy()

        dati_n[col] = norm.fit_transform(dati[col])
        dati_s[col] = stan.fit_transform(dati[col])

        return dati_n, dati_s

    else:

        dati_n1 = dati.copy()
        dati_n2 = dati_test.copy()
        dati_s1 = dati.copy()
        dati_s2 = dati_test.copy()

        dati_n1[col] = norm.fit_transform(dati[col])
        dati_n2[col] = norm.transform(dati_test[col])
        dati_s1[col] = stan.fit_transform(dati[col])
        dati_s2[col] = stan.transform(dati_test[col])

        return dati_n1, dati_n2, dati_s1, dati_s2


