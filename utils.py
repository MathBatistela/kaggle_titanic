import numpy as np
import itertools
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC


def select_best_knn(ks, X_treino, X_val, y_treino, y_val):

    def treinar_knn(k, X_treino, X_val, y_treino, y_val):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, y_treino)
        pred = knn.predict(X_val)
        return accuracy_score(y_val, pred)

    acuracias_val = Parallel(n_jobs=4)(delayed(treinar_knn)(
        k, X_treino, X_val, y_treino, y_val) for k in ks)

    melhor_val = max(acuracias_val)
    melhor_k = ks[np.argmax(acuracias_val)]
    knn = KNeighborsClassifier(n_neighbors=melhor_k)
    knn.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return knn, melhor_k, melhor_val


def do_cv_knn(X, y, cv_splits, ks):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    predicoes = []

    pgb = tqdm(total=cv_splits, desc='Folds avaliados')

    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(
            X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        knn, _, _ = select_best_knn(
            ks, X_treino, X_val, y_treino, y_val)
        pred = knn.predict(X_teste)

        acuracias.append(accuracy_score(y_teste, pred))
        predicoes.append((y_teste, pred))

        pgb.update(1)

    pgb.close()

    return acuracias, predicoes


def do_hp_cv_knn(X, y, cv_splits, ks):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    predicoes = []

    pgb = tqdm(total=cv_splits, desc='Folds avaliados')

    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)

        params = {'n_neighbors': ks}

        knn = KNeighborsClassifier()

        knn = GridSearchCV(knn, params, cv=StratifiedKFold(n_splits=cv_splits))

        knn.fit(X_treino, y_treino)

        pred = knn.predict(X_teste)

        acuracias.append(accuracy_score(y_teste, pred))
        predicoes.append((y_teste, pred))

        pgb.update(1)

    pgb.close()

    return acuracias, predicoes


def select_best_svm(Cs, gammas, X_treino: np.ndarray, X_val: np.ndarray,
                          y_treino: np.ndarray, y_val: np.ndarray, n_jobs=4):

    def treinar_svm(C, gamma, X_treino, X_val, y_treino, y_val):
        svm = SVC(C=C, gamma=gamma)
        svm.fit(X_treino, y_treino)
        pred = svm.predict(X_val)
        return accuracy_score(y_val, pred)

    combinacoes_parametros = list(itertools.product(Cs, gammas))

    acuracias_val = Parallel(n_jobs=n_jobs)(delayed(treinar_svm)
                                            (c, g, X_treino, X_val, y_treino, y_val) for c, g in combinacoes_parametros)

    melhor_val = max(acuracias_val)

    melhor_comb = combinacoes_parametros[np.argmax(acuracias_val)]
    melhor_c = melhor_comb[0]
    melhor_gamma = melhor_comb[1]

    svm = SVC(C=melhor_c, gamma=melhor_gamma)
    svm.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return svm, melhor_comb, melhor_val


def do_cv_svm(X, y, cv_splits, Cs=[1], gammas=['scale']):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    predicoes = []

    pgb = tqdm(total=cv_splits, desc='Folds avaliados')

    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(
            X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        svm, _, _ = select_best_svm(
            Cs, gammas, X_treino, X_val, y_treino, y_val)
        pred = svm.predict(X_teste)

        acuracias.append(accuracy_score(y_teste, pred))
        predicoes.append((y_teste, pred))

        pgb.update(1)

    pgb.close()

    return acuracias, predicoes


def do_hp_cv_svm(X, y, cv_splits, Cs=[1], gammas=['scale']):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    predicoes = []

    pgb = tqdm(total=cv_splits, desc='Folds avaliados')

    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)

        params = {'C': Cs, 'gamma': gammas}
        svm = SVC()
        svm = GridSearchCV(svm, params, cv=StratifiedKFold(n_splits=cv_splits))
        svm.fit(X_treino, y_treino)

        pred = svm.predict(X_teste)

        acuracias.append(accuracy_score(y_teste, pred))
        predicoes.append((y_teste, pred))

        pgb.update(1)

    pgb.close()

    return acuracias, predicoes
