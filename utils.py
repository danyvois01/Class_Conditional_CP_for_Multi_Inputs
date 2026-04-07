import numpy as np
import scipy as sc
import torch
import math


## General function
def accuracy_topk(pred, y, k=1):
    # pred : n x K array of float, predictions
    # y : n array of int, true labels
    top = np.argsort(pred, axis=1)[:, ::-1]
    topk = top[:, :k]
    count = 0
    for i, yy in enumerate(y):
        count += yy in topk[i]
    return count / y.shape[0]


## Score functions
def THR(y, y_eval):
    # Threshold
    # y: int array, size n, true classes
    # y_eval : Float array, size n x K
    try:
        return np.choose(y, y_eval.T)
    except ValueError:
        return np.array([y_eval[i, yy] for i, yy in enumerate(y)])


def APS(y, y_eval):
    # y: int array, size n, true classes
    # y_eval : Float array, size n x K
    n, _ = np.shape(y_eval)
    score_bool = y_eval <= (THR(y, y_eval).reshape(n, 1))
    return np.sum(y_eval * score_bool, axis=1)


def APSrd(y, y_eval):
    # y: int array, size n, true classes
    # y_eval : Float array, size n x K
    n, _ = np.shape(y_eval)
    u = np.random.uniform(0, 1, n)
    sy = THR(y, y_eval)
    score_bool = y_eval < (sy.reshape(n, 1))
    return np.sum(y_eval * score_bool, axis=1) + u * sy


def RAPS(y, y_eval, lamb=0.01, kreg=5):
    return APSrd(y, y_eval) - lamb * np.positive(RNK(y, y_eval) - kreg)


def RNK(y, y_eval):
    # rank
    # y: int array, size n, true classes
    # y_eval : Float array, size n x K
    n, _ = np.shape(y_eval)
    score_bool = y_eval <= (THR(y, y_eval).reshape(n, 1))
    return np.sum(score_bool, axis=1)


##


def compute_scores(y_cal_eval, y_cal, y_new_eval, scores, lraps=0.01, kraps=5):
    # y_new_eval: float array of size n_new x K
    # y_cal_eval: float array of size n x K
    # y_cal: int array of size n
    # scores : "THR" or "APS" or "RNK" or "RAPS" or "APSrd"
    n_new, K = y_new_eval.shape
    S_new = np.zeros((n_new, K))
    if scores == "THR":
        S_cal = THR(y_cal, y_cal_eval)
        S_new = y_new_eval
    elif scores == "APS":
        S_cal = APS(y_cal, y_cal_eval)
        for y in range(K):
            S_new[:, y] = APS(np.full(n_new, y, dtype=int), y_new_eval)
    elif scores == "RNK":
        S_cal = RNK(y_cal, y_cal_eval)
        for y in range(K):
            S_new[:, y] = RNK(np.full(n_new, y, dtype=int), y_new_eval)
    elif scores == "RAPS":
        S_cal = RAPS(y_cal, y_cal_eval)
        for y in range(K):
            S_new[:, y] = RAPS(
                np.full(n_new, y, dtype=int), y_new_eval, lamb=lraps, kreg=kraps
            )
    elif scores == "APSrd":
        S_cal = APSrd(y_cal, y_cal_eval)
        for y in range(K):
            S_new[:, y] = APSrd(np.full(n_new, y, dtype=int), y_new_eval)
    else:
        print("Unknown score function")
    return S_cal, S_new


## Score functions on p values
def Area(p):
    # mean of the p values, area (renormalized) below the curve
    # p : Float array, size m x K
    return np.mean(p, axis=0)


def Squared_Area(p):
    # squared area (renormalized) below the curve
    # p : Float array, size m x K
    m = np.shape(p)[0]
    return np.linalg.norm(p, axis=0) / m


def Dist_to_id(p, low=True):
    # p Float array, size m x _
    m = np.shape(p)[0]
    p = np.sort(p, axis=0)
    aux = (
        np.arange(1, m + 1).reshape((m,) + (len(np.shape(p)) - 1) * (1,)) / (m + 1) - p
    )
    if low:
        return aux * (aux > 0)
    else:
        return np.abs(aux)


def Lower_area_to_id(p):
    # (Renormalized) area of the curve below id
    # p : Float array (already ranker), size m x K
    aux = Dist_to_id(p, low=True)
    return np.mean(aux, axis=0)


def Area_to_id(p):
    # (Renormalized) area of the curve below and above id
    # p : Float array (already ranked), size m x K
    aux = Dist_to_id(p, low=False)
    return np.mean(aux, axis=0)


def Squared_Lower_area_to_id(p):
    # (Renormalized) area of the curve below id
    # p : Float array (already ranked), size m x K
    m = np.shape(p)[0]
    aux = Dist_to_id(p, low=True)
    return np.linalg.norm(aux, axis=0) / m


def Squared_Area_to_id(p):
    # (Renormalized) area of the curve below and above id
    # p : Float array (already ranked), size m x K
    m = np.shape(p)[0]
    aux = Dist_to_id(p, low=False)
    return np.linalg.norm(aux, axis=0) / m


def Score_Quantile(p, n):
    # Score of quantile envelope for a calibration set of size n
    # p :Float array (already ranked), size m x N
    # n : integer
    m, N = np.shape(p)
    aux = np.zeros((m, N))
    for mm in range(m):
        cdf = sc.stats.nhypergeom.cdf(np.arange(n + 1), n + m, n, mm + 1)
        aux[mm] = cdf[(n * p[mm]).astype(int)]
    return np.min(aux, axis=0)


def Score_Env_cal(p, ycal, class_size):
    # p :Float array (already ranked), size m x n
    # ycal : int array (labels), size n
    # class_size : int array, size K
    m, n = np.shape(p)
    K = np.shape(class_size)[0]
    aux = np.zeros((m, n))
    for y in range(K):
        ind = ycal == y
        ny = class_size[y]
        for mm in range(m):
            cdfy = sc.stats.nhypergeom.cdf(np.arange(ny + 1), ny + m, ny, mm + 1)
            aux[mm, ind] = cdfy[(ny * p[mm, ind]).astype(int)]
            # aux[m, ind] = sc.stats.nhypergeom.cdf(p[m, ind], ny + m, m, mm + 1)
    return np.min(aux, axis=0)


def Score_Env_cal_2side(p, ycal, class_size):
    # p :Float array (already ranked), size m x n
    # ycal : int array (labels), size n
    # class_size : int array, size K
    m, n = np.shape(p)
    K = np.shape(class_size)[0]
    aux = np.zeros((m, n))
    for y in range(K):
        ind = ycal == y
        ny = class_size[y]
        for mm in range(m):
            cdfy = sc.stats.nhypergeom.cdf(np.arange(ny + 1), ny + m, ny, mm + 1)
            cdfymm = cdfy[(ny * p[mm, ind]).astype(int)]
            aux[mm, ind] = np.minimum(cdfymm, 1 - cdfymm)
    return np.min(aux, axis=0)


def Score_Env_test(p, class_size):
    # p :Float array (already ranked), size m x n x K
    # class_size : int array, size K
    m, n, K = np.shape(p)
    aux = np.zeros((m, n, K))
    for y in range(K):
        ny = class_size[y]
        for mm in range(m):
            cdfy = sc.stats.nhypergeom.cdf(np.arange(ny + 1), ny + m, ny, mm + 1)
            aux[mm, :, y] = cdfy[(ny * p[mm, :, y]).astype(int)]
            # aux[m,:,y] = sc.stats.nhypergeom.cdf(p[m,:,y], ny + m, m, mm+1)
    return np.min(aux, axis=0)


def Score_Env_test_2side(p, class_size):
    # p :Float array (already ranked), size m x n x K
    # class_size : int array, size K
    m, n, K = np.shape(p)
    aux = np.zeros((m, n, K))
    for y in range(K):
        ny = class_size[y]
        for mm in range(m):
            cdfy = sc.stats.nhypergeom.cdf(np.arange(ny + 1), ny + m, ny, mm + 1)
            cdfymm = cdfy[(ny * p[mm, :, y]).astype(int)]
            aux[mm, :, y] = np.minimum(cdfymm, 1 - cdfymm)
    return np.min(aux, axis=0)


def Score_pvalue(p, scorename):
    # p : Float array (already ranked), size m x K
    # scorename : "L1", "L2", "L1Idlow", "L1Id", "L2Idlow", "L2Id"
    p = np.sort(p, axis=0)
    if scorename == "L1":
        return Area(p)
    elif scorename == "L2":
        return Squared_Area(p)
    elif scorename == "L1Idlow":
        return -Lower_area_to_id(p)
    elif scorename == "L1Id":
        return -Area_to_id(p)
    elif scorename == "L2Idlow":
        return -Squared_Lower_area_to_id(p)
    elif scorename == "L2Id":
        return -Squared_Area_to_id(p)


##
def conformal_set(S_cal, y_cal, S_new, alpha, cond=False, randomize=0):
    # S_cal : n_c array of float, calibration scores
    # S_new : (_,K) array of float, test scores for each class
    # randomize : add an uniform random variables to the score
    n_c = S_cal.shape[0]
    K = S_new.shape[-1]
    if randomize != 0:
        S_cal += randomize * np.random.uniform(0, 1, size=S_cal.shape)
        S_new += randomize * np.random.uniform(0, 1, size=S_new.shape)
    if cond:
        quant = np.zeros(K)
        for y in range(K):
            ind = y_cal == y
            k = int((np.sum(ind) + 1) * alpha) - 1
            if k != -1:
                quant[y] = np.sort(S_cal[ind])[k]
            else:
                quant[y] = -np.inf
    else:
        k = int((n_c + 1) * alpha) - 1
        if k != -1:
            quant = np.sort(S_cal)[k]
        else:
            quant = -np.inf
    CP = S_new >= quant
    return CP


def weighted_CP(S_cal, y_cal, S_new, alpha, weights, density):
    # S_cal : n_c array of float, calibration scores
    # S_new : (K) array of float, test scores for each class
    # weights : (n_c,K) array of float, ratio of density for each calibration points (not normalized)
    K = S_new.shape[-1]
    CP = np.zeros(K)
    for y in range(K):
        yind = y_cal == y
        wnew = density(S_new[y], S_cal[yind]) / density(S_new[y], S_cal)
        weightsy = np.append(weights[:, y], wnew)
        weightsy /= np.sum(weightsy)
        Saux = np.append(S_cal, S_new[y])
        # CP[y] = S_new[y] >= np.quantile(
        #   Saux, alpha, method="inverted_cdf", weights=weightsy
        # )
        CP[y] = np.sum(weightsy * (Saux >= S_new[y])) >= alpha
    return CP


def p_value(S_cal, y_cal, S_new, cond=False, randomize=False):
    # S_cal : n_c array of float, calibration scores
    # S_new : (_,K) array of float 2D or 3D, test scores for each class
    # NB: randomize only works for cd with shapeS=2 (interesting case)
    if cond:
        shapeS = np.shape(S_new)
        K = S_new.shape[-1]
        p = np.zeros(shapeS)
        for y in range(K):
            ind = y_cal == y
            if len(shapeS) == 2:
                if randomize:
                    rank = S_new[:, y, None] > S_cal[ind]
                    Ucal = np.random.uniform(0, 1, np.sum(ind))
                    Unew = np.random.uniform(0, 1, shapeS[0])
                    equality = (S_new[:, y, None] == S_cal[ind]) * (
                        Unew[:, None] > Ucal
                    )
                    p[:, y] = np.mean(rank + equality, axis=-1)
                else:
                    p[:, y] = np.mean(S_new[:, y, None] >= S_cal[ind], axis=-1)
            elif len(shapeS) == 3:
                p[:, :, y] = np.mean(S_new[:, :, y, None] >= S_cal[ind], axis=-1)
    else:
        # p-value for all point
        p = np.mean(np.expand_dims(S_new, -1) >= S_cal, axis=-1)
    return p


def p_value_cal_score(class_size, m, N):
    K = np.shape(class_size)[0]
    Scal = np.random.uniform(0, 1, (N, np.sum(class_size)))
    Snew = np.random.uniform(0, 1, (N, m))
    pcal = np.zeros((m, K * N))
    ycal = np.repeat(np.arange(K), N)
    indaux = 0
    for y, ysize in enumerate(class_size):
        for nind in range(N):
            pcal[:, y * N + nind] = np.mean(
                Snew[nind, :, None] >= Scal[nind, indaux : indaux + ysize], axis=1
            )
        indaux += ysize
    return pcal, ycal


def p_value_cal_score_fast(class_size, m, N):
    K = np.shape(class_size)[0]
    pcal = np.zeros((m, K * N))
    ycal = np.repeat(np.arange(K), N)
    for y, ysize in enumerate(class_size):
        pcal[:, y * N : (y + 1) * N] = gen_pvalues_vectorized(
            m, n=ysize, N_try=N, rng=np.random
        ).T
    return pcal, ycal


def randomize(S_cal, S_new):
    # S_cal : array of float, calibration scores
    # S_new : array of float, test scores
    n = S_cal.shape[0]
    m = S_new.shape[0]
    Saux = np.concatenate((S_cal, S_new))
    dist = Saux[:, None] - Saux[:, None].T
    dist = np.abs(dist)
    mask = dist != 0
    distmin = np.min(dist[mask])
    U = np.random.uniform(-1 / 2, 1 / 2, n + m)
    Saux += distmin * U
    return Saux[:n], Saux[n:]


def p_value_rand(S_cal, y_cal, S_new, cond=False):
    # S_cal : n_c array of float, calibration scores
    # S_new : (_,K) array of float 2D or 3D, test scores for each class
    shapeS = np.shape(S_new)
    n_cal = np.shape(S_cal)[0]
    if cond:
        K = S_new.shape[-1]
        p = np.zeros(shapeS)
        for y in range(K):
            ind = y_cal == y
            if len(shapeS) == 2:
                rank = S_new[:, y, None] > S_cal[ind]
                equality = (S_new[:, y, None] == S_cal[ind]) * np.random.binomial(
                    1, 1 / 2, (shapeS[0], np.sum(ind))
                )

                p[:, y] = np.mean(rank + equality, axis=-1)
            elif len(shapeS) == 3:
                p[:, :, y] = np.mean(S_new[:, :, y, None] >= S_cal[ind], axis=-1)
    else:
        # p-value for all point
        rank = np.expand_dims(S_new, -1) > S_cal
        equality = (np.expand_dims(S_new, -1) == S_cal) * np.random.binomial(
            1, 1 / 2, size=np.append(shapeS, n_cal)
        )
        p = np.mean(rank + equality, axis=-1)
    return p


def class_sizes(y, minlength=0):
    return np.bincount(y, minlength=minlength)


##
def length(CP):
    return np.sum(CP, axis=-1)


def coverage(CP, y_true):
    return np.mean(np.choose(y_true, CP.T), axis=-1)


def cond_average(A, y_true, K=None):
    # means of the last coordinate by class
    Aaux = np.transpose(A)
    y_true = np.array(y_true)
    if K is None:
        K = y_true.max() + 1
    cond_mean = np.zeros((K,) + Aaux.shape[1:])
    for y in range(K):
        ind = y_true == y
        cond_mean[y] = np.mean(Aaux[ind], axis=0)
    return np.transpose(cond_mean)


##Combination


def combination_majority_vote(
    CP, method, alpha, class_size=None, compQ=True, Qbin=None, Qnhgeo=None
):
    # method is "MAJ", "BIN", "BetaBIN" or "BetaBINcd" or "MAJex"
    # class_size is needed for BetaBin
    # compQ is True if we want to recompute the quantile, else we use the quantile of Qbin or Qnhgeo
    shapeCP = CP.shape
    CP_comb = np.zeros(shapeCP)
    if len(shapeCP) == 3:
        _, m, _ = shapeCP
        for i in range(m):
            threshold = threshold_aux(
                method,
                i + 1,
                alpha,
                class_size=class_size,
                comp=compQ,
                Qbin=Qbin,
                Qnhgeo=Qnhgeo,
            )
            CP_comb[:, i] = np.sum(CP[:, : (i + 1)], axis=-2) >= threshold
        if method in ["MAJex", "MAJexcd"]:
            for i in range(m):
                CP_comb[:, m - 1 - i] = np.prod(CP_comb[:, : (m - 1 - i)], axis=-2)
    elif len(shapeCP) == 2:
        m, _ = shapeCP
        for i in range(m):
            threshold = threshold_aux(
                method,
                i + 1,
                alpha,
                class_size=class_size,
                comp=compQ,
                Qbin=Qbin,
                Qnhgeo=Qnhgeo,
            )
            CP_comb[i] = np.sum(CP[: (i + 1)], axis=-2) >= threshold
        if method in ["MAJex", "MAJexcd"]:
            for i in range(m):
                CP_comb[m - 1 - i] = np.prod(CP_comb[: (m - 1 - i)], axis=-2)
    return CP_comb


def combination_max_1obs(pred, threshold_proba, threshold_cumsum=None, weights=None):
    # Cumulative sum threshold is not implemented
    m, K = pred.shape
    if weights is None:
        weights = np.ones((m, 1))
    else:
        weights = np.reshape(weights, (m, 1))
    score = np.max(pred * weights, axis=0)
    CP = score >= threshold_proba
    CP2 = np.zeros(K)
    for y in range(K):
        CP2[y] = np.sum(score[score > score[y]])
    CP = CP * (CP2 <= threshold_cumsum)
    return CP


def combination_majority_vote_1obs(
    CP, method, alpha, class_size=None, compQ=True, Qbin=None, Qnhgeo=None
):
    # method is "MAJ", "BIN", "BetaBIN" or "BetaBINcd" or "MAJex"
    # class_size is needed for BetaBin
    # compQ is True if we want to recompute the quantile, else we use the quantile of Qbin or Qnhgeo
    if method == "MAJex":
        CPaux = combination_majority_vote(
            CP,
            method,
            alpha,
            class_size=class_size,
            compQ=compQ,
            Qbin=Qbin,
            Qnhgeo=Qnhgeo,
        )
        return CPaux[-1]
    m, K = CP.shape
    CP_comb = np.zeros(K)
    threshold = threshold_aux(
        method,
        m,
        alpha,
        class_size=class_size,
        comp=compQ,
        Qbin=Qbin,
        Qnhgeo=Qnhgeo,
    )
    CP_comb = np.sum(CP, axis=0) >= threshold

    return CP_comb


def combination_majority_vote_Multobs(
    CP, method, alpha, class_size=None, compQ=True, Qbin=None, Qnhgeo=None
):
    # method is "MAJ", "BIN", "BetaBIN" or "BetaBINcd" or "MAJex"
    # class_size is needed for BetaBin
    # compQ is True if we want to recompute the quantile, else we use the quantile of Qbin or Qnhgeo
    if method in ["MAJex", "MAJexcd"]:
        CPaux = combination_majority_vote(
            CP,
            method,
            alpha,
            class_size=class_size,
            compQ=compQ,
            Qbin=Qbin,
            Qnhgeo=Qnhgeo,
        )
        return CPaux[:, -1]
    n, m, K = CP.shape
    CP_comb = np.zeros((n, K))
    threshold = threshold_aux(
        method,
        m,
        alpha,
        class_size=class_size,
        comp=compQ,
        Qbin=Qbin,
        Qnhgeo=Qnhgeo,
    )
    CP_comb = np.sum(CP, axis=1) >= threshold

    return CP_comb


def threshold_aux(method, m, alpha, class_size=None, comp=True, Qbin=None, Qnhgeo=None):
    if comp:
        return threshold_comp(method, m, alpha, class_size=class_size)
    else:
        return threshold_load(method, m, class_size, Qbin=Qbin, Qnhgeo=Qnhgeo)


def threshold_comp(method, m, alpha, class_size=None):
    # if method == "MAJ" or method == "MAJex":
    if method[:3] == "MAJ":
        threshold = m / 2
    elif method == "BIN":
        threshold = sc.stats.binom.ppf(alpha, m, 1 - alpha)
    elif method == "BetaBIN":
        n_cal = np.sum(class_size)
        if math.ceil((n_cal + 1) * (1 - alpha)) <= n_cal:
            threshold = sc.stats.nhypergeom.ppf(
                alpha, n_cal + m, m, math.ceil((n_cal + 1) * (1 - alpha))
            )
        else:
            threshold = m  # does not matter, all the points have been already accepted
    elif method == "BetaBINcd":
        threshold = []
        for n_cal in class_size:
            if math.ceil((n_cal + 1) * (1 - alpha)) <= n_cal:
                threshold += [
                    sc.stats.nhypergeom.ppf(
                        alpha, n_cal + m, m, math.ceil((n_cal + 1) * (1 - alpha))
                    )
                ]
            else:
                threshold += [m]
        threshold = np.array(threshold)
    return threshold


def threshold_load(method, m, class_size, Qbin=None, Qnhgeo=None):
    # if method == "MAJ" or method == "MAJex":
    if method[:3] == "MAJ":
        threshold = m / 2
    elif method == "BIN":
        threshold = Qbin[m - 1]
    elif method == "BetaBIN":
        n_cal = np.sum(class_size)
        # if math.ceil((n_cal + 1) * (1 - alpha)) <= n_cal:
        #    threshold = Qnhgeo[n_cal-1, m - 1]
        # else:
        #    threshold = m  # does not matter, all the points have been already accepted
        threshold = Qnhgeo[n_cal - 1, m - 1]
    elif method == "BetaBINcd":
        threshold = []
        for n_cal in class_size:
            # if math.ceil((n_cal + 1) * (1 - alpha)) <= n_cal:
            #    threshold += [Qnhgeo[n_cal -1, m - 1]]
            # else:
            #    threshold += [m]
            threshold += [Qnhgeo[n_cal - 1, m - 1]]
        threshold = np.array(threshold)
    return threshold


def conformal_combination(
    alpha, Xcal, ycal, Xnew, method, scores, cond, model, LB_evlp=None
):
    # Xcal : float array (n,d)
    # ycal : int array (n)
    # Xtest: int array (m,d)
    ynew = prediction_np(Xnew, model)
    ycal_eval = prediction_np(Xcal, model)
    S_cal, S_new = compute_scores(ycal_eval, ycal, ynew, scores)
    class_size = class_sizes(ycal)
    if method == "EVLP":
        pval = p_value(S_cal, ycal, S_new, cond=cond)
        if LB_evlp is None:
            print("Missing envelop for EVLP method")
        else:
            cp_set = combination_envelop(pval, LB_evlp, cond, class_size)
    else:
        CP = conformal_set(S_cal, ycal, S_new, alpha, cond=cond)
        cp_set = combination_majority_vote(CP, method, alpha, class_size=class_size)
    return cp_set


def combination_pvalscore(pvaltest, ycal, alpha, scorename, class_size=None, N_try=100):
    # pvaltest : Float array (m,K)
    # ycal : Int array shape K*N (can be ignored if we recreate ycal_cur)
    # alpha : Float in (0,1)
    m, K = pvaltest.shape
    CPcomb = np.zeros((m, K))
    
    if scorename == "ScEnv":
        for mm in range(m):
            m_cur = mm + 1
            pcal_cur, _ = p_value_cal_score_fast(class_size, m_cur, N_try)
            Scal = Score_Env_cal(np.sort(pcal_cur, axis=0), ycal, class_size)
            Snew = Score_Env_test(
                np.sort(pvaltest[:m_cur], axis=0).reshape((m_cur, 1, K)), class_size
            )
            CPcomb[mm, :] = conformal_set(Scal, ycal, Snew, alpha, cond=True)
        return CPcomb
        
    for mm in range(m):
        m_cur = mm + 1
        pcal_cur, ycal_cur = p_value_cal_score_fast(class_size, m_cur, N_try)
        
        Scal = Score_pvalue(pcal_cur, scorename)
        Snew = Score_pvalue(pvaltest[:m_cur], scorename)
        CPcomb[mm, :] = conformal_set(Scal, ycal_cur, Snew, alpha, cond=True)
        
    return CPcomb


def combination_pvalscore_1obs(pvaltest, pcal, ycal, alpha, scorename):
    # pvaltest : Float array (m,K)
    # pcal : Float array shape (m,K*N)
    # ycal : Int array shape K*N
    # alpha : Float in (0,1)
    Scal = Score_pvalue(pcal, scorename)
    Snew = Score_pvalue(pvaltest, scorename)
    CPcomb = conformal_set(Scal, ycal, Snew, alpha, cond=True)
    return CPcomb


def combination_pvalscore_Multobs(
    pvaltest, pcal, ycal, alpha, scorename, class_size=None, test=False
):
    # pvaltest : Float array (n,m,K)
    # pcal : Float array shape (m,K*N)
    # ycal : Int array shape K*N
    # alpha : Float in (0,1)
    pvaltest = np.sort(pvaltest, axis=1)
    pvaltest = np.transpose(pvaltest, (1, 0, 2))
    if scorename == "ScEnv":
        Scal = Score_Env_cal(pcal, ycal, class_size)
        Snew = Score_Env_test(pvaltest, class_size)
    elif scorename == "ScEnv2S":
        Scal = Score_Env_cal_2side(pcal, ycal, class_size)
        Snew = Score_Env_test_2side(pvaltest, class_size)
    else:
        Scal = Score_pvalue(pcal, scorename)
        Snew = Score_pvalue(pvaltest, scorename)
    CPcomb = conformal_set(Scal, ycal, Snew, alpha, cond=True)
    if test:
        argmax = np.argmax(Snew, axis=-1)
        for i, y in enumerate(argmax):
            CPcomb[i, y] = 1
    return CPcomb


##Data generation


def sample_labels(n, K, p=None):
    if p is None:
        y = np.random.random_integers(K, size=n) - 1
    else:
        y = np.random.choice(np.arange(K), size=n, p=p)
    return y


def sample_points(
    n, clt_center, sig_clust, sig_noise, p=None, cov=None, prop_outsider=0
):
    K, d = clt_center.shape
    if cov is None:
        cov = np.eye(d)
    y = sample_labels(n, K, p)
    noise_clust = np.random.multivariate_normal(np.zeros(d), sig_clust * cov, n)
    noise_obs = np.random.multivariate_normal(np.zeros(d), sig_noise * cov, n)
    Xtrue = clt_center[y] + noise_clust
    X = Xtrue + noise_obs
    ind_out = np.random.choice(np.arange(n), size=int(n * prop_outsider), replace=False)
    X[ind_out] = noise_obs[ind_out]
    return X, y


def sample_combination_points(
    n, m, clt_center, sig_clust, sig_noise, p=None, cov=None, prop_outsider=0
):
    K, d = clt_center.shape
    if cov is None:
        cov = np.eye(d)
    y = sample_labels(n, K, p)
    noise_clust_cb = np.random.multivariate_normal(np.zeros(d), sig_clust * cov, n)
    noise_obs_cb = np.random.multivariate_normal(np.zeros(d), sig_noise * cov, (n, m))
    Xtrue_cb = clt_center[y] + noise_clust_cb
    X_cb = np.zeros((n, m, d))
    for i in range(m):
        X_cb[:, i] = noise_obs_cb[:, i] + Xtrue_cb
        ind_out = np.random.choice(np.arange(n), size=int(n * prop_outsider))
        X_cb[ind_out, i] = noise_obs_cb[ind_out, i]
    if n == 1:
        X_cb = X_cb.reshape((m, d))
    return X_cb, y


def gen_pvalues(m, n, N_try=1000):
    Pvalues = []
    for _ in range(N_try):
        Un = np.random.normal(0, 1, size=n)
        Um = np.random.normal(0, 1, size=m)

        p = np.sum(Um[:, None] >= Un, axis=1) / n
        Pvalues.append(np.sort(p))

    return np.array(Pvalues)


# Prediction
def prediction_np(X, model):
    Xaux = torch.FloatTensor(X)
    with torch.no_grad():
        y = torch.softmax(model.forward(Xaux), dim=-1)
    y = y.numpy()
    return y


# Compute quantile
def compute_binomial_quantile(m, alpha):
    Qbin = np.zeros(m)
    for mm in range(m):
        Qbin[mm] = sc.stats.binom.ppf(alpha, mm + 1, 1 - alpha)
    return Qbin


def compute_betabinomial_quantile(m, n, alpha):
    Qnhgeo = np.zeros((n, m))
    for mm in range(m):
        for nn in range(n):
            if math.ceil((nn + 2) * (1 - alpha)) <= nn + 1:
                Qnhgeo[nn, mm] = sc.stats.nhypergeom.ppf(
                    alpha, nn + mm + 2, mm + 1, math.ceil((nn + 2) * (1 - alpha))
                )
            else:
                Qnhgeo[nn, mm] = mm + 1
    return Qnhgeo


# p-values generation


def gen_pvalues_vectorized(m, n=1000, N_try=1000, rng=np.random.default_rng(14)):
    # Generate all permutations at once (N_try, n+m)
    random_matrix = rng.random((N_try, n + m))
    sorted_indices = np.argpartition(random_matrix, m, axis=1)
    # Take first m columns and sort each row
    p = np.sort(sorted_indices[:, :m], axis=1)

    # Broadcast subtraction and division
    Pvalues = (p - np.arange(m)) / n

    return Pvalues


# generation envelope quantile


def gen_envelope_quant(n, m, alpha):
    envelope = np.zeros(m)
    for mm in range(m):
        envelope[mm] = sc.stats.nhypergeom.ppf(alpha, n + m, n, mm + 1)
    return envelope / n
