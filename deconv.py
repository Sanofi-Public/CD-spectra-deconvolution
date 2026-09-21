import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.stats import chi2, probplot

# -------------------- CONFIG --------------------
class CFG:
    INPUT_XLSX = "spectra_deconv.xlsx"
    SHEET_NAME = "Feuil1"
    X_AXIS_COL = "WL"
    TARGET_COL = None
    EXCLUDE_COLS = "WL"
    REF_FILTER = None
    USE_REGEX = False
    CASE_SENSITIVE = False

    # Bootstrap (optionnal)
    BOOTSTRAP = True
    N_BOOT = 20
    BLOCK_SIZE = 64
    RANDOM_STATE = 1234

    CRITERION = "BIC_RENFORCE"  # "AIC", "BIC" or "BIC_RENFORCE"
    BIC_PENALIZATION = 2.0

    AUTO_NL = False             # auto non-linearity test (AIC/BIC)
    FORCE_NL = True             # force non-linearity
    DELTA_BIC_MAX = 2.0
    BANDS_CONF_INT = 95.0

    OUTPUT_DIR = None


# -------------------- criterion --------------------
def aic(rss, n, p): return n * np.log(rss / n) + 2 * p
def bic(rss, n, p): return n * np.log(rss / n) + p * np.log(n)
def bic_renforce(rss, n, p, facteur_penalisation=2.0):
    return n * np.log(rss / n) + facteur_penalisation * p * np.log(n)

def criterion_value(rss, n, p, method="BIC_", facteur_penalisation=2.0):
    m = method.upper()
    if m == "AIC": return aic(rss, n, p)
    if m == "BIC": return bic(rss, n, p)
    if m == "BIC_RENFORCE": return bic_renforce(rss, n, p, facteur_penalisation)
    raise ValueError("Critère inconnu : " + method)


# -------------------- NNLS (optional bootstrap) --------------------
def _fit_nnls_once(y, X):
    b, _ = nnls(X, y)
    yhat = X @ b
    resid = y - yhat
    return b, yhat, resid

def fit_nnls(y, X_cols, *, bootstrap=True, n_boot=20, block_size=64, random_state=None):
    """
    Retourne:
      - beta_hat / yhat_hat / resid_hat : fit sur données originales
      - beta_boot / yhat_boot           : itérations bootstrap
      - beta_mean/std / yhat_mean       : fit moyenné (bootstrap mean)
    """
    rng = np.random.default_rng(random_state)

    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(c, float) for c in X_cols]) if X_cols else np.zeros((len(y), 0))
    n, p = len(y), X.shape[1]

    if p == 0:
        return {
            "beta_hat": np.zeros(0), "yhat_hat": np.zeros(n), "resid_hat": np.zeros(n),
            "beta_mean": np.zeros(0), "beta_std": np.zeros(0), "yhat_mean": np.zeros(n),
            "beta_boot": np.zeros((0, 0)), "yhat_boot": np.zeros((0, n)),
            "rss_hat": 0.0, "rmse_hat": 0.0, "nRMSD_hat": 0.0, "r2_hat": 0.0
        }

    beta_hat, yhat_hat, resid_hat = _fit_nnls_once(y, X)

    rss_hat = float(np.sum((y - yhat_hat) ** 2))
    rmse_hat = float(np.sqrt(rss_hat / n))
    y_range = float(np.max(y) - np.min(y))
    nRMSD_hat = rmse_hat / y_range if y_range > 0 else 0.0
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2_hat = 1 - rss_hat / tss if tss > 0 else 0.0

    if (not bootstrap) or (n_boot is None) or (int(n_boot) <= 0):
        return {
            "beta_hat": beta_hat, "yhat_hat": yhat_hat, "resid_hat": resid_hat,
            "beta_mean": beta_hat, "beta_std": np.zeros_like(beta_hat), "yhat_mean": yhat_hat,
            "beta_boot": np.zeros((0, p)), "yhat_boot": np.zeros((0, n)),
            "rss_hat": rss_hat, "rmse_hat": rmse_hat, "nRMSD_hat": nRMSD_hat, "r2_hat": r2_hat
        }

    n_boot = int(n_boot)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size doit être > 0.")
    if block_size > n:
        block_size = n

    starts = np.arange(0, max(1, n - block_size + 1))

    beta_boot = np.empty((n_boot, p), float)
    yhat_boot = np.empty((n_boot, n), float)

    for b in range(n_boot):
        idx_blocks = []
        while len(idx_blocks) < n:
            s = rng.choice(starts)
            idx_blocks.extend(range(s, min(s + block_size, n)))
        idx_blocks = idx_blocks[:n]

        y_star = yhat_hat + resid_hat[idx_blocks]
        b_star, _, _ = _fit_nnls_once(y_star, X)

        beta_boot[b, :] = b_star
        yhat_boot[b, :] = X @ b_star

    beta_mean = beta_boot.mean(axis=0)
    beta_std = beta_boot.std(axis=0, ddof=1) if n_boot > 1 else np.zeros_like(beta_mean)
    yhat_mean = X @ beta_mean

    return {
        "beta_hat": beta_hat, "yhat_hat": yhat_hat, "resid_hat": resid_hat,
        "beta_mean": beta_mean, "beta_std": beta_std, "yhat_mean": yhat_mean,
        "beta_boot": beta_boot, "yhat_boot": yhat_boot,
        "rss_hat": rss_hat, "rmse_hat": rmse_hat, "nRMSD_hat": nRMSD_hat, "r2_hat": r2_hat
    }


# -------------------- non-linearity --------------------
def build_nl_features(df, base_cols):
    feats, names = [], []
    for c in base_cols:
        arr = np.asarray(df[c], float)
        feats.append(arr); names.append(c)
    for c in base_cols:
        arr = np.asarray(df[c], float)
        feats.append(arr*arr); names.append(f"{c}^2")
    for i in range(len(base_cols)):
        for j in range(i+1, len(base_cols)):
            a = np.asarray(df[base_cols[i]], float)
            b = np.asarray(df[base_cols[j]], float)
            feats.append(a*b); names.append(f"{base_cols[i]}*{base_cols[j]}")
    return feats, names


# -------------------- contributions --------------------
def compute_contributions_sum_beta_with_disordered(
    feat_names,
    beta_mat,  # (n_boot, p) OR (1, p)
    *,
    rand_prefix="rand",
    disordered_prefix="Disordered_",
    ref_filter=None,
    case_sensitive=False,
):

    beta_mat = np.atleast_2d(beta_mat)
    n_boot, p = beta_mat.shape

    def _starts_with(s, prefix):
        if case_sensitive:
            return s.startswith(prefix)
        return s.lower().startswith(prefix.lower())

    tag = ""
    if ref_filter:
        rf = ref_filter if case_sensitive else ref_filter.lower()

        if rf == "dna":
            tag = "DNA"
        elif rf == "rna":
            tag = "RNA"
        else:
            names_join = " ".join(
                feat_names if case_sensitive
                else [n.lower() for n in feat_names]
            )

            if ("dna" in names_join) and ("rna" not in names_join):
                tag = "DNA"
            elif ("rna" in names_join) and ("dna" not in names_join):
                tag = "RNA"

    disordered_name = (
        f"{disordered_prefix}{tag}"
        if tag else disordered_prefix
    )

    def normalize_base_name(name):
        if _starts_with(name, rand_prefix):
            return disordered_name
        return name

    uniq = []

    for feat in feat_names:

        if "*" in feat:
            left, right = feat.split("*", 1)

            left = normalize_base_name(left)
            right = normalize_base_name(right)

            if left not in uniq:
                uniq.append(left)
            if right not in uniq:
                uniq.append(right)

        elif "^" in feat:
            base = feat.split("^", 1)[0]
            base = normalize_base_name(base)

            if base not in uniq:
                uniq.append(base)

        else:
            base = normalize_base_name(feat)

            if base not in uniq:
                uniq.append(base)

    if n_boot == 0 or p == 0:
        return (
            {u: 0.0 for u in uniq},
            {u: 0.0 for u in uniq},
        )

    contrib_boot = np.zeros((n_boot, len(uniq)), float)

    for b in range(n_boot):

        sums = {u: 0.0 for u in uniq}

        for j, feat in enumerate(feat_names):

            beta = float(beta_mat[b, j])

            if "*" in feat:

                left, right = feat.split("*", 1)

                left = normalize_base_name(left)
                right = normalize_base_name(right)

                if left == right:
                    sums[left] += beta
                else:
                    sums[left] += 0.5 * beta
                    sums[right] += 0.5 * beta

            elif "^" in feat:

                base = feat.split("^", 1)[0]
                base = normalize_base_name(base)

                sums[base] += beta

            else:

                base = normalize_base_name(feat)
                sums[base] += beta

        tot = sum(sums.values())

        if tot > 0:
            for k, u in enumerate(uniq):
                contrib_boot[b, k] = 100.0 * sums[u] / tot

    mean = contrib_boot.mean(axis=0)

    std = (
        contrib_boot.std(axis=0, ddof=1)
        if n_boot > 1
        else np.zeros(contrib_boot.shape[1])
    )

    return (
        {uniq[k]: float(mean[k]) for k in range(len(uniq))},
        {uniq[k]: float(std[k]) for k in range(len(uniq))},
    )

# -------------------- non bootstrap contributions --------------------
def contributions_hat_percent_with_disordered(
    feat_names,
    beta_hat,
    *,
    rand_prefix="rand",
    disordered_prefix="Disordered_",
    ref_filter=None,
    case_sensitive=False,
):
    beta_hat = np.asarray(beta_hat, float).reshape(-1)
    if beta_hat.size != len(feat_names):
        raise ValueError("beta_hat et feat_names n'ont pas la même taille.")

    mean, _std = compute_contributions_sum_beta_with_disordered(
        feat_names, beta_hat.reshape(1, -1),
        rand_prefix=rand_prefix,
        disordered_prefix=disordered_prefix,
        ref_filter=ref_filter,
        case_sensitive=case_sensitive
    )
    return mean


# -------------------- forward selection --------------------
def forward_selection(df, y, candidates, criter, bic_penalty=2.0):
    rem, sel, best_score = list(candidates), [], np.inf
    while rem:
        cand_scores = []
        for v in rem:
            X = [df[c] for c in sel + [v]]
            fit = fit_nnls(y, X, bootstrap=False)
            sc = criterion_value(fit["rss_hat"], len(y), len(sel) + 1, criter, bic_penalty)
            cand_scores.append((sc, v))
        score, v = min(cand_scores, key=lambda t: t[0])
        if score < best_score:
            sel.append(v); best_score = score; rem.remove(v)
        else:
            break
    return sel


# -------------------- diagnostics --------------------
def _durbin_watson(res): return np.sum((res[1:]-res[:-1])**2)/np.sum(res**2)

def _breusch_pagan_approx(yhat, res):
    n=len(yhat); e2=res**2
    X=np.column_stack([np.ones(n), yhat, yhat**2])
    b,*_=np.linalg.lstsq(X, e2, rcond=None)
    e2hat=X@b
    sst=np.sum((e2-np.mean(e2))**2); ssr=np.sum((e2-e2hat)**2)
    r2=1-ssr/sst if sst>0 else 0; k=X.shape[1]-1
    LM=n*r2; p=1-chi2.cdf(LM, df=k); return LM, k, p

def _stem_safe(x,y,**kw):
    try: return plt.stem(x,y,**kw)
    except TypeError:
        kw.pop("use_line_collection", None)
        return plt.stem(x,y,**kw)

def run_nonlinearity_diagnostics(df, xcol, y, yhat, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    res=y-yhat
    dw=_durbin_watson(res); LM,dfbp,pbp=_breusch_pagan_approx(yhat,res)

    (osm,osr),(slope,intcpt,_) = probplot(res, dist="norm", plot=None)
    plt.figure(); plt.scatter(osm,osr,s=8); plt.plot(osm,slope*osm+intcpt,'r',lw=1)
    plt.title("Residuals Q-Q"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"qqplot.png"), dpi=200); plt.close()

    def acf(x,nlags=50):
        x=x-np.mean(x); ac=np.correlate(x,x,mode="full"); ac=ac[ac.size//2:]; ac/=ac[0]; return ac[:nlags+1]
    acv=acf(res, min(100, len(res)-2))
    plt.figure(); _stem_safe(range(len(acv)),acv,use_line_collection=True)
    plt.title("Residuals ACF"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"acf.png"), dpi=200); plt.close()

    pd.DataFrame({"metric":["DW","BP_LM","BP_df","BP_pvalue"],
                  "value":[dw,LM,dfbp,pbp]}).to_csv(os.path.join(out_dir,"diagnostic.csv"), index=False)
    return {"DW":dw,"BP_LM":LM,"BP_df":dfbp,"BP_pvalue":pbp}


# -------------------- plot --------------------
def plot_fit(df, xcol, y, yhat_lin, yhat_nl, target_name,
             nRMSD_lin=None, nRMSD_nl=None,
             out_path=None):
    x = df[xcol].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color="black", lw=1.8, label="Observed spectrum")

    if yhat_lin is not None:
        label_lin = "Linear reconstructed spectrum"
        if nRMSD_lin is not None:
            label_lin += f"\nnRMSD = {nRMSD_lin:.5g}"
        plt.plot(x, yhat_lin, "--", color="crimson", lw=1.8, label=label_lin)

    if yhat_nl is not None:
        label_nl = "Semi-non-linear reconstructed spectrum"
        if nRMSD_nl is not None:
            label_nl += f"\nnRMSD = {nRMSD_nl:.5g}"
        plt.plot(x, yhat_nl, "--", color="royalblue", lw=1.8, label=label_nl)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("CD signal (Δε)")
    plt.title(str(target_name))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, frameon=True)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fit_with_bands(df, xcol, y, target_name,
                        yhat_lin=None, band_lin=None, nRMSD_lin=None,
                        yhat_nl=None, band_nl=None, nRMSD_nl=None,
                        out_path=None):
    x = df[xcol].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    
    plt.axhline(0, color="black", lw=2, alpha=0.5, zorder=0)

    plt.plot(x, y, color="black", lw=1.8, label="Observed spectrum")

    if yhat_lin is not None:
        label_lin = "Linear reconstructed spectrum"
        if nRMSD_lin is not None:
            label_lin += f"\nnRMSD = {nRMSD_lin:.5g}"
        plt.plot(x, yhat_lin, "--", color="crimson", lw=1.8, label=label_lin)
        if band_lin is not None:
            low, high = band_lin
            plt.fill_between(x, low, high, color="crimson", alpha=0.20,
                             label=f"Linear {CFG.BANDS_CONF_INT:.0f}% band")

    if yhat_nl is not None:
        label_nl = "Semi-non-linear reconstructed spectrum"
        if nRMSD_nl is not None:
            label_nl += f"\nnRMSD = {nRMSD_nl:.5g}"
        plt.plot(x, yhat_nl, "--", color="royalblue", lw=1.8, label=label_nl)
        if band_nl is not None:
            low, high = band_nl
            plt.fill_between(x, low, high, color="royalblue", alpha=0.18,
                             label=f"Semi-NL {CFG.BANDS_CONF_INT:.0f}% band")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("CD Signal (Δε)")
    plt.title(str(target_name))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14, frameon=True)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.close()



# -------------------- main --------------------
def main(
    INPUT_XLSX=None,
    TARGET_COL=None,
    EXCLUDE_COLS=None,
    REF_FILTER=None,
    can_do_Z_form=True,
    can_do_A_form=True,
    can_do_quad=True,
    can_do_i_motif=True):
    if INPUT_XLSX:  CFG.INPUT_XLSX = INPUT_XLSX
    if TARGET_COL:  CFG.TARGET_COL = TARGET_COL
    if REF_FILTER is not None: CFG.REF_FILTER = REF_FILTER

    CFG.OUTPUT_DIR = CFG.TARGET_COL
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(CFG.INPUT_XLSX, sheet_name=CFG.SHEET_NAME, engine="openpyxl")

    exclude_cols = set(EXCLUDE_COLS) if EXCLUDE_COLS else set(CFG.EXCLUDE_COLS)

    if not can_do_Z_form:
        exclude_cols.update(["Z-DNA", "Z-RNA"])

    if not can_do_A_form:
        exclude_cols.add("A-DNA")

    if not can_do_quad:
        for i in [c for c in df.columns if "quadruplex" in str(c).lower()]:
            exclude_cols.add(i)

    if not can_do_i_motif:
        exclude_cols.add("DNA i-motif")

    CFG.EXCLUDE_COLS = exclude_cols
    
    if CFG.X_AXIS_COL not in df.columns or CFG.TARGET_COL not in df.columns:
        raise ValueError("X_AXIS_COL ou TARGET_COL introuvable dans les colonnes.")
    y = df[CFG.TARGET_COL].to_numpy(dtype=float)

    candidates = [c for c in df.columns if c not in CFG.EXCLUDE_COLS and c != CFG.TARGET_COL]
    if CFG.REF_FILTER:
        if CFG.USE_REGEX:
            flags = 0 if CFG.CASE_SENSITIVE else re.IGNORECASE
            pattern = re.compile(CFG.REF_FILTER, flags)
            candidates = [c for c in candidates if pattern.search(c)]
        else:
            f = CFG.REF_FILTER
            candidates = [c for c in candidates
                          if (f in c if CFG.CASE_SENSITIVE else f.lower() in c.lower())]
    if not candidates:
        raise ValueError("Aucune colonne de référence valide après filtrage.")
    print(f"Candidats : {candidates}")

    # ---- Forward selection ----
    selected = forward_selection(
        df=df, y=y, candidates=candidates,
        criter=CFG.CRITERION,
        bic_penalty=CFG.BIC_PENALIZATION
    )
    print(f"Variables sélectionnées : {selected}")
    if not selected:
        raise RuntimeError("Aucune variable sélectionnée par la sélection progressive.")

    # ---- Linear fit ----
    X_sel = [df[c] for c in selected]
    fit = fit_nnls(
        y, X_sel,
        bootstrap=CFG.BOOTSTRAP,
        n_boot=CFG.N_BOOT,
        block_size=CFG.BLOCK_SIZE,
        random_state=CFG.RANDOM_STATE
    )

    beta_hat = fit["beta_hat"]
    beta_mean = fit["beta_mean"]
    beta_std = fit["beta_std"]

    yhat_lin_hat  = fit["yhat_hat"]   # true fit
    yhat_lin_mean = fit["yhat_mean"]  # mean fit
    yhat_lin_boot = fit["yhat_boot"]

    rss_lin  = fit["rss_hat"]
    nRMSD_lin = fit["nRMSD_hat"]
    r2_lin   = fit["r2_hat"]

    n, p = len(y), len(selected)
    aic_lin  = aic(rss_lin, n, p)
    bic_lin  = bic(rss_lin, n, p)
    bicr_lin = bic_renforce(rss_lin, n, p, CFG.BIC_PENALIZATION)

    # Linear bootstrap contribution
    if fit["beta_boot"].shape[0] > 0:
        contrib_mean_lin, contrib_std_lin = compute_contributions_sum_beta_with_disordered(
            selected, fit["beta_boot"],
            ref_filter=CFG.REF_FILTER,
            case_sensitive=CFG.CASE_SENSITIVE,
        )
    else:
        contrib_mean_lin, contrib_std_lin = compute_contributions_sum_beta_with_disordered(
            selected, beta_hat.reshape(1, -1),
            ref_filter=CFG.REF_FILTER,
            case_sensitive=CFG.CASE_SENSITIVE,
        )

    # True contribution
    contrib_hat_lin = contributions_hat_percent_with_disordered(
        selected, beta_hat,
        ref_filter=CFG.REF_FILTER,
        case_sensitive=CFG.CASE_SENSITIVE
    )

    # Bootstrap bands
    yhat_lin_low = yhat_lin_high = None
    if CFG.BOOTSTRAP and yhat_lin_boot.shape[0] > 0:
        alpha = max(0.0, min(100.0, 100.0 - CFG.BANDS_CONF_INT))
        q_low, q_high = alpha/2.0, 100.0 - alpha/2.0
        yhat_lin_low  = np.percentile(yhat_lin_boot, q_low,  axis=0)
        yhat_lin_high = np.percentile(yhat_lin_boot, q_high, axis=0)
    elif CFG.BOOTSTRAP:
        print("[WARN] Bootstrap activé mais yhat_lin_boot est vide -> pas de bandes linéaires.")

    # ---- Semi-non-linear ----
    yhat_nl_hat = yhat_nl_mean = None
    yhat_nl_low = yhat_nl_high = None
    nRMSD_nl = None
    contrib_mean_nl = contrib_std_nl = None
    contrib_hat_nl = None

    if CFG.AUTO_NL or CFG.FORCE_NL:
        feats_nl, names_nl = build_nl_features(df, selected)
        fit_nl = fit_nnls(
            y, feats_nl,
            bootstrap=CFG.BOOTSTRAP,
            n_boot=CFG.N_BOOT,
            block_size=CFG.BLOCK_SIZE,
            random_state=CFG.RANDOM_STATE
        )

        yhat_nl_hat  = fit_nl["yhat_hat"]
        yhat_nl_mean = fit_nl["yhat_mean"]
        yhat_nl_boot = fit_nl["yhat_boot"]

        rss_nl = fit_nl["rss_hat"]
        nRMSD_nl = fit_nl["nRMSD_hat"]
        p_nl = len(names_nl)

        aic_nl  = aic(rss_nl, n, p_nl)
        bic_nl  = bic(rss_nl, n, p_nl)
        bicr_nl = bic_renforce(rss_nl, n, p_nl, CFG.BIC_PENALIZATION)

        improve = (aic_nl < aic_lin and bic_nl <= bic_lin + CFG.DELTA_BIC_MAX)

        if CFG.FORCE_NL:
            print("Modèle semi-non-linéaire activé.")
        elif not improve:
            print("Modèle semi-non-linéaire non retenu (pas d’amélioration AIC/BIC).")
            yhat_nl_hat = yhat_nl_mean = nRMSD_nl = None
        else:
            print("Modèle semi-non-linéaire retenu (AIC/BIC améliorés).")

        if yhat_nl_mean is not None:
            # Non-linear bootstrap contribution
            if fit_nl["beta_boot"].shape[0] > 0:
                contrib_mean_nl, contrib_std_nl = compute_contributions_sum_beta_with_disordered(
                    names_nl, fit_nl["beta_boot"],
                    ref_filter=CFG.REF_FILTER,
                    case_sensitive=CFG.CASE_SENSITIVE,
                )
            else:
                contrib_mean_nl, contrib_std_nl = compute_contributions_sum_beta_with_disordered(
                    names_nl, fit_nl["beta_hat"].reshape(1, -1),
                    ref_filter=CFG.REF_FILTER,
                    case_sensitive=CFG.CASE_SENSITIVE,
                )

            # True non-linear contribtion
            contrib_hat_nl = contributions_hat_percent_with_disordered(
                names_nl, fit_nl["beta_hat"],
                ref_filter=CFG.REF_FILTER,
                case_sensitive=CFG.CASE_SENSITIVE
            )

            # Non-linear bands
            if CFG.BOOTSTRAP and yhat_nl_boot.shape[0] > 0:
                alpha = max(0.0, min(100.0, 100.0 - CFG.BANDS_CONF_INT))
                q_low, q_high = alpha/2.0, 100.0 - alpha/2.0
                yhat_nl_low  = np.percentile(yhat_nl_boot, q_low,  axis=0)
                yhat_nl_high = np.percentile(yhat_nl_boot, q_high, axis=0)
            elif CFG.BOOTSTRAP:
                print("[WARN] Bootstrap activé mais yhat_nl_boot est vide -> pas de bandes semi-NL.")

    # Diagnostics
    diag_dir = os.path.join(CFG.OUTPUT_DIR, "diagnostics")
    diags = run_nonlinearity_diagnostics(df, CFG.X_AXIS_COL, y, yhat_lin_hat, diag_dir)

    # -------------------- Excel exports --------------------
    out_xlsx = os.path.join(CFG.OUTPUT_DIR, "deconv_results_beta_sum.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        pd.DataFrame({
            "component": selected,
            "beta_mean": beta_mean,
            "beta_std":  beta_std
        }).to_excel(writer, sheet_name="coeffs_lin", index=False)

        pd.DataFrame({
            "component": list(contrib_mean_lin.keys()),
            "mean_pct":  list(contrib_mean_lin.values()),
            "std_pct":   [contrib_std_lin.get(c, 0.0) for c in contrib_mean_lin]
        }).to_excel(writer, sheet_name="contrib_lin_beta_sum", index=False)

        pd.DataFrame({
            "metric": ["nRMSD_lin", "R2_lin", "AIC_lin", "BIC_lin", "BICR_lin"],
            "value":  [nRMSD_lin, r2_lin, aic_lin,  bic_lin,  bicr_lin]
        }).to_excel(writer, sheet_name="quality_lin", index=False)

        if yhat_nl_mean is not None:
            pd.DataFrame({
                "metric": ["nRMSD_nl", "AIC_nl", "BIC_nl", "BICR_nl"],
                "value":  [nRMSD_nl, aic_nl,  bic_nl,  bicr_nl]
            }).to_excel(writer, sheet_name="quality_nl", index=False)

        if contrib_mean_nl:
            pd.DataFrame({
                "component": list(contrib_mean_nl.keys()),
                "mean_pct":  list(contrib_mean_nl.values()),
                "std_pct":   [contrib_std_nl.get(c, 0.0) for c in contrib_mean_nl]
            }).to_excel(writer, sheet_name="contrib_nl_beta_sum", index=False)

        
        pd.DataFrame({
            CFG.X_AXIS_COL: df[CFG.X_AXIS_COL],
            "Observed":     y,

            "Linear_fit":   yhat_lin_mean,
            "SemiNL_fit":   yhat_nl_mean if yhat_nl_mean is not None else np.nan,

            "Linear_fit_hat": yhat_lin_hat,
            "SemiNL_fit_hat": yhat_nl_hat if yhat_nl_hat is not None else np.nan,

            "Linear_low":   yhat_lin_low  if yhat_lin_low  is not None else np.nan,
            "Linear_high":  yhat_lin_high if yhat_lin_high is not None else np.nan,
            "SemiNL_low":   yhat_nl_low   if yhat_nl_low   is not None else np.nan,
            "SemiNL_high":  yhat_nl_high  if yhat_nl_high  is not None else np.nan
        }).to_excel(writer, sheet_name="fit", index=False)

        
        fit_hat_df = pd.DataFrame({
            CFG.X_AXIS_COL: df[CFG.X_AXIS_COL],
            "Observed": y,
            "Linear_fit_hat": yhat_lin_hat,
            "SemiNL_fit_hat": yhat_nl_hat if yhat_nl_hat is not None else np.nan,
        })

        contrib_hat_rows = []
        for k, v in sorted(contrib_hat_lin.items(), key=lambda kv: kv[1], reverse=True):
            contrib_hat_rows.append({"model": "Linear_hat", "component": k, "pct": v})
        if contrib_hat_nl:
            for k, v in sorted(contrib_hat_nl.items(), key=lambda kv: kv[1], reverse=True):
                contrib_hat_rows.append({"model": "SemiNL_hat", "component": k, "pct": v})
        contrib_hat_df = pd.DataFrame(contrib_hat_rows)

        fit_hat_df.to_excel(writer, sheet_name="fit_hat_and_contrib_hat", index=False, startrow=0)

        
        startrow = len(fit_hat_df) + 3
        contrib_hat_df.to_excel(writer, sheet_name="fit_hat_and_contrib_hat", index=False, startrow=startrow)

        pd.DataFrame({
            "metric": list(diags.keys()),
            "value":  list(diags.values())
        }).to_excel(writer, sheet_name="diagnostics", index=False)

    print(f"[Export] Results saved → {out_xlsx}")

    # -------------------- Figures --------------------
    fig1_name = f"{CFG.TARGET_COL}_deconv_fit_beta_sum.png"
    out_png1 = os.path.join(CFG.OUTPUT_DIR, fig1_name)
    plot_fit(
        df=df, xcol=CFG.X_AXIS_COL, y=y,
        yhat_lin=yhat_lin_mean, yhat_nl=yhat_nl_mean,
        target_name=CFG.TARGET_COL,
        nRMSD_lin=nRMSD_lin, nRMSD_nl=nRMSD_nl,
        out_path=out_png1
    )
    print(f"[Plot] Graph saved → {out_png1}")

    fig2_name = f"{CFG.TARGET_COL}_deconv_fit_beta_sum_with_bounds.png"
    out_png2 = os.path.join(CFG.OUTPUT_DIR, fig2_name)

    band_lin = (yhat_lin_low, yhat_lin_high) if (yhat_lin_low is not None and yhat_lin_high is not None) else None
    band_nl  = (yhat_nl_low, yhat_nl_high) if (yhat_nl_low is not None and yhat_nl_high is not None) else None

    plot_fit_with_bands(
        df=df, xcol=CFG.X_AXIS_COL, y=y, target_name=CFG.TARGET_COL,
        yhat_lin=yhat_lin_mean, band_lin=band_lin, nRMSD_lin=nRMSD_lin,
        yhat_nl=yhat_nl_mean, band_nl=band_nl, nRMSD_nl=nRMSD_nl,
        out_path=out_png2
    )
    print(f"[Plot] Graph with band saved → {out_png2}")

    fig3_name = f"{CFG.TARGET_COL}_deconv_fit_beta_sum_with_bounds_lin.png"
    out_png3 = os.path.join(CFG.OUTPUT_DIR, fig3_name)

    band_lin = (yhat_lin_low, yhat_lin_high) if (yhat_lin_low is not None and yhat_lin_high is not None) else None
    band_nl  = (yhat_nl_low, yhat_nl_high) if (yhat_nl_low is not None and yhat_nl_high is not None) else None

    plot_fit_with_bands(
        df=df, xcol=CFG.X_AXIS_COL, y=y, target_name=CFG.TARGET_COL,
        yhat_lin=yhat_lin_mean, band_lin=band_lin, nRMSD_lin=nRMSD_lin,
        yhat_nl=None, band_nl=band_nl, nRMSD_nl=nRMSD_nl,
        out_path=out_png3
    )
    print(f"[Plot] Graph with band saved → {out_png3}")

    fig4_name = f"{CFG.TARGET_COL}_deconv_fit_beta_sum_with_bounds_nl.png"
    out_png4 = os.path.join(CFG.OUTPUT_DIR, fig4_name)

    band_lin = (yhat_lin_low, yhat_lin_high) if (yhat_lin_low is not None and yhat_lin_high is not None) else None
    band_nl  = (yhat_nl_low, yhat_nl_high) if (yhat_nl_low is not None and yhat_nl_high is not None) else None

    plot_fit_with_bands(
        df=df, xcol=CFG.X_AXIS_COL, y=y, target_name=CFG.TARGET_COL,
        yhat_lin=None, band_lin=band_lin, nRMSD_lin=nRMSD_lin,
        yhat_nl=yhat_nl_mean, band_nl=band_nl, nRMSD_nl=nRMSD_nl,
        out_path=out_png4
    )
    print(f"[Plot] Graph with band saved → {out_png4}")

    print("\n=== Bootstrap linear composition – mean % ± std % ===")
    for comp in contrib_mean_lin:
        print(f"  {comp}: {contrib_mean_lin[comp]:.2f}% ± {contrib_std_lin.get(comp,0):.2f}%")

    print("\n=== True linear composition – % ===")
    for comp, v in sorted(contrib_hat_lin.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {comp}: {v:.2f}%")

    if contrib_mean_nl:
        print("\n=== Bootstrap semi-non-linear composition – mean % ± std % ===")
        for comp in contrib_mean_nl:
            print(f"  {comp}: {contrib_mean_nl[comp]:.2f}% ± {contrib_std_nl.get(comp,0):.2f}%")

    if contrib_hat_nl:
        print("\n=== True semi-non-linear composition  – % ===")
        for comp, v in sorted(contrib_hat_nl.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {comp}: {v:.2f}%")

    print(f"\nBootstrap activated ? {CFG.BOOTSTRAP} (N_BOOT={CFG.N_BOOT}, BLOCK_SIZE={CFG.BLOCK_SIZE})")
    print("\n=== Deconvolution finished ===")
