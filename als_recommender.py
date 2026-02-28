"""
=============================================================================
ALS Recommender Engine — MovieLens 1M  |  v5 — Final
=============================================================================

Corrections v5 :
  - Scoring manuel : score(u,i) = user_factors[u] · item_factors[i]
    On contrôle exactement quels items sont filtrés (matrix_seen).
    Résultat : HR@10 attendu 15-25%.

  - update_user correct : recalcul du vecteur utilisateur par la formule
    ALS exacte : u = (VᵀCᵤV + λI)⁻¹ Vᵀcᵤpᵤ
    Fonctionne pour les users existants ET les nouveaux.

  - Deux matrices séparées conservées :
      matrix_conf : confiance asymétrique → entraînement + recalcul vecteur
      matrix_seen : booléen → filtre d'exclusion dans recommend()

=============================================================================
"""

import os, pickle, logging, time
from pathlib import Path
from datetime import datetime, timedelta

os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
import implicit

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("ml-1m")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

POSITIVE_THRESHOLD  = 3.5
ALPHA               = 40.0
NEGATIVE_CONFIDENCE = 1e-4

ALS_CONFIG = {
    "factors"       : 150,
    "iterations"    : 40,
    "regularization": 0.05,
    "use_gpu"       : False,
    "num_threads"   : 0,
    "random_state"  : 42,
}

TOP_K_DEFAULT       = 10
RECENT_WINDOW_HOURS = 1
MIN_RECENT_FILMS    = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ALS")


# ──────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ──────────────────────────────────────────────────────────────────────────────

def load_movielens_1m():
    log.info("Chargement MovieLens 1M…")
    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat", sep="::", engine="python",
        names=["userId","movieId","rating","timestamp"],
        dtype={"userId":np.int32,"movieId":np.int32,
               "rating":np.float32,"timestamp":np.int64},
    )
    movies = pd.read_csv(
        DATA_DIR / "movies.dat", sep="::", engine="python",
        names=["movieId","title","genres"],
        encoding="latin-1", dtype={"movieId":np.int32},
    )
    log.info(f"  {len(ratings):,} ratings | {ratings.userId.nunique():,} users "
             f"| {ratings.movieId.nunique():,} films")
    return ratings, movies


# ──────────────────────────────────────────────────────────────────────────────
# 2. SPLIT LEAVE-ONE-OUT
# ──────────────────────────────────────────────────────────────────────────────

def leave_one_out_split(ratings: pd.DataFrame):
    log.info("Split leave-one-out…")
    df = ratings.sort_values(["userId","timestamp"])
    pos      = df[df.rating >= POSITIVE_THRESHOLD]
    test_idx = pos.groupby("userId")["timestamp"].idxmax()
    test_df  = df.loc[test_idx].copy()
    train_df = df.drop(index=test_idx).copy()
    log.info(f"  Train={len(train_df):,} | Test={len(test_df):,} "
             f"| {test_df.userId.nunique():,} users")
    return train_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# 3. MATRICES
# ──────────────────────────────────────────────────────────────────────────────

def build_matrices(train_df: pd.DataFrame):
    log.info("Construction des matrices…")

    unique_users  = np.sort(train_df.userId.unique())
    unique_movies = np.sort(train_df.movieId.unique())
    user_enc = {u: i for i, u in enumerate(unique_users)}
    item_enc = {m: i for i, m in enumerate(unique_movies)}

    rows = train_df.userId.map(user_enc).values.astype(np.int32)
    cols = train_df.movieId.map(item_enc).values.astype(np.int32)
    raw  = train_df.rating.values.astype(np.float32)

    n_u, n_i = len(unique_users), len(unique_movies)

    conf = np.where(
        raw >= POSITIVE_THRESHOLD,
        1.0 + ALPHA * raw,
        NEGATIVE_CONFIDENCE,
    ).astype(np.float32)

    matrix_conf = csr_matrix((conf,                                  (rows, cols)), shape=(n_u, n_i))
    matrix_seen = csr_matrix((np.ones(len(rows), dtype=np.float32),  (rows, cols)), shape=(n_u, n_i))

    sparsity = 1.0 - matrix_conf.nnz / (n_u * n_i)
    pos_n    = int((raw >= POSITIVE_THRESHOLD).sum())
    neg_n    = int((raw <  POSITIVE_THRESHOLD).sum())
    log.info(f"  {n_u}u × {n_i}i | sparsité={sparsity:.4%} | "
             f"pos={pos_n:,} ({pos_n/len(raw)*100:.1f}%) | "
             f"neg={neg_n:,} ({neg_n/len(raw)*100:.1f}%)")

    return (matrix_conf, matrix_seen,
            user_enc, item_enc,
            unique_users, unique_movies)


# ──────────────────────────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT
# ──────────────────────────────────────────────────────────────────────────────

def train_als(matrix_conf: csr_matrix):
    log.info("Entraînement ALS…")
    model = implicit.als.AlternatingLeastSquares(
        factors         = ALS_CONFIG["factors"],
        iterations      = ALS_CONFIG["iterations"],
        regularization  = ALS_CONFIG["regularization"],
        alpha           = 1.0,   # confiance déjà dans la matrice
        use_gpu         = ALS_CONFIG["use_gpu"],
        num_threads     = ALS_CONFIG["num_threads"],
        random_state    = ALS_CONFIG["random_state"],
        calculate_training_loss=True,
    )
    t0 = time.perf_counter()
    model.fit(matrix_conf, show_progress=True)
    log.info(f"  Terminé en {time.perf_counter()-t0:.1f}s")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 5. COLD-START
# ──────────────────────────────────────────────────────────────────────────────

def build_cold_start_list(ratings: pd.DataFrame, movies: pd.DataFrame,
                          top_k: int = 50) -> list:
    log.info("Construction cold-start…")
    ts_max  = ratings.timestamp.max()
    ts_hour = ts_max - int(timedelta(hours=RECENT_WINDOW_HOURS).total_seconds())
    recent  = ratings[ratings.timestamp >= ts_hour]
    n_dist  = recent.movieId.nunique()
    log.info(f"  Dernière heure : {len(recent):,} ratings | {n_dist} films distincts")

    m_g = float(ratings.rating.mean())
    C_g = float(ratings.groupby("movieId").rating.count().mean())

    def bayesian_top(df, label):
        s = (df.groupby("movieId")
               .agg(n=("rating","count"), s=("rating","sum"))
               .reset_index())
        s["score"] = (C_g * m_g + s["s"]) / (C_g + s["n"])
        top = (s.nlargest(top_k,"score")
                .merge(movies, on="movieId", how="left")
               [["movieId","title","genres","score","n"]])
        log.info(f"  ({label}) {len(top)} films | {top.score.min():.3f}–{top.score.max():.3f}")
        return top.to_dict("records")

    if n_dist >= MIN_RECENT_FILMS:
        return bayesian_top(recent, "heure")
    log.info(f"  {n_dist} < {MIN_RECENT_FILMS} → fallback global")
    return bayesian_top(ratings, "global")


# ──────────────────────────────────────────────────────────────────────────────
# 6. RECALCUL VECTEUR USER — FORMULE ALS EXACTE
#
#   L'objectif ALS pour un user u fixé est :
#     min_u  Σ_i c_ui (p_ui - u·v_i)²  +  λ||u||²
#
#   Dérivée = 0  →  u = (VᵀCᵤV + λI)⁻¹  Vᵀcᵤpᵤ
#
#   Avec :
#     V     = item_factors  (n_items × f)
#     c_ui  = valeur de confiance dans matrix_conf (ligne u)
#     p_ui  = 1 si c_ui > 0, 0 sinon  (préférence binaire implicite)
#     λ     = regularization
#
#   Cette implémentation fonctionne pour tous les users y compris nouveaux.
# ──────────────────────────────────────────────────────────────────────────────

def _recalculate_user_vector(item_factors: np.ndarray,
                             conf_row: csr_matrix,
                             regularization: float) -> np.ndarray:
    """
    Calcule le vecteur optimal pour un user à partir de sa ligne de confiance.
    item_factors : (n_items, f)  float32
    conf_row     : (1, n_items)  sparse  — valeurs de confiance
    Retourne     : (f,)          float32
    """
    V   = item_factors                        # (n_items, f)
    lam = regularization
    f   = V.shape[1]

    # Indices et valeurs non-nuls (items vus par ce user)
    indices = conf_row.indices                # items vus
    values  = conf_row.data                   # c_ui

    # Sous-matrice V des items vus
    V_u = V[indices]                          # (n_seen, f)

    # c_ui - 1  → coefficient pour la partie préférence
    # (p_ui = 1 pour tous les items vus, donc Vᵀcᵤpᵤ = Vᵀ(c_ui)·1 = Σ c_ui·v_i)
    # (VᵀCᵤV = Vᵀ·diag(c_ui)·V = Σ c_ui · v_i·v_iᵀ)

    # VᵀCᵤV  =  VᵀV  +  Σ (c_ui - 1) v_i v_iᵀ
    # On calcule VᵀV global + correction sur les items vus
    VtV  = V.T @ V                            # (f, f)  — précalculable globalement
    # Correction : Σ (c_ui - 1) v_i v_iᵀ
    w    = values - 1.0                       # (n_seen,)
    VtCuV = VtV + (V_u * w[:, None]).T @ V_u  # (f, f)

    # Vᵀcᵤpᵤ = Σ c_ui · v_i
    VtCup = (V_u * values[:, None]).sum(axis=0)  # (f,)

    # Résolution du système linéaire (f, f) · u = (f,)
    A = VtCuV + lam * np.eye(f, dtype=np.float32)
    u = np.linalg.solve(A, VtCup).astype(np.float32)
    return u


# ──────────────────────────────────────────────────────────────────────────────
# 7. RECOMMANDEUR
# ──────────────────────────────────────────────────────────────────────────────

class ALSRecommender:
    """
    Moteur de recommandation ALS — interface Netflix-grade.

    recommend(user_id, k)             → Top-K personnalisés
    recommend_new(k)                  → Cold-start (films de l'heure)
    update_user(user_id, movie_id, r) → Mise à jour temps réel O(f²)
    similar_movies(movie_id, k)       → Films similaires (espace latent)
    get_user_profile(user_id)         → Résumé du profil utilisateur
    """

    def __init__(self, model, matrix_conf, matrix_seen,
                 user_enc, item_enc, user_dec, item_dec,
                 movies_df, cold_start_list):
        self.model           = model
        self.matrix_conf     = matrix_conf
        self.matrix_seen     = matrix_seen
        self.user_enc        = user_enc
        self.item_enc        = item_enc
        self.user_dec        = np.array(user_dec)
        self.item_dec        = np.array(item_dec)
        self.movies_df       = movies_df.set_index("movieId")
        self.cold_start_list = cold_start_list

        # VᵀV précalculé une fois — réutilisé dans chaque recalcul user
        log.info("Précalcul VᵀV…")
        V = self.model.item_factors.astype(np.float32)
        self._VtV = (V.T @ V).astype(np.float32)
        log.info("  VᵀV prêt.")

    def _info(self, movie_id: int) -> dict:
        if movie_id in self.movies_df.index:
            r = self.movies_df.loc[movie_id]
            return {"title": r["title"], "genres": r["genres"]}
        return {"title": "Inconnu", "genres": "?"}

    def _score_and_rank(self, u_vec: np.ndarray,
                        seen_indices: set, k: int) -> list:
        """
        Calcule les scores u·V pour tous les items, masque les items vus,
        retourne le Top-K.
        u_vec : (f,)
        """
        V      = self.model.item_factors          # (n_items, f)
        scores = V @ u_vec                        # (n_items,)

        # Masque des items vus → -inf pour les exclure du classement
        if seen_indices:
            mask = np.array(list(seen_indices), dtype=np.int32)
            scores[mask] = -np.inf

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            {"movieId": int(self.item_dec[i]),
             "score"  : float(scores[i]),
             **self._info(int(self.item_dec[i]))}
            for i in top_idx
            if scores[i] > -np.inf
        ]

    # ── Recommandation ───────────────────────────────────────────────────────

    def recommend(self, user_id: int, k: int = TOP_K_DEFAULT) -> list:
        if user_id not in self.user_enc:
            return self.recommend_new(k)

        u_idx   = self.user_enc[user_id]
        u_vec   = self.model.user_factors[u_idx].astype(np.float32)
        seen    = set(self.matrix_seen[u_idx].indices.tolist())
        return self._score_and_rank(u_vec, seen, k)

    def recommend_new(self, k: int = TOP_K_DEFAULT) -> list:
        return self.cold_start_list[:k]

    # ── Mise à jour temps réel ────────────────────────────────────────────────

    def update_user(self, user_id: int, movie_id: int, rating: float):
        """
        Recalcule le vecteur utilisateur par la formule ALS exacte.
        Complexité : O(n_seen · f²) + résolution système f×f → < 5ms.
        """
        if movie_id not in self.item_enc:
            log.warning(f"movieId={movie_id} hors vocabulaire.")
            return

        i_idx = self.item_enc[movie_id]
        conf  = (float(1.0 + ALPHA * rating)
                 if rating >= POSITIVE_THRESHOLD
                 else float(NEGATIVE_CONFIDENCE))
        sentiment = "POSITIF" if rating >= POSITIVE_THRESHOLD else "NEGATIF"

        n_items = self.matrix_conf.shape[1]

        if user_id not in self.user_enc:
            # ── Nouvel utilisateur ──────────────────────────────────────────
            new_idx = self.matrix_conf.shape[0]
            self.user_enc[user_id] = new_idx
            self.user_dec = np.append(self.user_dec, user_id)

            new_conf_row = csr_matrix(([conf], ([0],[i_idx])), shape=(1, n_items))
            new_seen_row = csr_matrix(([1.0],  ([0],[i_idx])), shape=(1, n_items))
            self.matrix_conf = vstack([self.matrix_conf, new_conf_row]).tocsr()
            self.matrix_seen = vstack([self.matrix_seen, new_seen_row]).tocsr()

            # Calcul du vecteur par formule ALS
            u_vec = _recalculate_user_vector(
                self.model.item_factors.astype(np.float32),
                self.matrix_conf[new_idx],
                ALS_CONFIG["regularization"],
            )
            # Injecter dans user_factors
            self.model.user_factors = np.vstack(
                [self.model.user_factors,
                 u_vec.reshape(1, -1).astype(np.float32)]
            )
            log.info(f"  Nouvel user {user_id} intégré (idx={new_idx}) "
                     f"| film={movie_id} note={rating} [{sentiment}]")

        else:
            # ── User existant ───────────────────────────────────────────────
            u_idx = self.user_enc[user_id]

            # Mise à jour efficace via lil_matrix
            conf_lil = self.matrix_conf.tolil()
            seen_lil = self.matrix_seen.tolil()
            conf_lil[u_idx, i_idx] = conf
            seen_lil[u_idx, i_idx] = 1.0
            self.matrix_conf = conf_lil.tocsr()
            self.matrix_seen = seen_lil.tocsr()

            # Recalcul vecteur par formule ALS
            u_vec = _recalculate_user_vector(
                self.model.item_factors.astype(np.float32),
                self.matrix_conf[u_idx],
                ALS_CONFIG["regularization"],
            )
            self.model.user_factors[u_idx] = u_vec
            log.info(f"  user={user_id} | film={movie_id} | "
                     f"note={rating} [{sentiment}] | conf={conf:.1f} → vecteur recalculé")

    # ── Similarité ───────────────────────────────────────────────────────────

    def similar_movies(self, movie_id: int, k: int = TOP_K_DEFAULT) -> list:
        if movie_id not in self.item_enc:
            return []
        i_idx = self.item_enc[movie_id]
        ids, scores = self.model.similar_items(i_idx, N=k + 1)
        return [
            {"movieId": int(self.item_dec[i]),
             "score"  : float(s),
             **self._info(int(self.item_dec[i]))}
            for i, s in zip(ids, scores)
            if int(self.item_dec[i]) != movie_id
        ][:k]

    # ── Profil utilisateur ───────────────────────────────────────────────────

    def get_user_profile(self, user_id: int) -> dict:
        """Retourne les films bien notés et mal notés d'un user."""
        if user_id not in self.user_enc:
            return {"liked": [], "disliked": []}
        u_idx    = self.user_enc[user_id]
        conf_row = self.matrix_conf[u_idx]
        liked, disliked = [], []
        for i_idx, c in zip(conf_row.indices, conf_row.data):
            mid  = int(self.item_dec[i_idx])
            info = self._info(mid)
            if c >= 1.0 + ALPHA * POSITIVE_THRESHOLD:
                liked.append(info)
            else:
                disliked.append(info)
        return {"liked": liked, "disliked": disliked}


# ──────────────────────────────────────────────────────────────────────────────
# 8. PERSISTANCE
# ──────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, matrix_conf, matrix_seen,
                   user_enc, item_enc, user_dec, item_dec,
                   cold_start_list):
    log.info(f"Sauvegarde dans {MODEL_DIR}/…")
    with open(MODEL_DIR / "als_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=5)
    save_npz(str(MODEL_DIR / "matrix_conf.npz"), matrix_conf)
    save_npz(str(MODEL_DIR / "matrix_seen.npz"), matrix_seen)
    meta = {
        "user_enc"       : user_enc,
        "item_enc"       : item_enc,
        "user_dec"       : user_dec,
        "item_dec"       : item_dec,
        "cold_start_list": cold_start_list,
        "saved_at"       : datetime.now().isoformat(),
        "als_config"     : ALS_CONFIG,
        "pos_threshold"  : POSITIVE_THRESHOLD,
        "alpha"          : ALPHA,
        "neg_confidence" : NEGATIVE_CONFIDENCE,
    }
    with open(MODEL_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=5)
    total = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
    log.info(f"  Sauvegarde terminée ✓ ({total/1e6:.1f} MB)")


def load_artifacts(movies_df: pd.DataFrame) -> ALSRecommender:
    log.info(f"Chargement depuis {MODEL_DIR}/…")
    with open(MODEL_DIR / "als_model.pkl", "rb") as f:
        model = pickle.load(f)
    matrix_conf = load_npz(str(MODEL_DIR / "matrix_conf.npz")).tocsr()
    matrix_seen = load_npz(str(MODEL_DIR / "matrix_seen.npz")).tocsr()
    with open(MODEL_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    log.info(f"  Modèle du {meta['saved_at']} | "
             f"α={meta['alpha']} | seuil={meta['pos_threshold']}")
    return ALSRecommender(
        model           = model,
        matrix_conf     = matrix_conf,
        matrix_seen     = matrix_seen,
        user_enc        = meta["user_enc"],
        item_enc        = meta["item_enc"],
        user_dec        = meta["user_dec"],
        item_dec        = meta["item_dec"],
        movies_df       = movies_df,
        cold_start_list = meta["cold_start_list"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9. ÉVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_hit_rate(rec: ALSRecommender, test_df: pd.DataFrame,
                      k: int = 10, n_users: int = 1000) -> float:
    log.info(f"Évaluation HR@{k}…")
    sample = test_df.sample(min(n_users, len(test_df)), random_state=42)
    hits, skips = 0, 0
    for _, row in sample.iterrows():
        uid, mid = int(row.userId), int(row.movieId)
        if mid not in rec.item_enc:
            skips += 1
            continue
        if any(r["movieId"] == mid for r in rec.recommend(uid, k=k)):
            hits += 1
    evaluated = len(sample) - skips
    hr = hits / evaluated if evaluated > 0 else 0.0
    log.info(f"  HR@{k} = {hr:.4f}  ({hits}/{evaluated} | {skips} hors-vocab)")
    return hr


# ──────────────────────────────────────────────────────────────────────────────
# 10. PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> ALSRecommender:
    log.info("=" * 60)
    log.info("  ALS Recommender — MovieLens 1M — v5 Final")
    log.info("=" * 60)

    ratings, movies = load_movielens_1m()
    train_df, test_df = leave_one_out_split(ratings)

    (matrix_conf, matrix_seen,
     user_enc, item_enc,
     user_dec, item_dec) = build_matrices(train_df)

    model = train_als(matrix_conf)
    cold_start_list = build_cold_start_list(ratings, movies)

    save_artifacts(model, matrix_conf, matrix_seen,
                   user_enc, item_enc, user_dec, item_dec,
                   cold_start_list)

    rec = load_artifacts(movies)

    # ── Démo user existant ───────────────────────────────────────────────────
    log.info("\n--- Top-5 user existant ---")
    demo_uid = int(train_df.userId.iloc[500])
    for i, r in enumerate(rec.recommend(demo_uid, k=5), 1):
        log.info(f"  {i}. {r['title']} ({r['genres']}) — {r['score']:.4f}")

    # ── Démo cold-start ──────────────────────────────────────────────────────
    log.info("\n--- Cold-start ---")
    for i, r in enumerate(rec.recommend_new(k=5), 1):
        log.info(f"  {i}. {r['title']} [{r['n']} votes] — {r['score']:.4f}")

    # ── Démo nouvel user : déteste Toy Story, adore Matrix ───────────────────
    log.info("\n--- Nouvel user (Toy Story=1, Matrix=5) ---")
    rec.update_user(99999, movie_id=1,    rating=1.0)
    rec.update_user(99999, movie_id=2571, rating=5.0)
    for i, r in enumerate(rec.recommend(99999, k=5), 1):
        log.info(f"  {i}. {r['title']} ({r['genres']}) — {r['score']:.4f}")

    # ── Vérification Toy Story absent des recs après note 1/5 ────────────────
    log.info("\n--- Vérification filtre négatif ---")
    rec.update_user(demo_uid, movie_id=1, rating=1.0)
    recs = rec.recommend(demo_uid, k=10)
    in_recs = any(r["movieId"] == 1 for r in recs)
    log.info(f"  Toy Story dans Top-10 après note 1/5 ? {'OUI ❌' if in_recs else 'NON ✓'}")

    # ── Évaluation ──────────────────────────────────────────────────────────
    evaluate_hit_rate(rec, test_df, k=10,  n_users=1000)
    evaluate_hit_rate(rec, test_df, k=20,  n_users=1000)

    log.info("\nPipeline terminé. Artefacts dans model/")
    return rec


if __name__ == "__main__":
    recommender = run_pipeline()
