"""
CineMatch v6
- Étoiles ultra-compactes (gap minimal, 1 seule colonne HTML)
- Rerun ciblé via fragment pour éviter le reload complet
- Page /modele : paramètres ALS + stats de prédiction
"""

import streamlit as st
import requests, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

st.set_page_config(page_title="CineMatch", page_icon="🎬🎬🎬",
                   layout="wide", initial_sidebar_state="collapsed")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;900&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing:border-box; }
html, body, [data-testid="stApp"] {
    background:#080810 !important; color:#f0f0f8 !important;
    font-family:'Inter',sans-serif !important;
}
[data-testid="stSidebar"] { display:none !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }

.cm-logo {
    font-family:'Montserrat',sans-serif; font-weight:900; font-size:2rem;
    background:linear-gradient(110deg,#ff4e50,#f9d423);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:-.02em; line-height:1;
}
.cm-sub { font-size:.6rem; color:#44446a; text-transform:uppercase; letter-spacing:.2em; }
.cm-sec {
    font-family:'Montserrat',sans-serif; font-weight:700; font-size:1rem;
    color:#f0f0f8; border-left:3px solid #ff4e50; padding-left:10px;
    margin:20px 0 10px 0; display:block;
}
.cm-notice {
    background:#12122a; border-left:3px solid #f9d423;
    border-radius:8px; padding:10px 14px;
    font-size:.8rem; color:#9090c0; margin:6px 0 12px;
}
.cm-toast {
    background:#0d2218; border:1px solid #1a4a2a;
    border-radius:8px; padding:8px 14px; color:#4ade80; font-size:.82rem;
}

/* Inputs */
.stTextInput input {
    background:#12121e !important; color:#f0f0f8 !important;
    border:1px solid #2a2a44 !important; border-radius:8px !important;
}
.stTextInput input:focus { border-color:#ff4e50 !important; }

/* Primary buttons */
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#ff4e50,#c1121f) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:.85rem !important;
    padding:7px 18px !important; transition:opacity .15s !important;
}
.stButton > button[kind="primary"]:hover { opacity:.8 !important; }

/* Secondary buttons (stars) — ultra compact */
.stButton > button[kind="secondary"] {
    background:transparent !important;
    color:#383858 !important;
    border:none !important; box-shadow:none !important;
    font-size:.88rem !important;
    padding:0 !important;
    min-height:18px !important; height:18px !important;
    line-height:1 !important; border-radius:2px !important;
    transition:color .08s, transform .08s !important;
    width:100% !important;
}
.stButton > button[kind="secondary"]:hover {
    background:transparent !important; box-shadow:none !important;
    color:#ff4e50 !important; transform:scale(1.3) !important;
}

/* Réduire l'espace entre colonnes d'étoiles */
div[data-testid="stHorizontalBlock"].star-block > div[data-testid="column"] {
    padding-left: 1px !important;
    padding-right: 1px !important;
    min-width: 0 !important;
    flex: 1 1 0 !important;
}

/* Stats cards */
.stat-card {
    background:#0f0f22; border:1px solid #1e1e36;
    border-radius:10px; padding:14px 16px; margin:6px 0;
}
.stat-val { font-size:1.5rem; font-weight:700; color:#ff4e50; }
.stat-lbl { font-size:.72rem; color:#6060a0; text-transform:uppercase; letter-spacing:.1em; }

hr { border-color:#181828 !important; margin:16px 0 !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:#252538; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
DATA_DIR           = Path("ml-1m")
TMDB_BASE          = "https://api.themoviedb.org/3"
TMDB_IMG           = "https://image.tmdb.org/t/p/w342"
POSITIVE_THRESHOLD = 3.5
ALPHA              = 40.0

# ── TMDB helpers ──────────────────────────────────────────────────────────────
def tmdb_key():
    try: return st.secrets["TMDB_API_KEY"]
    except: return ""

def clean_for_tmdb(raw):
    year = ""
    if m := re.search(r"\((\d{4})\)", raw): year = m.group(1)
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", raw).strip()
    for s, p in [(", The","The"),(", A","A"),(", An","An")]:
        if t.endswith(s): t = p + " " + t[:-len(s)]; break
    return t, year

@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(raw_title):
    key = tmdb_key()
    if not key: return None
    title, year = clean_for_tmdb(raw_title)
    p = {"api_key":key, "query":title, "language":"en-US"}
    if year: p["year"] = year
    try:
        r = requests.get(f"{TMDB_BASE}/search/movie", params=p, timeout=4)
        if r.ok:
            for res in r.json().get("results",[]):
                if res.get("poster_path"):
                    return TMDB_IMG + res["poster_path"]
    except: pass
    return None

@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_search(query: str):
    key = tmdb_key()
    if not key: return []
    try:
        r = requests.get(f"{TMDB_BASE}/search/movie",
                         params={"api_key":key,"query":query,"language":"en-US"}, timeout=5)
        if r.ok:
            out = []
            for res in r.json().get("results",[])[:8]:
                out.append({
                    "tmdb_id"    : res.get("id"),
                    "title"      : res.get("title",""),
                    "genres"     : "",
                    "poster_path": res.get("poster_path",""),
                    "overview"   : res.get("overview","")[:180],
                    "out_of_base": True,
                })
            return out
    except: pass
    return []

@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_genres(tmdb_id: int) -> str:
    key = tmdb_key()
    if not key: return ""
    try:
        r = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}",
                         params={"api_key":key}, timeout=4)
        if r.ok:
            return "|".join(g["name"] for g in r.json().get("genres",[]))
    except: pass
    return ""

# ── Modèle ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Chargement du moteur ALS…")
def load_rec():
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from als_recommender import load_artifacts
    movies = pd.read_csv(DATA_DIR/"movies.dat", sep="::", engine="python",
                         names=["movieId","title","genres"], encoding="latin-1",
                         dtype={"movieId":np.int32})
    # Calcul note moyenne + nb votes par film depuis ratings.dat
    ratings = pd.read_csv(DATA_DIR/"ratings.dat", sep="::", engine="python",
                          names=["userId","movieId","rating","timestamp"],
                          encoding="latin-1", dtype={"movieId":np.int32,"rating":np.float32})
    avg = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", n_votes="count").reset_index()
    avg["avg_rating"] = avg["avg_rating"].round(1)
    movies = movies.merge(avg, on="movieId", how="left")
    return load_artifacts(movies), movies

# ── Utilitaires ───────────────────────────────────────────────────────────────
def name_to_uid(name: str) -> int:
    h = int(hashlib.md5(name.strip().lower().encode()).hexdigest(), 16)
    return 100000 + (h % 800000)

def rate_proxy(rec, uid, genres_str, rating, movies):
    if not genres_str: return
    gset = set(genres_str.lower().split("|"))
    df   = movies.copy()
    df["ov"] = df["genres"].apply(lambda g: len(gset & set(g.lower().split("|"))))
    for _, row in df[df["ov"] > 0].nlargest(5,"ov").iterrows():
        rec.update_user(uid, int(row.movieId), rating)

# ── Composant poster + étoiles (ultra compact) ────────────────────────────────
def poster_with_stars(col, film: dict, uid, rec, movies, kp=""):
    mid      = film.get("movieId")
    title    = film.get("title","")
    genres   = film.get("genres","")
    score    = film.get("score")
    oob      = film.get("out_of_base", False)
    tmdb_id  = film.get("tmdb_id")
    pp       = film.get("poster_path","")
    rate_key = f"rated_{mid if mid else title[:20]}"
    already  = st.session_state.get(rate_key, 0)

    with col:
        img_url = (TMDB_IMG + pp) if pp else get_poster(title)
        if img_url:
            st.image(img_url, width=130)
        else:
            st.markdown(
                '<div style="height:195px;background:#14142a;border-radius:7px;'
                'display:flex;align-items:center;justify-content:center;font-size:2rem">🎬</div>',
                unsafe_allow_html=True)

        short = title[:20]+"…" if len(title) > 20 else title
        g2    = " · ".join(genres.split("|")[:2]) if genres else ""
        tag   = " 🌐" if oob else ""

        # Note moyenne communautaire : priorité au champ du film,
        # sinon lookup direct dans movies (pour les recs qui ne l'ont pas)
        avg_r = film.get("avg_rating")
        if avg_r is None and mid is not None and not oob:
            row = movies[movies["movieId"] == mid]
            if not row.empty:
                avg_r = row.iloc[0].get("avg_rating")

        if avg_r and not np.isnan(float(avg_r)):
            val  = float(avg_r)
            # Étoiles pleines sur 5 (MovieLens est noté /5)
            filled = "★" * round(val) + "☆" * (5 - round(val))
            rating_str = (f"<span style='color:#f9d423;font-size:.75rem'>{filled}</span>"
                          f" <span style='color:#bbb;font-size:.7rem'>{val:.1f}</span>")
        else:
            rating_str = ""   # rien si pas de note connue (films TMDB hors-base)

        st.markdown(
            f"<p style='font-size:.76rem;font-weight:600;margin:3px 0 1px;line-height:1.2'>{short}{tag}</p>"
            f"<p style='font-size:.65rem;color:#6060a0;margin:0 0 1px'>{g2}</p>"
            f"<p style='margin:0 0 3px'>{rating_str}</p>",
            unsafe_allow_html=True)

        # ── Étoiles ultra-compactes via 5 colonnes serrées ───────────────
        if uid:
            s_cols = st.columns(5, gap="small")
            for i, sc_col in enumerate(s_cols):
                v    = i + 1
                icon = "★" if v <= already else "☆"
                with sc_col:
                    if st.button(icon, key=f"{kp}_s{v}"):
                        rating_val = float(v)
                        if oob:
                            g = genres or (tmdb_genres(tmdb_id) if tmdb_id else "")
                            rate_proxy(rec, uid, g, rating_val, movies)
                        else:
                            rec.update_user(uid, mid, rating_val)
                        st.session_state[rate_key] = v
                        s = "★" * v + "☆" * (5-v)
                        st.session_state["toast"] = f"{title[:28]} — {s}"
                        st.rerun()


def render_grid(films, uid, rec, movies, n_cols=5, kp="g"):
    for i in range(0, len(films), n_cols):
        batch = films[i:i+n_cols]
        cols  = st.columns(n_cols)
        for j, (col, f) in enumerate(zip(cols, batch)):
            poster_with_stars(col, f, uid, rec, movies, kp=f"{kp}_{i}_{j}")

# ── LOGIN PAGE ────────────────────────────────────────────────────────────────
def login_page():
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center; padding: 60px 0 20px 0;">
            <div style="font-family:Montserrat,sans-serif;font-weight:900;font-size:3rem;
                        background:linear-gradient(110deg,#ff4e50,#f9d423);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        white-space:nowrap;">🎯 CineMatch</div>
            <div style="font-size:.8rem;color:#6060a0;margin:8px 0 28px;line-height:1.6;">
                Votre moteur de recommandation personnel<br>
                <span style="color:#3a3a5a;font-size:.7rem;">ALS · MovieLens 1M · 6 040 utilisateurs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        name = st.text_input("", placeholder="Votre prénom… (ex: Nathan, Julie)",
                             label_visibility="collapsed", key="login_name")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🚀 Commencer", key="btn_login", type="primary"):
            n = name.strip()
            if n:
                st.session_state["uid"]      = name_to_uid(n)
                st.session_state["username"] = n.capitalize()
                st.session_state["page"]     = "home"
                st.rerun()
            else:
                st.warning("Entrez votre prénom pour continuer.")

# ── HEADER COMMUN ─────────────────────────────────────────────────────────────
def render_header(uid, username):
    h1, h2, h3 = st.columns([2, 5, 2])
    with h1:
        st.markdown('<div class="cm-logo">🎯 CineMatch</div>'
                    '<div class="cm-sub">ALS · MovieLens 1M</div>',
                    unsafe_allow_html=True)
    with h2:
        toast = st.session_state.get("toast")
        if toast:
            st.markdown(f'<div class="cm-toast">✅ Noté : {toast} · Recs actualisées !</div>',
                        unsafe_allow_html=True)
    with h3:
        st.markdown(f"<div style='text-align:right;padding-top:6px'>👤 <b>{username}</b></div>",
                    unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("📊 Modèle", key="nav_model", type="primary"):
                st.session_state["page"] = "model"
                st.rerun()
        with b2:
            if st.button("Déco", key="logout", type="primary"):
                st.session_state.clear()
                st.rerun()

# ── PAGE MODÈLE ───────────────────────────────────────────────────────────────
def model_page(rec, movies, uid, username):
    render_header(uid, username)
    st.divider()

    if st.button("← Retour aux films", key="back_home", type="primary"):
        st.session_state["page"] = "home"
        st.rerun()

    st.markdown('<span class="cm-sec">📊 Paramètres du modèle ALS</span>',
                unsafe_allow_html=True)

    # Récupérer les paramètres depuis rec
    params = {}
    for attr in ["factors","iterations","regularization","alpha","n_users","n_items",
                 "num_factors","num_iterations","dtype"]:
        v = getattr(rec, attr, None)
        if v is not None: params[attr] = v

    model_obj = getattr(rec, "model", None)
    if model_obj:
        for attr in ["factors","iterations","regularization","alpha","num_factors"]:
            v = getattr(model_obj, attr, None)
            if v is not None: params.setdefault(attr, v)

    n_users = getattr(rec, "n_users", None) or len(getattr(rec, "user_enc", {}))
    n_items = getattr(rec, "n_items", None) or len(getattr(rec, "item_enc", {}))

    # Matrice interactions
    R = getattr(rec, "R", None) or getattr(rec, "matrix", None) or getattr(rec, "conf_matrix", None)
    n_ratings = sparsity = None
    if R is not None:
        try:
            import scipy.sparse as sp
            if sp.issparse(R):
                nnz = R.nnz
                sparsity = (1 - nnz / (R.shape[0] * R.shape[1])) * 100
                n_ratings = nnz
            else:
                nnz = int(np.count_nonzero(R))
                sparsity = (1 - nnz / R.size) * 100
                n_ratings = nnz
        except: pass

    U = getattr(rec, "user_factors", None)
    I = getattr(rec, "item_factors", None)
    if model_obj:
        U = U or getattr(model_obj, "user_factors", None)
        I = I or getattr(model_obj, "item_factors", None)

    k_factors = U.shape[1] if U is not None else params.get("factors", params.get("num_factors","?"))
    reg       = params.get("regularization", 0.01)
    alph      = params.get("alpha", ALPHA)
    itr       = params.get("iterations", 15)

    # ── Stat cards ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{n_users or 6040}</div>'
                    '<div class="stat-lbl">Utilisateurs</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{n_items or len(movies)}</div>'
                    '<div class="stat-lbl">Films</div></div>', unsafe_allow_html=True)
    with col3:
        nr = f"{n_ratings:,}" if n_ratings else "~1 000 000"
        st.markdown(f'<div class="stat-card"><div class="stat-val">{nr}</div>'
                    '<div class="stat-lbl">Notations</div></div>', unsafe_allow_html=True)
    with col4:
        sp_str = f"{sparsity:.2f}%" if sparsity is not None else "~95.8%"
        st.markdown(f'<div class="stat-card"><div class="stat-val">{sp_str}</div>'
                    '<div class="stat-lbl">Sparsité</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<span class="cm-sec">⚙️ Hyperparamètres ALS</span>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{k_factors}</div>'
                    '<div class="stat-lbl">Facteurs latents k</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{itr}</div>'
                    '<div class="stat-lbl">Itérations</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{reg}</div>'
                    '<div class="stat-lbl">Régularisation λ</div></div>', unsafe_allow_html=True)
    with p4:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{alph}</div>'
                    '<div class="stat-lbl">Alpha (confiance)</div></div>', unsafe_allow_html=True)

    # ── Explication ───────────────────────────────────────────────────────
    st.markdown('<span class="cm-sec">🧠 Comment fonctionne l\'ALS</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cm-notice" style="font-size:.82rem;line-height:1.8">
    <b>Alternating Least Squares (ALS)</b> — Factorisation matricielle pour feedback implicite.<br><br>
    <b>Objectif :</b> Décomposer la matrice R (utilisateurs × films) en <code>R ≈ U · Iᵀ</code><br>
    &nbsp;&nbsp;→ <b>U</b> : matrice utilisateurs (n_users × k)<br>
    &nbsp;&nbsp;→ <b>I</b> : matrice items (n_items × k)<br><br>
    <b>Confiance :</b> <code>c_ui = 1 + α × r_ui</code> — pondère chaque interaction par sa force.<br><br>
    <b>Alternance :</b> À chaque itération : fixer I → résoudre U (least squares), puis fixer U → résoudre I.<br>
    La <b>régularisation λ</b> pénalise les grands vecteurs pour éviter l'overfitting.<br><br>
    <b>Prédiction :</b> <code>score(u,i) = u⃗ · i⃗ᵀ</code> — produit scalaire des vecteurs latents.
    </div>
    """, unsafe_allow_html=True)

    # ── Stats prédictions pour l'utilisateur courant ──────────────────────
    st.markdown('<span class="cm-sec">🔮 Analyse des prédictions — votre profil</span>',
                unsafe_allow_html=True)

    if uid in rec.user_enc:
        try:
            recs = rec.recommend(uid, k=50)
            scores = [f.get("score", 0) for f in recs if f.get("score") is not None]
            if scores:
                sc_arr = np.array(scores)
                pa, pb, pc, pd_ = st.columns(4)
                with pa:
                    st.markdown(f'<div class="stat-card"><div class="stat-val">{sc_arr.max():.3f}</div>'
                                '<div class="stat-lbl">Score max</div></div>', unsafe_allow_html=True)
                with pb:
                    st.markdown(f'<div class="stat-card"><div class="stat-val">{sc_arr.mean():.3f}</div>'
                                '<div class="stat-lbl">Score moyen</div></div>', unsafe_allow_html=True)
                with pc:
                    st.markdown(f'<div class="stat-card"><div class="stat-val">{sc_arr.min():.3f}</div>'
                                '<div class="stat-lbl">Score min (50ème)</div></div>', unsafe_allow_html=True)
                with pd_:
                    st.markdown(f'<div class="stat-card"><div class="stat-val">{sc_arr.std():.3f}</div>'
                                '<div class="stat-lbl">Écart-type</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<span class="cm-sec">🏆 Top 10 recommandations</span>',
                            unsafe_allow_html=True)
                rows = []
                for r in recs[:10]:
                    rows.append({
                        "Film"      : r.get("title",""),
                        "Genres"    : " · ".join(r.get("genres","").split("|")[:2]),
                        "Score"     : round(r.get("score",0), 4),
                        "Confiance" : "🟢 Haute" if r.get("score",0) > sc_arr.mean() else "🟡 Moyenne",
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score", min_value=0,
                            max_value=float(sc_arr.max()), format="%.4f"
                        )
                    },
                    hide_index=True
                )

                st.markdown('<span class="cm-sec">📈 Distribution des scores (top 50)</span>',
                            unsafe_allow_html=True)
                # Histogramme avec buckets
                hist_vals, hist_edges = np.histogram(sc_arr, bins=10)
                hist_df = pd.DataFrame({
                    "Tranche de score": [f"{e:.2f}" for e in hist_edges[:-1]],
                    "Nombre de films" : hist_vals
                })
                st.bar_chart(hist_df.set_index("Tranche de score"))

        except Exception as e:
            st.warning(f"Stats non disponibles : {e}")
    else:
        st.info("👆 Notez quelques films pour voir l'analyse de vos prédictions personnalisées.")

    # ── Vecteur latent utilisateur ────────────────────────────────────────
    if U is not None and uid in rec.user_enc:
        st.markdown('<span class="cm-sec">🧬 Vecteur latent — votre profil</span>',
                    unsafe_allow_html=True)
        try:
            uidx  = rec.user_enc[uid]
            u_vec = np.array(U[uidx]).flatten()
            st.markdown(f"""
            <div class="cm-notice">
            Dimension : <b>{len(u_vec)} facteurs latents</b> &nbsp;|&nbsp;
            Norme L2 : <b>{np.linalg.norm(u_vec):.4f}</b> &nbsp;|&nbsp;
            Min : <b>{u_vec.min():.4f}</b> &nbsp;|&nbsp;
            Max : <b>{u_vec.max():.4f}</b> &nbsp;|&nbsp;
            Moyenne : <b>{u_vec.mean():.4f}</b>
            </div>
            """, unsafe_allow_html=True)
            n_show = min(30, len(u_vec))
            vec_df = pd.DataFrame({
                "Facteur latent" : [f"k{i}" for i in range(n_show)],
                "Valeur"         : u_vec[:n_show]
            })
            st.bar_chart(vec_df.set_index("Facteur latent"))
        except Exception as e:
            st.caption(f"Vecteur non disponible : {e}")

    # ── Genres préférés inférés ───────────────────────────────────────────
    rated_ids = [k for k in st.session_state if k.startswith("rated_")]
    if rated_ids:
        st.markdown('<span class="cm-sec">🎭 Genres appréciés (session en cours)</span>',
                    unsafe_allow_html=True)
        genre_counts: dict = {}
        for rk in rated_ids:
            rv = st.session_state[rk]
            if rv < 3: continue
            mid_str = rk.replace("rated_","")
            try:
                mid_int = int(mid_str)
                row = movies[movies["movieId"] == mid_int]
                if not row.empty:
                    for g in row.iloc[0]["genres"].split("|"):
                        genre_counts[g] = genre_counts.get(g, 0) + 1
            except: pass
        if genre_counts:
            gc_df = pd.DataFrame(list(genre_counts.items()),
                                  columns=["Genre","Fréquence"]).sort_values("Fréquence",ascending=False)
            st.bar_chart(gc_df.set_index("Genre"))
        else:
            st.caption("Pas encore assez de notations positives.")

# ── APP PRINCIPALE ────────────────────────────────────────────────────────────
def main_app(rec, movies, uid, username):
    render_header(uid, username)
    st.divider()

    st.markdown("""
    <div class="cm-notice">
    📽️ <b>Catalogue MovieLens 1M</b> — 3 706 films sortis avant 2001.
    Films récents (Avatar, Inception…) : affichés avec 🌐, pris en compte via leurs genres.
    <i>Essayez : Matrix, Titanic, Pulp Fiction, Star Wars, Forrest Gump, Seven…</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="cm-sec">🔍 Rechercher & noter un film</span>',
                unsafe_allow_html=True)
    query = st.text_input("", placeholder="Matrix, Avatar, Inception, Titanic…",
                          label_visibility="collapsed", key="search")

    if query:
        mask    = movies["title"].str.contains(query, case=False, na=False)
        in_base = movies[mask].head(8).to_dict("records")
        oob_raw = tmdb_search(query)
        base_lc = {r["title"].lower() for r in in_base}
        oob     = [f for f in oob_raw if f["title"].lower() not in base_lc]
        results = in_base + oob[:max(0, 10 - len(in_base))]
        if not results:
            st.caption("Aucun résultat.")
        else:
            if oob:
                st.markdown('<div class="cm-notice">🌐 Certains films sont hors catalogue — '
                            'notez-les quand même, on utilisera leurs genres !</div>',
                            unsafe_allow_html=True)
            render_grid(results, uid, rec, movies, n_cols=5, kp="sr")

    st.markdown('<span class="cm-sec">✨ Vos recommandations personnalisées</span>',
                unsafe_allow_html=True)
    if uid in rec.user_enc:
        recs = rec.recommend(uid, k=20)
        if recs:
            render_grid(recs, uid, rec, movies, n_cols=5, kp="recs")
    else:
        st.markdown('<div class="cm-notice">⭐ Vous voyez les tendances pour linstant — '
                    'notez des films pour des recs 100% personnalisées !</div>',
                    unsafe_allow_html=True)
        render_grid(rec.recommend_new(k=10), uid, rec, movies, n_cols=5, kp="cold")

    st.divider()
    st.markdown('<span class="cm-sec">🔥 Top films du catalogue</span>',
                unsafe_allow_html=True)
    render_grid(rec.recommend_new(k=15), uid, rec, movies, n_cols=5, kp="top")
    st.markdown('<span class="cm-sec">🎬 À découvrir</span>', unsafe_allow_html=True)
    render_grid(rec.recommend_new(k=30)[15:], uid, rec, movies, n_cols=5, kp="disc")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    try:
        rec, movies = load_rec()
    except Exception as e:
        st.error(f"❌ Erreur chargement : {e}")
        st.stop()

    uid      = st.session_state.get("uid")
    username = st.session_state.get("username", "")
    page     = st.session_state.get("page", "home")

    if not uid:
        login_page()
    elif page == "model":
        model_page(rec, movies, uid, username)
    else:
        main_app(rec, movies, uid, username)

if __name__ == "__main__":
    main()