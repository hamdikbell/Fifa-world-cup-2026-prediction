import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="FIFA World Cup 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── Reset & base ── */
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: #f8f8f6; }

/* ── Hero ── */
.hero {
    background: #ffffff;
    border-bottom: 1px solid #e8e6e0;
    padding: 2.5rem 3rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 1rem;
}
.hero-left {}
.hero-eyebrow {
    font-size: 11px;
    letter-spacing: .12em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: .4rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 600;
    color: #111;
    line-height: 1.15;
    margin: 0;
}
.hero-title .accent { color: #1D9E75; }
.hero-sub { font-size: .9rem; color: #666; margin-top: .35rem; }
.hero-stats {
    display: flex;
    gap: 2rem;
}
.hero-stat-label { font-size: 11px; color: #999; margin-bottom: 2px; }
.hero-stat-val { font-size: 1.5rem; font-weight: 600; color: #111; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff;
    border-bottom: 1px solid #e8e6e0;
    padding: 0 2.5rem;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 400;
    color: #888;
    padding: .9rem 1.2rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
    color: #1D9E75 !important;
    border-bottom: 2px solid #1D9E75 !important;
    font-weight: 500;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 1.5rem 2.5rem;
}

/* ── Group cards ── */
.group-card {
    background: #fff;
    border: 1px solid #e8e6e0;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.group-header {
    padding: .6rem 1rem;
    background: #f4f4f0;
    display: flex;
    align-items: center;
    gap: .6rem;
    border-bottom: 1px solid #e8e6e0;
}
.group-letter {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1D9E75;
}
.group-teams-list {
    font-size: 12px;
    color: #888;
}

/* ── Standings table ── */
.standings-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.standings-table th {
    font-size: 10px;
    font-weight: 500;
    color: #aaa;
    text-align: center;
    padding: .3rem .5rem;
    letter-spacing: .06em;
}
.standings-table th.th-team { text-align: left; padding-left: 1rem; }
.standings-table td {
    padding: .5rem .5rem;
    text-align: center;
    border-top: 1px solid #f0f0ec;
    color: #555;
}
.standings-table td.td-team { text-align: left; padding-left: 1rem; color: #111; }
.standings-table td.td-pts { font-weight: 600; color: #111; }
.standings-table td.td-pos { padding-left: .75rem; }
.q-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: #E1F5EE;
    color: #0F6E56;
    font-size: 9px;
    font-weight: 700;
}
.gd-pos { color: #1D9E75; font-weight: 500; }
.gd-neg { color: #D85A30; font-weight: 500; }

/* ── Match cards ── */
.match-card {
    background: #fff;
    border: 1px solid #e8e6e0;
    border-radius: 10px;
    padding: .8rem 1.2rem;
    margin-bottom: .6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.match-tag-real {
    font-size: 10px;
    background: #E1F5EE;
    color: #0F6E56;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: 500;
    white-space: nowrap;
}
.match-tag-pred {
    font-size: 10px;
    background: #E6F1FB;
    color: #185FA5;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: 500;
    white-space: nowrap;
}
.match-teams {
    flex: 1;
    display: flex;
    align-items: center;
    gap: .75rem;
    justify-content: center;
}
.match-team-name { font-size: 13px; font-weight: 500; color: #111; }
.match-score {
    background: #f4f4f0;
    border-radius: 6px;
    padding: .3rem .8rem;
    font-size: 15px;
    font-weight: 600;
    color: #111;
    min-width: 56px;
    text-align: center;
}
.match-winner {
    font-size: 11px;
    color: #1D9E75;
    font-weight: 500;
    white-space: nowrap;
    min-width: 80px;
    text-align: right;
}

/* ── Qualified section ── */
.qual-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.qual-table th {
    font-size: 11px;
    color: #aaa;
    text-align: left;
    padding: .4rem .75rem;
    font-weight: 400;
    border-bottom: 1px solid #e8e6e0;
}
.qual-table td {
    padding: .55rem .75rem;
    border-bottom: 1px solid #f0f0ec;
    color: #444;
}
.qual-table td.td-team { font-weight: 500; color: #111; }
.qual-group-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #E1F5EE;
    color: #0F6E56;
    font-size: 11px;
    font-weight: 600;
}
.qual-in { color: #1D9E75; font-size: 11px; font-weight: 500; }
.qual-out { color: #bbb; font-size: 11px; }

/* ── Section title ── */
.section-title {
    font-size: 11px;
    font-weight: 500;
    color: #888;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin: 1.5rem 0 .75rem;
}

/* ── Metric row ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 120px;
    background: #fff;
    border: 1px solid #e8e6e0;
    border-radius: 10px;
    padding: .85rem 1rem;
}
.metric-label { font-size: 11px; color: #999; margin-bottom: 4px; }
.metric-val { font-size: 1.6rem; font-weight: 600; color: #111; line-height: 1; }
.metric-val .metric-accent { color: #1D9E75; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DONNÉES & MODÈLE
# ============================================================
@st.cache_data
def load_and_train():
    from pathlib import Path
    from sklearn.ensemble import GradientBoostingRegressor

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "results.csv"

    # 🔥 SAFE CHECK
    if not DATA_PATH.exists():
        st.error(f"❌ results.csv introuvable ici : {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])

    today = pd.Timestamp('2026-06-14')
    df_played = df[df['home_score'].notna() & (df['date'] <= today)].copy()
    df_wc2026 = df[df['home_score'].isna() & (df['tournament'] == 'FIFA World Cup')].copy()
    df_train  = df_played[df_played['date'] >= '2010-01-01'].copy()

    tournament_weights = {
        'FIFA World Cup': 5, 'UEFA Euro': 4, 'Copa América': 4,
        'Africa Cup of Nations': 4, 'AFC Asian Cup': 4,
        'CONCACAF Gold Cup': 3, 'FIFA World Cup qualification': 3,
        'UEFA Euro qualification': 3, 'Friendly': 1,
    }
    def get_weight(t):
        for k in tournament_weights:
            if k in t: return tournament_weights[k]
        return 2
    df_train['weight'] = df_train['tournament'].apply(get_weight)

    def compute_team_features(team, date, df, n=10):
        mask = (((df['home_team']==team)|(df['away_team']==team)) & (df['date']<date))
        matches = df[mask].sort_values('date', ascending=False).head(n)
        if len(matches)==0:
            return {'form_points':0,'goals_scored_avg':0,'goals_conceded_avg':0,'win_rate':0}
        points, gs_list, gc_list = [], [], []
        for _, row in matches.iterrows():
            gs = row['home_score'] if row['home_team']==team else row['away_score']
            gc = row['away_score'] if row['home_team']==team else row['home_score']
            gs_list.append(gs); gc_list.append(gc)
            points.append(3 if gs>gc else (1 if gs==gc else 0))
        return {
            'form_points'       : sum(points)/len(points),
            'goals_scored_avg'  : sum(gs_list)/len(gs_list),
            'goals_conceded_avg': sum(gc_list)/len(gc_list),
            'win_rate'          : points.count(3)/len(points),
        }

    def compute_h2h(home, away, date, df, n=5):
        mask = ((((df['home_team']==home)&(df['away_team']==away))|
                 ((df['home_team']==away)&(df['away_team']==home))) & (df['date']<date))
        matches = df[mask].sort_values('date', ascending=False).head(n)
        if len(matches)==0:
            return {'h2h_home_wins':0,'h2h_away_wins':0,'h2h_draws':0}
        hw=dr=aw=0
        for _, row in matches.iterrows():
            hs = row['home_score'] if row['home_team']==home else row['away_score']
            as_ = row['away_score'] if row['home_team']==home else row['home_score']
            if hs>as_: hw+=1
            elif hs==as_: dr+=1
            else: aw+=1
        total = len(matches)
        return {'h2h_home_wins':hw/total,'h2h_away_wins':aw/total,'h2h_draws':dr/total}

    FEATURES = ['home_form','home_goals_scored','home_goals_conceded','home_win_rate',
                'away_form','away_goals_scored','away_goals_conceded','away_win_rate',
                'h2h_home_wins','h2h_away_wins','h2h_draws','neutral','weight']
    rows = []
    for _, match in df_train.iterrows():
        hf  = compute_team_features(match['home_team'], match['date'], df_train)
        af  = compute_team_features(match['away_team'], match['date'], df_train)
        h2h = compute_h2h(match['home_team'], match['away_team'], match['date'], df_train)
        rows.append({
            'home_score':match['home_score'], 'away_score':match['away_score'],
            'home_form':hf['form_points'], 'home_goals_scored':hf['goals_scored_avg'],
            'home_goals_conceded':hf['goals_conceded_avg'], 'home_win_rate':hf['win_rate'],
            'away_form':af['form_points'], 'away_goals_scored':af['goals_scored_avg'],
            'away_goals_conceded':af['goals_conceded_avg'], 'away_win_rate':af['win_rate'],
            **h2h, 'neutral':int(match['neutral']), 'weight':match['weight'],
        })
    df_features = pd.DataFrame(rows)
    X = df_features[FEATURES]

    model_home = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model_away = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model_home.fit(X, df_features['home_score'], sample_weight=df_features['weight'])
    model_away.fit(X, df_features['away_score'], sample_weight=df_features['weight'])

    wc_played = df_played[(df_played['tournament']=='FIFA World Cup') &
                          (df_played['date']>='2026-06-11')].copy()
    played_pairs = set(zip(wc_played['home_team'], wc_played['away_team']))

    rows_wc = []
    for _, match in df_wc2026.iterrows():
        home, away, date = match['home_team'], match['away_team'], match['date']
        if (home, away) in played_pairs: continue
        hf  = compute_team_features(home, date, df_played)
        af  = compute_team_features(away, date, df_played)
        h2h = compute_h2h(home, away, date, df_played)
        rows_wc.append({
            'home_team':home, 'away_team':away, 'date':date,
            'home_form':hf['form_points'], 'home_goals_scored':hf['goals_scored_avg'],
            'home_goals_conceded':hf['goals_conceded_avg'], 'home_win_rate':hf['win_rate'],
            'away_form':af['form_points'], 'away_goals_scored':af['goals_scored_avg'],
            'away_goals_conceded':af['goals_conceded_avg'], 'away_win_rate':af['win_rate'],
            **h2h, 'neutral':int(match['neutral']), 'weight':5,
        })
    wc_pred = pd.DataFrame(rows_wc)
    X_wc = wc_pred[FEATURES]
    wc_pred['pred_home_r'] = np.maximum(0, model_home.predict(X_wc)).round().astype(int)
    wc_pred['pred_away_r'] = np.maximum(0, model_away.predict(X_wc)).round().astype(int)
    wc_pred['source'] = 'prédit'

    real = wc_played[['home_team','away_team','home_score','away_score']].copy()
    real.columns = ['home_team','away_team','pred_home_r','pred_away_r']
    real['pred_home_r'] = real['pred_home_r'].astype(int)
    real['pred_away_r'] = real['pred_away_r'].astype(int)
    real['source'] = 'réel'

    all_wc = pd.concat([
        real[['home_team','away_team','pred_home_r','pred_away_r','source']],
        wc_pred[['home_team','away_team','pred_home_r','pred_away_r','source']]
    ], ignore_index=True)

    n_real  = len(real)
    n_pred  = len(wc_pred)
    n_teams = 48

    return all_wc, n_real, n_pred, n_teams


groups_official = {
    'A': ['Mexico', 'South Korea', 'Czech Republic', 'South Africa'],
    'B': ['Canada', 'Qatar', 'Switzerland', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['United States', 'Australia', 'Turkey', 'Paraguay'],
    'E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Iran', 'New Zealand', 'Egypt'],
    'H': ['Spain', 'Saudi Arabia', 'Uruguay', 'Cape Verde'],
    'I': ['France', 'Norway', 'Senegal', 'Iraq'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}


def calculate_standings(teams, matches):
    stats = {t: {'pts':0,'j':0,'g':0,'n':0,'p':0,'gf':0,'ga':0,'gd':0} for t in teams}
    gm = matches[matches['home_team'].isin(teams) & matches['away_team'].isin(teams)]
    for _, row in gm.iterrows():
        h, a = row['home_team'], row['away_team']
        hg, ag = row['pred_home_r'], row['pred_away_r']
        if h not in stats or a not in stats: continue
        stats[h]['gf']+=hg; stats[h]['ga']+=ag; stats[h]['j']+=1
        stats[a]['gf']+=ag; stats[a]['ga']+=hg; stats[a]['j']+=1
        if hg>ag:
            stats[h]['pts']+=3; stats[h]['g']+=1; stats[a]['p']+=1
        elif hg==ag:
            stats[h]['pts']+=1; stats[a]['pts']+=1; stats[h]['n']+=1; stats[a]['n']+=1
        else:
            stats[a]['pts']+=3; stats[a]['g']+=1; stats[h]['p']+=1
    for t in teams:
        stats[t]['gd'] = stats[t]['gf'] - stats[t]['ga']
    ranked = sorted(teams, key=lambda t:(stats[t]['pts'],stats[t]['gd'],stats[t]['gf']), reverse=True)
    return ranked, stats


def gd_display(gd):
    if gd > 0: return f"+{gd}"
    return str(gd)


def result_label(hg, ag, home, away):
    if hg > ag:   return f"✓ {home}"
    elif hg == ag: return "Nul"
    else:          return f"✓ {away}"


# ============================================================
# LOAD
# ============================================================
with st.spinner("Calcul des prédictions…"):
    all_wc, n_real, n_pred, n_teams = load_and_train()

# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="hero">
  <div class="hero-left">
    <h1 class="hero-title">FIFA World Cup <span class="accent">2026</span></h1>
    <div class="hero-sub">Prédictions de la phase de groupes — données depuis 2010</div>
  </div>
  <div class="hero-stats">
    <div>
      <div class="hero-stat-label">Équipes</div>
      <div class="hero-stat-val">{n_teams}</div>
    </div>
    <div>
      <div class="hero-stat-label">Scores réels</div>
      <div class="hero-stat-val" style="color:#1D9E75">{n_real}</div>
    </div>
    <div>
      <div class="hero-stat-label">Prédictions</div>
      <div class="hero-stat-val">{n_pred}</div>
    </div>
    <div>
      <div class="hero-stat-label">Groupes</div>
      <div class="hero-stat-val">12</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["Classements par groupe", "Scores & résultats", "Équipes qualifiées"])


# ─── TAB 1 : Classements ───────────────────────────────────
with tab1:
    cols = st.columns(2, gap="medium")
    for i, (group_name, teams) in enumerate(groups_official.items()):
        ranked, stats = calculate_standings(teams, all_wc)
        with cols[i % 2]:
            rows_html = ""
            for j, team in enumerate(ranked):
                s = stats[team]
                pos_badge = '<span class="q-badge">✓</span>' if j < 2 else f'<span style="color:#bbb;font-size:12px">{j+1}</span>'
                gd_cls = "gd-pos" if s['gd'] > 0 else ("gd-neg" if s['gd'] < 0 else "")
                rows_html += f"""
                <tr>
                  <td class="td-pos">{pos_badge}</td>
                  <td class="td-team">{team}</td>
                  <td>{s['j']}</td><td>{s['g']}</td><td>{s['n']}</td><td>{s['p']}</td>
                  <td>{s['gf']}</td><td>{s['ga']}</td>
                  <td class="{gd_cls}">{gd_display(s['gd'])}</td>
                  <td class="td-pts">{s['pts']}</td>
                </tr>"""

            team_list = " · ".join(teams)
            st.markdown(f"""
            <div class="group-card">
              <div class="group-header">
                <span class="group-letter">Groupe {group_name}</span>
                <span class="group-teams-list">{team_list}</span>
              </div>
              <table class="standings-table">
                <thead>
                  <tr>
                    <th class="td-pos"></th>
                    <th class="th-team">Équipe</th>
                    <th>J</th><th>G</th><th>N</th><th>P</th>
                    <th>GF</th><th>GA</th><th>GD</th><th>Pts</th>
                  </tr>
                </thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;color:#aaa;margin-top:.5rem">✓ = qualifié pour les 8èmes de finale</div>', unsafe_allow_html=True)


# ─── TAB 2 : Scores ────────────────────────────────────────
with tab2:
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        group_sel = st.selectbox(
            "Groupe",
            list(groups_official.keys()),
            format_func=lambda x: f"Groupe {x} — {' · '.join(groups_official[x])}"
        )
    with col_info:
        teams_sel = groups_official[group_sel]
        ranked_sel, stats_sel = calculate_standings(teams_sel, all_wc)
        leader = ranked_sel[0]
        st.markdown(f"""
        <div style="padding:.6rem 0;font-size:13px;color:#555">
          Leader actuel&nbsp; <strong style="color:#1D9E75">{leader}</strong>
          &nbsp;·&nbsp; {stats_sel[leader]['pts']} pts
          &nbsp;·&nbsp; GD {gd_display(stats_sel[leader]['gd'])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)

    gm = all_wc[all_wc['home_team'].isin(teams_sel) & all_wc['away_team'].isin(teams_sel)]
    for _, row in gm.iterrows():
        h, a   = row['home_team'], row['away_team']
        hg, ag = row['pred_home_r'], row['pred_away_r']
        src    = row['source']
        tag_html = '<span class="match-tag-real">Score réel</span>' if src == 'réel' else '<span class="match-tag-pred">Prédit</span>'
        winner = result_label(hg, ag, h, a)
        st.markdown(f"""
        <div class="match-card">
          {tag_html}
          <div class="match-teams">
            <span class="match-team-name">{h}</span>
            <span class="match-score">{hg} – {ag}</span>
            <span class="match-team-name">{a}</span>
          </div>
          <span class="match-winner">{winner}</span>
        </div>
        """, unsafe_allow_html=True)


# ─── TAB 3 : Qualifiés ─────────────────────────────────────
with tab3:
    qualified = []
    third_place = []
    for gname, teams in groups_official.items():
        ranked, stats = calculate_standings(teams, all_wc)
        qualified.append({'Groupe': gname, 'Place': '1er', 'Équipe': ranked[0],
                          'Pts': stats[ranked[0]]['pts'], 'GD': stats[ranked[0]]['gd']})
        qualified.append({'Groupe': gname, 'Place': '2e',  'Équipe': ranked[1],
                          'Pts': stats[ranked[1]]['pts'], 'GD': stats[ranked[1]]['gd']})
        third_place.append({'Groupe': gname, 'Équipe': ranked[2],
                            'Pts': stats[ranked[2]]['pts'], 'GD': stats[ranked[2]]['gd']})

    st.markdown('<div class="section-title">1ers et 2es — qualifiés directement (24 équipes)</div>', unsafe_allow_html=True)

    rows_q = ""
    for q in qualified:
        badge = f'<span class="qual-group-badge">{q["Groupe"]}</span>'
        rows_q += f"""<tr>
          <td>{badge}</td>
          <td style="color:#888;font-size:12px">{q['Place']}</td>
          <td class="td-team">{q['Équipe']}</td>
          <td style="text-align:center">{q['Pts']}</td>
          <td style="text-align:center">{gd_display(q['GD'])}</td>
        </tr>"""

    st.markdown(f"""
    <table class="qual-table">
      <thead><tr><th>Gr.</th><th>Place</th><th>Équipe</th><th style="text-align:center">Pts</th><th style="text-align:center">GD</th></tr></thead>
      <tbody>{rows_q}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:2rem">Meilleurs 3es — 8 qualifiés sur 12</div>', unsafe_allow_html=True)

    third_sorted = sorted(third_place, key=lambda x:(x['Pts'],x['GD']), reverse=True)
    rows_t = ""
    for i, t in enumerate(third_sorted):
        badge  = f'<span class="qual-group-badge">{t["Groupe"]}</span>'
        status = '<span class="qual-in">Qualifié ✓</span>' if i < 8 else '<span class="qual-out">Éliminé</span>'
        rows_t += f"""<tr>
          <td>{badge}</td>
          <td class="td-team">{t['Équipe']}</td>
          <td style="text-align:center">{t['Pts']}</td>
          <td style="text-align:center">{gd_display(t['GD'])}</td>
          <td>{status}</td>
        </tr>"""

    st.markdown(f"""
    <table class="qual-table">
      <thead><tr><th>Gr.</th><th>Équipe</th><th style="text-align:center">Pts</th><th style="text-align:center">GD</th><th>Statut</th></tr></thead>
      <tbody>{rows_t}</tbody>
    </table>
    """, unsafe_allow_html=True)
# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="
    text-align:center;
    margin-top:40px;
    padding:25px;
    color:#666;
    border-top:1px solid #e8e6e0;
">
    <div style="font-size:15px;font-weight:600;color:#111;">
        FIFA World Cup 2026 Prediction System
    </div>
    <div style="font-size:13px;margin-top:6px;">
        Developed by <span style="color:#1D9E75;font-weight:600;">Ikbel Hamdi</span>
    </div>
    <div style="font-size:12px;color:#999;margin-top:2px;">
        Data Science Student
    </div>
</div>
""", unsafe_allow_html=True)