#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coleta séries anuais iNat/GBIF para Mamíferos no BR ou Sudeste.

Uso:
  BR (created, research-grade & verifiable=true por padrão):
    python inat_gbif_growth_mammals_region.py --region br

  Sudeste, observed, sem verificação (tudo):
    python inat_gbif_growth_mammals_region.py --region sudeste --date-field observed --verifiable false --quality-grade ""
"""
import argparse
import math
from datetime import datetime, UTC
from pathlib import Path
import time
import os
from tqdm import tqdm

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import SSLError, ConnectionError, ReadTimeout


# -------------------------------
# Constantes e IDs
# -------------------------------
GBIF_INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"  # iNat Research-grade no GBIF
GBIF_TAXONKEY_MAMMALIA = 359
INAT_TAXON_ID_MAMMALIA = 40151

PLACE_NAMES = {
    "br": ["Brazil"],
    "sudeste": ["Espírito Santo, BR", "Minas Gerais, BR", "Rio de Janeiro, BR", "São Paulo, BR"],
}

# GADM IDs por estado (p/ GBIF regional do Sudeste)
SUDESTE_GADM = ["BR.ES_1_0", "BR.MG_1_0", "BR.RJ_1_0", "BR.SP_1_0"]

# -------------------------------
# Sessão HTTP com retry/backoff
# -------------------------------
DEFAULT_SLEEP_BASE = 0.35
_SLEEP_BASE = DEFAULT_SLEEP_BASE

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": f"saguis-dinov3/1.0 (+github.com/costaleirbag; contact: {os.getenv('USER','user')}@local)"
})
_RETRY = Retry(
    total=5,
    connect=3,
    read=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",),
    raise_on_status=False,
)
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))
_SESSION.mount("http://", HTTPAdapter(max_retries=_RETRY))

def _request_json(url: str, params: dict, max_attempts: int = 6) -> dict:
    """GET robusto: respeita Retry-After p/ 429 e aplica backoff exponencial."""
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = _SESSION.get(url, params=params, timeout=60)
        except (SSLError, ConnectionError, ReadTimeout) as e:
            last = e
            time.sleep(_SLEEP_BASE * (2 ** (attempt - 1)))
            continue

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                time.sleep(int(ra))
            else:
                time.sleep(_SLEEP_BASE * (2 ** (attempt - 1)))
            continue

        r.raise_for_status()
        return r.json()

    if last:
        raise last  # erro de rede persistente
    # Última tentativa, se ainda não retornou
    r.raise_for_status()


# Filtros padrão do iNat (podem ser sobrescritos por args)
INAT_DEFAULT_FILTERS = {
    "verifiable": "true",        # "true" = apenas verificáveis; "false" = todos
    "quality_grade": "research"  # remova/"" para incluir todas as qualidades
}

# -------------------------------
# Helpers
# -------------------------------
def _now_stamp():
    return datetime.now(UTC).strftime("%Y%m%d")

def _ensure_year_bounds(dfs):
    year_min = min(df["year"].min() for df in dfs if not df.empty)
    year_max = max(df["year"].max() for df in dfs if not df.empty)
    return int(year_min), int(year_max)

def _sanitize_inat_filters(verifiable: str | None, quality_grade: str | None) -> dict:
    """
    Regras:
    - Se quality_grade foi explicitamente definido (research/needs_id/casual), NÃO enviar 'verifiable'.
    - Se quiser 'tudo' (incluindo casual), não enviar NENHUM dos dois (quality_grade vazio e verifiable=false).
    - Se só quiser 'verificáveis', enviar APENAS verifiable=true.
    """
    f = {}
    qg = (quality_grade or "").strip()

    if qg:  # usuário fixou uma quality específica
        f["quality_grade"] = qg      # não envia verifiable junto
        return f

    if verifiable == "true":
        f["verifiable"] = "true"     # apenas verificáveis
        return f

    # verifiable == "false"  -> queremos tudo: não enviar verifiable nem quality_grade
    return f  # vazio

def _date_params(date_field: str, start_year: int, end_year: int) -> dict:
    """
    Para 'created', usar created_d1/created_d2.
    Para 'observed', usar d1/d2.
    NÃO enviar 'date_field' junto (pode causar 422).
    """
    if date_field == "created":
        return {
            "created_d1": f"{start_year}-01-01",
            "created_d2": f"{end_year}-12-31",
        }
    else:  # observed
        return {
            "d1": f"{start_year}-01-01",
            "d2": f"{end_year}-12-31",
        }

def _inat_places_autocomplete(q, per_page=20):
    """Chama /v1/places/autocomplete com ordenação por área e retorna a lista bruta."""
    url = "https://api.inaturalist.org/v1/places/autocomplete"
    params = {"q": q, "per_page": per_page, "order_by": "area"}
    payload = _request_json(url, params)
    return payload.get("results", [])

def _resolve_place_id(place_query: str, require_admin_level=None, require_place_type=None):
    """
    Retorna o primeiro place.id que bate pelo nome e (opcionalmente)
    por admin_level/place_type. Se nada bater, levanta ValueError.
    """
    candidates = _inat_places_autocomplete(place_query)
    best = None
    for p in candidates:
        if require_admin_level is not None and p.get("admin_level") != require_admin_level:
            continue
        if require_place_type is not None and p.get("place_type") != require_place_type:
            continue
        best = p
        break
    if not best and candidates:
        best = candidates[0]
    if not best:
        raise ValueError(f"Não encontrei place_id para '{place_query}'")
    return str(best["id"])

# cache simples em memória na execução
_PLACE_CACHE = {}

def resolve_region_place_ids(region: str) -> list[str]:
    """
    Converte PLACE_NAMES[region] -> lista de place_id válidos no iNat.
    Força admin_level=1 (estados) para o Sudeste; para 'br', pega o país.
    """
    names = PLACE_NAMES[region]
    place_ids = []
    for name in names:
        key = (name, region)
        if key in _PLACE_CACHE:
            place_ids.append(_PLACE_CACHE[key])
            continue
        if region == "br":
            pid = _resolve_place_id(name, require_admin_level=0)  # país costuma ser admin_level=0
        else:
            pid = _resolve_place_id(name, require_admin_level=1)  # estados = 1
        _PLACE_CACHE[key] = pid
        place_ids.append(pid)
        time.sleep(0.2)  # educação com a API
    return place_ids

# -------------------------------
# GBIF (resiliente)
# -------------------------------
def gbif_counts_by_year_mammals_br():
    """Contagem anual de ocorrências GBIF (iNat + Mammalia) para o Brasil inteiro."""
    url = "https://api.gbif.org/v1/occurrence/counts/year"
    params = {
        "datasetKey": GBIF_INAT_DATASET,
        "country": "BR",
        "taxonKey": GBIF_TAXONKEY_MAMMALIA,
    }
    payload = _request_json(url, params)
    # Pode vir dict {year: count} OU {"results":[...]}
    if isinstance(payload, dict) and "results" in payload:
        rows = payload["results"]
    elif isinstance(payload, dict):
        rows = [{"year": int(k), "count": v} for k, v in payload.items()]
    else:
        rows = payload
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["year", "gbif_records_mammals_br"])
    df = df.rename(columns={"count": "gbif_records_mammals_br"})
    df["year"] = df["year"].astype(int)
    return df.sort_values("year").reset_index(drop=True)

def gbif_counts_by_year_mammals_sudeste(start_year=2008, end_year=None):
    """Contagem anual GBIF para o Sudeste somando por estados via gadmGid, com backoff."""
    if end_year is None:
        end_year = datetime.now(UTC).year
    rows = []
    for y in range(start_year, end_year + 1):
        total = 0
        for gid in SUDESTE_GADM:
            params = {
                "datasetKey": GBIF_INAT_DATASET,
                "taxonKey": GBIF_TAXONKEY_MAMMALIA,
                "gadmGid": gid,
                "year": y,
                "limit": 0,  # apenas total
            }
            # Retry manual extra para oscilações de rede/SSL
            attempts = 0
            while True:
                attempts += 1
                try:
                    payload = _request_json("https://api.gbif.org/v1/occurrence/search", params)
                    total += int(payload.get("count", 0))
                    break
                except (SSLError, ConnectionError, ReadTimeout):
                    time.sleep(_SLEEP_BASE * (2 ** min(attempts, 4)))
                    if attempts >= 6:
                        raise
            time.sleep(_SLEEP_BASE * 0.5)  # pausa entre estados
        rows.append({"year": y, "gbif_records_mammals_sudeste": total})
        time.sleep(_SLEEP_BASE * 0.5)  # pausa entre anos
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

# -------------------------------
# iNaturalist: Observações
# -------------------------------
def inat_histogram_year_for_place(place_id: str, start_year: int, end_year: int,
                                  date_field: str, extra_filters: dict):
    """Retorna dict ano->contagem para um único place_id via /observations/histogram."""
    url = "https://api.inaturalist.org/v1/observations/histogram"
    params = {
        "interval": "year",
        "taxon_id": INAT_TAXON_ID_MAMMALIA,
        "place_id": place_id,
        **_date_params(date_field, start_year, end_year),
    }
    if extra_filters:
        params.update({k: v for k, v in extra_filters.items() if v not in (None, "", [])})
    payload = _request_json(url, params)
    buckets = payload.get("results", {}).get("year", {})
    year_counts = {}
    for k, v in buckets.items():
        # chaves vêm como 'YYYY-01-01'
        y = int(str(k)[:4])
        year_counts[y] = year_counts.get(y, 0) + int(v)
    return year_counts

def inat_observations_by_year(region: str, start_year=2008, end_year=None,
                              date_field="created", extra_filters=None):
    """Observações anuais (soma por estados se sudeste)."""
    if end_year is None:
        end_year = datetime.now(UTC).year
    place_ids = resolve_region_place_ids(region)
    merged = {}
    # barra de progresso para os estados/regiões
    for pid in tqdm(place_ids, desc=f"Observações {region}", unit="estado"):
        hc = inat_histogram_year_for_place(pid, start_year, end_year, date_field, extra_filters or {})
        for y, c in hc.items():
            merged[y] = merged.get(y, 0) + c
    if not merged:
        return pd.DataFrame(columns=["year", f"inat_obs_mammals_{region}"])
    years = sorted(merged.keys())
    counts = [merged[y] for y in years]
    return pd.DataFrame({"year": years, f"inat_obs_mammals_{region}": counts}).reset_index(drop=True)

# -------------------------------
# iNaturalist: Usuários distintos
# -------------------------------
def _inat_fetch_observer_ids_for_year(place_id: str, date_args: dict, base_filters: dict) -> set:
    """Baixa TODOS os observadores (user.id) p/ (place_id, ano) com paginação."""
    url = "https://api.inaturalist.org/v1/observations/observers"
    params_base = {
        "taxon_id": INAT_TAXON_ID_MAMMALIA,
        "place_id": place_id,
        "per_page": 200,
        "page": 1,
        **date_args,  # created_d1/d2 OU d1/d2
    }
    if base_filters:
        params_base.update({k: v for k, v in base_filters.items() if v is not None and v != ""})

    payload = _request_json(url, params_base)
    total = int(payload.get("total_results", 0))
    if total == 0:
        return set()

    pages = math.ceil(total / 200)
    user_ids = set()

    def add_users(res_json):
        for item in res_json.get("results", []):
            u = item.get("user") or {}
            if "id" in u and u["id"] is not None:
                user_ids.add(int(u["id"]))

    add_users(payload)
    for p in range(2, pages + 1):
        time.sleep(_SLEEP_BASE)  # educação mínima a cada página
        page_payload = _request_json(url, {**params_base, "page": p})
        add_users(page_payload)
    return user_ids


def inat_distinct_observers_by_year(region: str, start_year=2008, end_year=None,
                                    date_field="created", extra_filters=None):
    """Usuários distintos/ano:
       - BR: usa total_results direto (1 chamada)
       - Sudeste: união de user.id nos 4 estados (paginação)
    """
    if end_year is None:
        end_year = datetime.now(UTC).year
    rows = []
    place_ids = resolve_region_place_ids(region)

    # barra de progresso para os anos
    for y in tqdm(range(start_year, end_year + 1), desc=f"Usuários {region}", unit="ano"):
        date_args = _date_params(date_field, y, y)
        if region == "br":
            url = "https://api.inaturalist.org/v1/observations/observers"
            params = {
                "taxon_id": INAT_TAXON_ID_MAMMALIA,
                "place_id": place_ids[0],
                "per_page": 1,
                **date_args,
            }
            if extra_filters:
                params.update({k: v for k, v in extra_filters.items() if v is not None and v != ""})
            payload = _request_json(url, params)
            total = int(payload.get("total_results", 0))
            rows.append({"year": y, f"inat_users_mammals_{region}": total})
        else:
            union_ids = set()
            # barra de progresso interna para os estados de cada ano
            for pid in tqdm(place_ids, leave=False, desc=f"Estados {region}", unit="estado"):
                time.sleep(_SLEEP_BASE)
                union_ids |= _inat_fetch_observer_ids_for_year(pid, date_args, extra_filters or {})
            rows.append({"year": y, f"inat_users_mammals_{region}": len(union_ids)})

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Séries anuais iNat/GBIF para Mamíferos no BR ou Sudeste.")
    parser.add_argument("--region", choices=["br", "sudeste"], default="br",
                        help="Região alvo (br | sudeste).")
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--date-field", choices=["created", "observed"], default="created",
                        help="Campo temporal no iNat: created (quando cadastrado) | observed (quando observado).")
    # filtros iNat
    parser.add_argument("--verifiable", choices=["true", "false"], default=INAT_DEFAULT_FILTERS.get("verifiable", "true"))
    parser.add_argument("--quality-grade", default=INAT_DEFAULT_FILTERS.get("quality_grade", "research"),
                        help="Ex.: research | needs_id | casual (remova/\"\" para incluir todas)")
    parser.add_argument("--sleep-base", type=float, default=DEFAULT_SLEEP_BASE,
                        help="Pausa base (segundos) entre chamadas/páginas p/ evitar 429. Ex.: 0.5, 1.0")
    parser.add_argument("--skip-users", action="store_true",
                        help="Pula o cálculo de usuários distintos (rápido para testes).")
    parser.add_argument("--skip-gbif", action="store_true",
                        help="Pula consultas ao GBIF (útil para teste rápido só com dados do iNat).")

    args = parser.parse_args()

    global _SLEEP_BASE
    _SLEEP_BASE = max(0.0, float(args.sleep_base))

    # monta filtros coerentes p/ iNat
    inat_filters = _sanitize_inat_filters(args.verifiable, args.quality_grade)

    region = args.region
    suffix = region

    # iNat
    df_inat_obs = inat_observations_by_year(
        region, start_year=args.start_year, end_year=args.end_year,
        date_field=args.date_field, extra_filters=inat_filters
    )
    if args.skip_users:
        df_inat_users = pd.DataFrame(columns=["year", f"inat_users_mammals_{region}"])
    else:
        df_inat_users = inat_distinct_observers_by_year(
            region, start_year=args.start_year, end_year=args.end_year,
            date_field=args.date_field, extra_filters=inat_filters
        )

    # GBIF
    if args.skip_gbif:
        # note o nome da coluna vazio coerente com a região
        gbif_col = "gbif_records_mammals_br" if region == "br" else "gbif_records_mammals_sudeste"
        df_gbif = pd.DataFrame(columns=["year", gbif_col])
    elif region == "br":
        df_gbif = gbif_counts_by_year_mammals_br()
    else:
        df_gbif = gbif_counts_by_year_mammals_sudeste(args.start_year, args.end_year)

    # agrega
    dfs = [df for df in [df_gbif, df_inat_obs, df_inat_users] if not df.empty]
    if not dfs:
        print("Nenhum dado retornado. Verifique filtros.")
        return

    y0, y1 = _ensure_year_bounds(dfs)
    out = pd.DataFrame({"year": list(range(y0, y1 + 1))})
    for d in dfs:
        out = out.merge(d, on="year", how="left")

    # salvar
    save_dir = Path("data/inaturalist/metrics")
    save_dir.mkdir(parents=True, exist_ok=True)
    base = f"inat_gbif_mammals_{suffix}_{_now_stamp()}"
    out.to_csv(save_dir / f"{base}.csv", index=False)
    out.to_parquet(save_dir / f"{base}.parquet", index=False)

    print("Salvo em:", save_dir.resolve())
    print(out.tail(10))
    print("\nColunas:", list(out.columns))

if __name__ == "__main__":
    main()
