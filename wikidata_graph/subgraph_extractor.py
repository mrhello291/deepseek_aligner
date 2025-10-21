# wikidata_graph/subgraph_extractor.py
import requests
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import networkx as nx
import time

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# A typed predicate map example
TYPED_PREDICATES = {
    "person": ["wdt:P69", "wdt:P108", "wdt:P166", "wdt:P39", "wdt:P69"],
    "place": ["wdt:P31", "wdt:P17", "wdt:P276", "wdt:P625"]
}

def sparql_query(q: str) -> dict:
    headers = {"Accept": "application/sparql-results+json", "User-Agent": "GraphRAG/1.0 (email@example.com)"}
    r = requests.get(SPARQL_ENDPOINT, params={"query": q}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def get_entity_qid(name: str) -> str:
    q = f'''
    SELECT ?item WHERE {{
      ?item rdfs:label "{name}"@en.
    }} LIMIT 1
    '''
    res = sparql_query(q)
    bindings = res.get("results", {}).get("bindings", [])
    if not bindings:
        return None
    uri = bindings[0]['item']['value']
    return uri.split("/")[-1]

def fetch_typed_neighbors(qid: str, predicates: List[str]) -> List[Tuple[str,str,str]]:
    """
    Returns list of triples (subject_qid, predicate, object_qid_or_literal)
    """
    preds = " ".join(predicates)
    # Build SPARQL that fetches predicate triples limited to listed predicates
    pred_filter = "||".join([f"?p = {p}" for p in predicates])
    q = f'''
    SELECT ?p ?o WHERE {{
      wd:{qid} ?p ?o .
      FILTER ( {" || ".join([f"?p = {p}" for p in predicates])} )
    }} LIMIT 500
    '''
    # Simpler, explicit expansion per predicate:
    triples = []
    for p in predicates:
        q = f"SELECT ?o WHERE {{ wd:{qid} {p} ?o . }} LIMIT 200"
        try:
            res = sparql_query(q)
            for b in res.get("results", {}).get("bindings", []):
                o = b['o']['value']
                triples.append((qid, p, o))
        except Exception as e:
            print("SPARQL error", e)
    return triples

def build_filtered_subgraph(name: str, entity_type: str = "person", second_degree_limit=50):
    qid = get_entity_qid(name)
    if qid is None:
        raise ValueError("Entity not found on Wikidata: " + name)
    G = nx.DiGraph()
    pred_list = TYPED_PREDICATES.get(entity_type, [])
    first_triples = fetch_typed_neighbors(qid, pred_list)
    for s,p,o in first_triples:
        G.add_edge(s, o, predicate=p)
    # second-degree: neighbors of neighbors, limited & centrality filtered
    candidates = set([t[2].split("/")[-1] if t[2].startswith("http") else t[2] for t in first_triples])
    # naive second-degree fetch (limit)
    for c in list(candidates)[:second_degree_limit]:
        try:
            triples = fetch_typed_neighbors(c, pred_list)
            for s,p,o in triples:
                G.add_edge(s, o, predicate=p)
        except Exception:
            continue
    # prune using degree centrality: keep top-k nodes
    centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_nodes = {n for n,_ in sorted_nodes[:100]}
    subG = G.subgraph(top_nodes).copy()
    return qid, subG
