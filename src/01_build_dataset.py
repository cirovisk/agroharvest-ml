import os
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# Configurações de caminhos dinâmicos para portabilidade host/container
ML_DIR = Path(__file__).resolve().parent.parent
AGRO_DIR = Path("/home/demonduck/Projects/Pessoais/portfolio/cultivares-duckdb")
DB_PATH = AGRO_DIR / "data" / "storage" / "cultivares.duckdb"
COORDS_PATH = AGRO_DIR / "data" / "open_meteo" / "municipios_coords.csv"

OUTPUT_DATA_DIR = ML_DIR / "outputs" / "data"
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula a distância em km entre dois pontos geográficos."""
    r = 6371.0 # Raio da Terra em km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c

def main():
    print("=" * 60)
    print("ETAPA 1: Construção do Dataset Analítico de Machine Learning")
    print("=" * 60)

    # Mudar o diretório de trabalho para o Agroharvest-BRv2 para que as views do DuckDB encontrem os Parquets relativos
    os.chdir(str(AGRO_DIR))

    # 1. Conexão com o DuckDB do Lakehouse
    print(f"Conectando ao banco DuckDB: {DB_PATH}")
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # 2. Carregar dados das Dimensões e Fatos do Lakehouse
    print("Carregando tabelas do DuckDB...")
    df_pam = conn.execute("""
        SELECT 
            id_municipio, 
            id_cultura, 
            CAST(ano AS INTEGER) as ano, 
            area_plantada_ha, 
            area_colhida_ha, 
            qtde_produzida_ton,
            valor_producao_mil_reais
        FROM fato_producao_pam
        WHERE id_cultura IN (1, 2) -- 1: Soja, 2: Milho
    """).fetchdf()

    df_mun = conn.execute("""
        SELECT id_municipio, codigo_ibge, nome, uf 
        FROM dim_municipio
    """).fetchdf()

    df_cult = conn.execute("""
        SELECT id_cultura, nome_padronizado as cultura
        FROM dim_cultura
    """).fetchdf()

    df_meteo = conn.execute("""
        SELECT 
            id_municipio,
            precipitacao_total_mm,
            temp_max_c,
            temp_min_c,
            temp_media_c,
            umidade_media
        FROM fato_meteorologia
    """).fetchdf()

    df_zarc = conn.execute("""
        SELECT 
            id_municipio, 
            id_cultura,
            AVG(risco_climatico) as risco_zarc_medio
        FROM fato_risco_zarc
        GROUP BY id_municipio, id_cultura
    """).fetchdf()

    df_ndvi = conn.execute("""
        SELECT id_municipio, CAST(ano AS INTEGER) as ano, ndvi_max_safra, ndvi_mean_safra
        FROM fato_ndvi_satelite
    """).fetchdf()

    df_fert_mun = conn.execute("""
        SELECT id_municipio, COUNT(*) as qtd_fertilizantes_mun
        FROM fato_fertilizantes_estabelecimentos
        WHERE id_municipio IS NOT NULL
        GROUP BY id_municipio
    """).fetchdf()

    df_fert_uf = conn.execute("""
        SELECT uf, COUNT(*) as qtd_fertilizantes_uf
        FROM fato_fertilizantes_estabelecimentos
        GROUP BY uf
    """).fetchdf()

    df_preco_mun = conn.execute("""
        SELECT id_cultura, id_municipio, AVG(valor_kg) as preco_medio_mun
        FROM fato_precos_conab_mensal
        WHERE id_municipio IS NOT NULL
        GROUP BY id_cultura, id_municipio
    """).fetchdf()

    df_preco_uf = conn.execute("""
        SELECT id_cultura, uf, AVG(valor_kg) as preco_medio_uf
        FROM fato_precos_conab_mensal
        GROUP BY id_cultura, uf
    """).fetchdf()

    df_preco_nat = conn.execute("""
        SELECT id_cultura, AVG(valor_kg) as preco_medio_nat
        FROM fato_precos_conab_mensal
        GROUP BY id_cultura
    """).fetchdf()

    conn.close()

    print(f"  PAM: {len(df_pam)} registros")
    print(f"  Municípios: {len(df_mun)} registros")
    print(f"  Culturas: {len(df_cult)} registros")
    print(f"  Clima: {len(df_meteo)} registros diários")
    print(f"  ZARC: {len(df_zarc)} registros agregados")

    # 3. Carregar coordenadas globais de todos os municípios do Brasil
    print(f"Carregando coordenadas globais de {COORDS_PATH}...")
    df_coords = pd.read_csv(COORDS_PATH)
    # Garantir que codigo_ibge seja string de 7 dígitos para join
    df_coords["codigo_ibge"] = df_coords["codigo_ibge"].astype(str)
    df_mun["codigo_ibge"] = df_mun["codigo_ibge"].astype(str)

    # Merge das coordenadas no df_mun
    df_mun_coords = pd.merge(
        df_mun, 
        df_coords[["codigo_ibge", "latitude", "longitude"]], 
        on="codigo_ibge", 
        how="left"
    )

    # Identificar quais municípios têm dados de meteorologia (estações de clima)
    estacoes_ids = df_meteo["id_municipio"].unique()
    df_estacoes = df_mun_coords[df_mun_coords["id_municipio"].isin(estacoes_ids)].copy()
    
    # Remover qualquer estação sem coordenadas válidas
    df_estacoes = df_estacoes.dropna(subset=["latitude", "longitude"])
    
    print(f"Estações climáticas disponíveis: {len(df_estacoes)} municípios com coordenadas")

    # 4. Agregar meteorologia diária em médias regionais
    print("Agregando meteorologia por estação...")
    df_meteo_agregado = df_meteo.groupby("id_municipio").agg(
        precipitacao_anual_mm=("precipitacao_total_mm", lambda x: x.sum() / 2.0), # Dividido por 2 porque há 2 anos de histórico
        temp_max_media=("temp_max_c", "mean"),
        temp_min_media=("temp_min_c", "mean"),
        temp_media_anual=("temp_media_c", "mean"),
        umidade_media_anual=("umidade_media", "mean")
    ).reset_index()

    df_estacoes_clima = pd.merge(df_estacoes, df_meteo_agregado, on="id_municipio", how="inner")

    # 5. Mapear cada município produtor do PAM para a estação meteorológica mais próxima
    print("Mapeando a distância geográfica para associar clima a todos os municípios...")
    
    # Listas para o cálculo de distância
    coords_estacoes = df_estacoes_clima[["id_municipio", "latitude", "longitude"]].values
    
    municipios_mapeados = []
    
    for idx, row in df_mun_coords.iterrows():
        m_id = row["id_municipio"]
        lat = row["latitude"]
        lon = row["longitude"]
        
        if pd.isna(lat) or pd.isna(lon):
            # Fallback se não achar coordenadas: usa a primeira estação como padrão (ou nulo)
            municipios_mapeados.append((m_id, None, np.nan))
            continue
            
        # Calcular distâncias para todas as estações
        min_dist = float("inf")
        closest_estacao_id = None
        
        for est_id, est_lat, est_lon in coords_estacoes:
            dist = haversine_distance(lat, lon, est_lat, est_lon)
            if dist < min_dist:
                min_dist = dist
                closest_estacao_id = est_id
                
        municipios_mapeados.append((m_id, closest_estacao_id, min_dist))

    df_proximidade = pd.DataFrame(municipios_mapeados, columns=["id_municipio", "id_estacao_proxima", "distancia_estacao_km"])

    # 6. Unificar tudo no dataset de ML
    print("Unificando tabelas de produção, cultura, coordenadas, clima e ZARC...")
    df_dataset = pd.merge(df_pam, df_mun_coords, on="id_municipio", how="inner")
    df_dataset = pd.merge(df_dataset, df_cult, on="id_cultura", how="inner")
    
    # Join com proximidade climática
    df_dataset = pd.merge(df_dataset, df_proximidade, on="id_municipio", how="inner")
    
    # Join com o clima da estação mais próxima
    df_dataset = pd.merge(
        df_dataset, 
        df_estacoes_clima[["id_municipio", "precipitacao_anual_mm", "temp_max_media", "temp_min_media", "temp_media_anual", "umidade_media_anual"]], 
        left_on="id_estacao_proxima", 
        right_on="id_municipio", 
        how="left",
        suffixes=("", "_estacao")
    )
    
    # Remover coluna extra id_municipio do join do clima
    if "id_municipio_estacao" in df_dataset.columns:
        df_dataset = df_dataset.drop(columns=["id_municipio_estacao"])

    # Join com ZARC
    df_dataset = pd.merge(df_dataset, df_zarc, on=["id_municipio", "id_cultura"], how="left")

    # Join com NDVI de satélite
    df_dataset = pd.merge(df_dataset, df_ndvi, on=["id_municipio", "ano"], how="left")
    ndvi_max_mean = df_dataset["ndvi_max_safra"].mean()
    ndvi_mean_mean = df_dataset["ndvi_mean_safra"].mean()
    df_dataset["ndvi_max_safra"] = df_dataset["ndvi_max_safra"].fillna(ndvi_max_mean)
    df_dataset["ndvi_mean_safra"] = df_dataset["ndvi_mean_safra"].fillna(ndvi_mean_mean)

    # Join com Contagem de Fertilizantes
    df_dataset = pd.merge(df_dataset, df_fert_mun, on="id_municipio", how="left")
    df_dataset["qtd_fertilizantes_mun"] = df_dataset["qtd_fertilizantes_mun"].fillna(0)
    df_dataset = pd.merge(df_dataset, df_fert_uf, on="uf", how="left")
    df_dataset["qtd_fertilizantes_uf"] = df_dataset["qtd_fertilizantes_uf"].fillna(0)

    # Join com Preços CONAB
    df_dataset = pd.merge(df_dataset, df_preco_mun, on=["id_cultura", "id_municipio"], how="left")
    df_dataset = pd.merge(df_dataset, df_preco_uf, on=["id_cultura", "uf"], how="left")
    df_dataset = pd.merge(df_dataset, df_preco_nat, on="id_cultura", how="left")
    
    df_dataset["preco_medio_ano_anterior"] = df_dataset["preco_medio_mun"] \
        .fillna(df_dataset["preco_medio_uf"]) \
        .fillna(df_dataset["preco_medio_nat"])
        
    df_dataset = df_dataset.drop(columns=["preco_medio_mun", "preco_medio_uf", "preco_medio_nat"])

    # 7. Engenharia de Features e Limpeza
    print("Realizando engenharia de features...")
    
    # Produtividade em kg/ha
    # qtde_produzida_ton * 1000 / area_colhida_ha
    # Tratar divisões por zero ou nulos
    df_dataset = df_dataset[df_dataset["area_colhida_ha"] > 0].copy()
    df_dataset["produtividade_kg_ha"] = (df_dataset["qtde_produzida_ton"] * 1000.0) / df_dataset["area_colhida_ha"]
    
    # Feature região
    def get_regiao(uf):
        if uf in ["PR", "SC", "RS"]: return "Sul"
        elif uf in ["SP", "RJ", "MG", "ES"]: return "Sudeste"
        elif uf in ["MS", "MT", "GO", "DF"]: return "Centro-Oeste"
        elif uf in ["BA", "SE", "AL", "PE", "PB", "RN", "CE", "PI", "MA"]: return "Nordeste"
        else: return "Norte"
        
    df_dataset["regiao"] = df_dataset["uf"].apply(get_regiao)

    # Feature Amplitude térmica
    df_dataset["amplitude_termica_media"] = df_dataset["temp_max_media"] - df_dataset["temp_min_media"]

    # Imputar ZARC nulo com a média da cultura
    zarc_soja_mean = df_dataset[df_dataset["id_cultura"] == 1]["risco_zarc_medio"].mean()
    zarc_milho_mean = df_dataset[df_dataset["id_cultura"] == 2]["risco_zarc_medio"].mean()
    
    df_dataset.loc[(df_dataset["id_cultura"] == 1) & (df_dataset["risco_zarc_medio"].isna()), "risco_zarc_medio"] = zarc_soja_mean
    df_dataset.loc[(df_dataset["id_cultura"] == 2) & (df_dataset["risco_zarc_medio"].isna()), "risco_zarc_medio"] = zarc_milho_mean

    # Limpar registros com nulos em colunas essenciais
    cols_check = ["produtividade_kg_ha", "precipitacao_anual_mm", "temp_media_anual"]
    df_dataset = df_dataset.dropna(subset=cols_check)

    # 8. Salvar dataset final
    out_file = OUTPUT_DATA_DIR / "dataset_ml.csv"
    df_dataset.to_csv(out_file, index=False)
    print(f"✓ Dataset analítico de ML gerado com sucesso em: {out_file}")
    print(f"Total de registros: {len(df_dataset)}")
    print(f"Colunas disponíveis: {list(df_dataset.columns)}")
    print(df_dataset[["cultura", "regiao", "produtividade_kg_ha"]].head())

if __name__ == "__main__":
    main()
