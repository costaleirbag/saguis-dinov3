# src/app/pipeline/check_cv_split.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from app.utils.io import load_parquet

def run_test():
    """
    Este script testa a divisão da validação cruzada para verificar
    se algum dos 'folds' (subconjuntos de teste) acaba com apenas uma classe.
    """
    print("--- Iniciando Teste de Divisão da Validação Cruzada ---")
    
    # Carrega os mesmos dados que o script de busca de hiperparâmetros
    try:
        df = load_parquet("embeddings.parquet")
    except FileNotFoundError:
        print("\n[ERRO] Arquivo 'embeddings.parquet' não encontrado.")
        print("Por favor, gere o arquivo primeiro com 'make_embeddings.py'.")
        return

    # Cria a variável alvo (target) da mesma forma
    df['target'] = df['species'].apply(lambda x: 1 if x == 'hybrid' else 0)
    y = df['target']
    
    print(f"\nTotal de amostras no dataset: {len(y)}")
    print("Distribuição de classes no dataset completo:")
    print(y.value_counts())
    print("-" * 50)
    
    # Usa EXATAMENTE o mesmo StratifiedKFold que está causando o problema
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"Analisando a distribuição das classes em cada um dos {n_splits} folds de validação...\n")
    
    all_folds_are_ok = True
    for i, (train_idx, val_idx) in enumerate(cv.split(df, y)):
        y_val = y.iloc[val_idx]
        
        print(f"--- Fold #{i+1} ---")
        print(f"Tamanho do fold de validação: {len(y_val)}")
        print("Distribuição de classes neste fold:")
        print(y_val.value_counts())
        
        # O teste principal: verifica se o fold tem menos de 2 classes
        if len(y_val.unique()) < 2:
            print("\n[!!!] PROBLEMA ENCONTRADO NESTE FOLD [!!!]")
            print("Este subconjunto de validação contém apenas uma classe.")
            print("O cálculo do ROC AUC falhará aqui, resultando em 'NaN'.\n")
            all_folds_are_ok = False
        else:
            print("Este fold parece OK (contém ambas as classes).\n")

    print("-" * 50)
    if not all_folds_are_ok:
        print("Diagnóstico: Pelo menos um fold não continha amostras de ambas as classes.")
        print("Isso acontece geralmente com datasets pequenos ou muito desbalanceados.")
        print("Sugestão: Reduza o número de splits (n_splits=3) no script 'hyperparam_search.py'.")
    else:
        print("Diagnóstico: Todos os folds parecem ter ambas as classes.")
        print("O problema pode ser outro, mas a causa mais provável foi testada.")

if __name__ == "__main__":
    run_test()