"""
Análise Comparativa de Algoritmos de Machine Learning para Detecção de Fraudes Bancárias

Este script implementa uma análise comparativa de diferentes algoritmos de ML
para detecção de fraudes bancárias usando múltiplos datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Configuração para visualizações
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FraudDetectionAnalyzer:
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.results = {}
        self.scalers = {}
        
    def load_datasets(self):
        """Carrega todos os datasets disponíveis com amostras otimizadas"""
        print("Carregando datasets...")
        
        # Dataset 1: Credit Card Fraud Detection (amostra estratificada)
        try:
            df1_full = pd.read_csv('data/credit card fraud detection/creditcard.csv')
            # Amostra estratificada para manter proporção de fraudes
            fraud_samples = df1_full[df1_full['Class'] == 1]
            normal_samples = df1_full[df1_full['Class'] == 0].sample(n=min(50000, len(df1_full[df1_full['Class'] == 0])), random_state=42)
            df1 = pd.concat([fraud_samples, normal_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
            df1['dataset'] = 'Credit Card'
            df1['target'] = df1['Class']
            self.datasets['credit_card'] = df1
            print(f"✓ Credit Card dataset: {df1.shape[0]} amostras, {df1.shape[1]} features")
        except Exception as e:
            print(f"✗ Erro ao carregar Credit Card dataset: {e}")
        
        # Dataset 2: Bank Account Fraud (amostra)
        try:
            df2 = pd.read_csv('data/Bank Account Fraud Dataset Suite/Base.csv', nrows=50000)
            df2['dataset'] = 'Bank Account'
            df2['target'] = df2['fraud_bool']
            self.datasets['bank_account'] = df2
            print(f"✓ Bank Account dataset: {df2.shape[0]} amostras, {df2.shape[1]} features")
        except Exception as e:
            print(f"✗ Erro ao carregar Bank Account dataset: {e}")
        
        # Dataset 3: Fraud Transactions (amostra menor)
        try:
            df3 = pd.read_csv('data/Credit Card Transactions Fraud Detection Dataset/fraudTrain.csv', nrows=20000)
            df3['dataset'] = 'Fraud Transactions'
            df3['target'] = df3['is_fraud']
            self.datasets['fraud_transactions'] = df3
            print(f"✓ Fraud Transactions dataset: {df3.shape[0]} amostras, {df3.shape[1]} features")
        except Exception as e:
            print(f"✗ Erro ao carregar Fraud Transactions dataset: {e}")
    
    def preprocess_data(self, df, dataset_name):
        """Preprocessa os dados para análise"""
        print(f"\nPreprocessando {dataset_name}...")
        
        # Remover colunas desnecessárias
        cols_to_drop = ['dataset', 'target']
        if 'Unnamed: 0' in df.columns:
            cols_to_drop.append('Unnamed: 0')
        
        # Para dataset de transações, remover colunas de texto
        if dataset_name == 'fraud_transactions':
            text_cols = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 
                        'street', 'city', 'state', 'job', 'dob', 'trans_num']
            cols_to_drop.extend([col for col in text_cols if col in df.columns])
        
        df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Separar features e target
        if 'target' in df.columns:
            y = df['target']
        else:
            y = df.iloc[:, -1]  # Última coluna como target
        
        X = df_clean.drop(columns=['target'] if 'target' in df_clean.columns else [df_clean.columns[-1]])
        
        # Lidar com valores categóricos
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Lidar com valores faltantes
        X = X.fillna(X.mean())
        
        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[dataset_name] = scaler
        
        print(f"  - Features: {X_scaled.shape[1]}")
        print(f"  - Amostras: {X_scaled.shape[0]}")
        print(f"  - Fraudes: {y.sum()} ({y.mean()*100:.2f}%)")
        
        return X_scaled, y
    
    def initialize_models(self):
        """Inicializa os modelos de ML para comparação (otimizados para velocidade)"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=50),
            'SVM': SVC(random_state=42, probability=True, kernel='linear'),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Naive Bayes': GaussianNB()
        }
        print(f"Inicializados {len(self.models)} modelos de ML (otimizados)")
    
    def train_and_evaluate_models(self, X, y, dataset_name):
        """Treina e avalia todos os modelos"""
        print(f"\nTreinando modelos para {dataset_name}...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        dataset_results = {}
        
        for name, model in self.models.items():
            print(f"  - Treinando {name}...")
            
            try:
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Predições
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Métricas
                auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                
                # Cross-validation (reduzido para velocidade)
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
                
                dataset_results[name] = {
                    'model': model,
                    'auc': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"    AUC: {auc_score:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"    Erro: {e}")
                dataset_results[name] = None
        
        self.results[dataset_name] = dataset_results
        return X_test, y_test
    
    def create_comparison_plots(self):
        """Cria gráficos comparativos dos resultados"""
        print("\nCriando visualizações comparativas...")
        
        # Preparar dados para comparação
        comparison_data = []
        for dataset_name, results in self.results.items():
            for model_name, result in results.items():
                if result is not None:
                    comparison_data.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'AUC': result['auc'],
                        'CV_Mean': result['cv_mean'],
                        'CV_Std': result['cv_std']
                    })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Gráfico 1: Comparação de AUC por modelo
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(data=df_comparison, x='Model', y='AUC', hue='Dataset')
        plt.title('Comparação de AUC por Modelo e Dataset')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Gráfico 2: Heatmap de performance
        plt.subplot(2, 2, 2)
        pivot_auc = df_comparison.pivot(index='Model', columns='Dataset', values='AUC')
        sns.heatmap(pivot_auc, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Heatmap de AUC Score')
        
        # Gráfico 3: Comparação de CV scores
        plt.subplot(2, 2, 3)
        sns.barplot(data=df_comparison, x='Model', y='CV_Mean', hue='Dataset')
        plt.title('Comparação de Cross-Validation Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Gráfico 4: Boxplot de distribuição de scores
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df_comparison, x='Model', y='AUC')
        plt.title('Distribuição de AUC Scores por Modelo')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('graficos/comparacao_algoritmos_ml.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_comparison
    
    def generate_detailed_report(self, df_comparison):
        """Gera relatório detalhado dos resultados"""
        print("\nGerando relatório detalhado...")
        
        # Melhor modelo por dataset
        best_models = df_comparison.loc[df_comparison.groupby('Dataset')['AUC'].idxmax()]
        
        # Estatísticas gerais
        overall_stats = df_comparison.groupby('Model').agg({
            'AUC': ['mean', 'std', 'min', 'max'],
            'CV_Mean': ['mean', 'std']
        }).round(4)
        
        print("\n" + "="*80)
        print("RELATÓRIO DE ANÁLISE COMPARATIVA DE ALGORITMOS DE ML")
        print("="*80)
        
        print("\n1. MELHORES MODELOS POR DATASET:")
        print("-" * 50)
        for _, row in best_models.iterrows():
            print(f"{row['Dataset']}: {row['Model']} (AUC: {row['AUC']:.4f})")
        
        print("\n2. ESTATÍSTICAS GERAIS POR MODELO:")
        print("-" * 50)
        print(overall_stats)
        
        print("\n3. RANKING GERAL DE MODELOS:")
        print("-" * 50)
        model_ranking = df_comparison.groupby('Model')['AUC'].mean().sort_values(ascending=False)
        for i, (model, score) in enumerate(model_ranking.items(), 1):
            print(f"{i}. {model}: {score:.4f}")
        
        # Salvar relatório
        with open('relatorio_comparacao_ml_fraude.md', 'w', encoding='utf-8') as f:
            f.write("# Relatório de Análise Comparativa de Algoritmos de ML para Detecção de Fraudes\n\n")
            f.write("## Resumo Executivo\n\n")
            f.write("Este relatório apresenta uma análise comparativa de diferentes algoritmos de Machine Learning ")
            f.write("aplicados à detecção de fraudes bancárias utilizando múltiplos datasets.\n\n")
            
            f.write("## Metodologia\n\n")
            f.write("- **Datasets utilizados**: Credit Card, Bank Account, Fraud Transactions\n")
            f.write("- **Algoritmos testados**: 7 algoritmos diferentes\n")
            f.write("- **Métricas de avaliação**: AUC, Cross-Validation\n")
            f.write("- **Preprocessamento**: Normalização, tratamento de valores faltantes\n\n")
            
            f.write("## Resultados Principais\n\n")
            f.write("### Melhores Modelos por Dataset\n")
            for _, row in best_models.iterrows():
                f.write(f"- **{row['Dataset']}**: {row['Model']} (AUC: {row['AUC']:.4f})\n")
            
            f.write("\n### Ranking Geral\n")
            for i, (model, score) in enumerate(model_ranking.items(), 1):
                f.write(f"{i}. **{model}**: {score:.4f}\n")
            
            f.write("\n## Conclusões\n\n")
            f.write("A análise revela que diferentes algoritmos apresentam performance variada dependendo do tipo de dataset. ")
            f.write("Recomenda-se a utilização de ensemble methods para melhorar a robustez do sistema de detecção de fraudes.\n")
        
        print("\nRelatório salvo em: relatorio_comparacao_ml_fraude.md")
    
    def run_complete_analysis(self):
        """Executa a análise completa"""
        print("INICIANDO ANÁLISE COMPARATIVA DE ALGORITMOS DE ML PARA DETECÇÃO DE FRAUDES")
        print("="*80)
        
        # 1. Carregar datasets
        self.load_datasets()
        
        # 2. Inicializar modelos
        self.initialize_models()
        
        # 3. Processar cada dataset
        for dataset_name, df in self.datasets.items():
            X, y = self.preprocess_data(df, dataset_name)
            self.train_and_evaluate_models(X, y, dataset_name)
        
        # 4. Criar visualizações
        df_comparison = self.create_comparison_plots()
        
        # 5. Gerar relatório
        self.generate_detailed_report(df_comparison)
        
        print("\n" + "="*80)
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("="*80)

if __name__ == "__main__":
    # Criar diretório para gráficos
    import os
    os.makedirs('graficos', exist_ok=True)
    
    # Executar análise
    analyzer = FraudDetectionAnalyzer()
    analyzer.run_complete_analysis()
