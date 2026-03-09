import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import os
import random

class MT5RealisticTickConverter:
    """
    Conversor de ticks com geração realista de BID/ASK para MT5
    Versão otimizada para backtesting mais próximo da conta demo
    """
    
    def __init__(self, db_path, symbol='WINJ26'):
        self.db_path = db_path
        self.symbol = symbol
        self.trading_start = time(9, 0)   # 09:00
        self.trading_end = time(18, 25)    # 18:25
        
        # Parâmetros realistas para Mini Índice
        self.base_spread = 5                # Spread base em pontos
        self.min_spread = 2                  # Spread mínimo
        self.max_spread = 15                 # Spread máximo
        self.tick_size = 5                    # Tamanho do tick (Mini Índice)
        self.slippage_probability = 0.15      # 15% de chance de slippage
        self.max_slippage_ticks = 2           # Máximo slippage em ticks
        
        # Estatísticas do mercado para tornar mais realista
        self.volume_profile = {
            '09:00-10:00': 1.5,   # Abertura: mais volume
            '10:00-12:00': 1.0,   # Normal
            '12:00-13:00': 0.6,   # Almoço: menos volume
            '13:00-17:00': 1.2,   # Tarde: mais volume
            '17:00-18:25': 1.8    # Fechamento: pico de volume
        }
        
    def convert_ticks_to_mt5(self, output_path=None, remove_duplicates=True, 
                              aggregate_volume=True, add_realistic_spread=True):
        """
        Converte ticks do formato SQLite para CSV realista para MT5
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query otimizada
            query = """
            SELECT 
                TIME_MSC,
                LAST,
                BID,
                ASK,
                VOLUME
            FROM TICKS 
            ORDER BY TIME_MSC ASC
            """
            
            print("Carregando dados do SQLite...")
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                print("Nenhum dado encontrado!")
                return None
                
            print(f"Total de ticks carregados: {len(df):,}")
            
            # Converter timestamp para datetime (UTC+3)
            df['datetime'] = pd.to_datetime(df['TIME_MSC'], unit='ms') + pd.Timedelta(hours=3)
            
            # Filtrar horário do pregão
            df['time_only'] = df['datetime'].dt.time
            df = df[(df['time_only'] >= self.trading_start) & 
                    (df['time_only'] <= self.trading_end)]
            
            print(f"Ticks após filtro de pregão: {len(df):,}")
            
            if len(df) == 0:
                print("Nenhum tick no horário do pregão!")
                return None
            
            # Remover ticks duplicados
            if remove_duplicates:
                df = df.drop_duplicates(subset=['datetime', 'LAST'], keep='first')
                print(f"Ticks após remover duplicatas: {len(df):,}")
            
            # Formatar para MT5
            df['<DATE>'] = df['datetime'].dt.strftime('%Y.%m.%d')
            df['<TIME>'] = df['datetime'].dt.strftime('%H:%M:%S')
            df['<CLOSE>'] = df['LAST']
            
            # Tratar volume - agregar por segundo
            if aggregate_volume:
                df = self._aggregate_by_second(df)
            else:
                df['<VOLUME>'] = df['VOLUME'].clip(lower=1)
            
            # Adicionar BID/ASK realistas
            if add_realistic_spread:
                df = self._add_realistic_bid_ask(df)
            else:
                df['<BID>'] = df['<CLOSE>']
                df['<ASK>'] = df['<CLOSE>']
                df['<LAST>'] = df['<CLOSE>']
            
            # Preencher gaps pequenos
            df = self._fill_small_gaps(df)
            
            # Adicionar micro-estruturas de mercado
            df = self._add_market_microstructure(df)
            
            # Salvar CSV no formato MT5 (6 colunas)
            if output_path is None:
                output_path = f"{self.symbol}_MT5_TICKS.csv"
            
            # Formato final para MT5: DATE TIME BID ASK LAST VOLUME
            mt5_df = df[['<DATE>', '<TIME>', '<BID>', '<ASK>', '<LAST>', '<VOLUME>']]
            mt5_df.to_csv(output_path, index=False, sep=' ')
            
            print(f"\n✅ Arquivo salvo: {output_path}")
            print(f"✅ Total de ticks: {len(mt5_df):,}")
            
            # Estatísticas detalhadas
            self._print_detailed_statistics(mt5_df, df)
            
            return mt5_df
            
        except Exception as e:
            print(f"Erro: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if conn:
                conn.close()
    
    def _aggregate_by_second(self, df):
        """
        Agrega ticks por segundo com tratamento inteligente de volume
        """
        df['second_key'] = df['datetime'].dt.strftime('%Y%m%d%H%M%S')
        df['hour'] = df['datetime'].dt.hour
        
        # Ajustar volume baseado no perfil do dia
        df['volume_multiplier'] = 1.0
        df.loc[df['hour'] < 10, 'volume_multiplier'] = self.volume_profile['09:00-10:00']
        df.loc[(df['hour'] >= 10) & (df['hour'] < 12), 'volume_multiplier'] = self.volume_profile['10:00-12:00']
        df.loc[(df['hour'] >= 12) & (df['hour'] < 13), 'volume_multiplier'] = self.volume_profile['12:00-13:00']
        df.loc[(df['hour'] >= 13) & (df['hour'] < 17), 'volume_multiplier'] = self.volume_profile['13:00-17:00']
        df.loc[df['hour'] >= 17, 'volume_multiplier'] = self.volume_profile['17:00-18:25']
        
        # Aplicar multiplicador ao volume
        df['VOLUME'] = df['VOLUME'] * df['volume_multiplier']
        
        # Agregar por segundo
        df_agg = df.groupby('second_key').agg({
            '<DATE>': 'first',
            '<TIME>': 'first',
            '<CLOSE>': 'last',
            'VOLUME': 'sum',
            'datetime': 'last',
            'hour': 'first'
        }).reset_index(drop=True)
        
        # Volume mínimo e máximo realista
        df_agg['<VOLUME>'] = df_agg['VOLUME'].clip(lower=1, upper=50).round().astype(int)
        
        print(f"Ticks após agregação por segundo: {len(df_agg):,}")
        return df_agg
    
    def _add_realistic_bid_ask(self, df):
        """
        Gera BID/ASK realistas com spread variável e micro-estruturas
        """
        print("Gerando BID/ASK realistas...")
        
        # Calcular volatilidade local (desvio padrão móvel)
        df['volatility'] = df['<CLOSE>'].rolling(window=20, min_periods=1).std().fillna(1)
        
        # Normalizar volatilidade
        max_vol = df['volatility'].max()
        if max_vol > 0:
            df['vol_norm'] = df['volatility'] / max_vol
        else:
            df['vol_norm'] = 0.1
        
        # Spread base + componente de volatilidade
        df['spread'] = self.base_spread + (df['vol_norm'] * 10).round()
        df['spread'] = df['spread'].clip(self.min_spread, self.max_spread).astype(int)
        
        # Seed para reprodutibilidade
        random.seed(42)
        
        bids = []
        asks = []
        lasts = []
        
        for idx, row in df.iterrows():
            price = row['<CLOSE>']
            spread = row['spread']
            hour = row['hour'] if 'hour' in row else 12
            
            # Assimetria no spread (mais realista)
            # Em certos horários, o spread tende a ser mais assimétrico
            asymmetry_factor = 0.3 if 12 <= hour <= 13 else 0.1  # Mais assimétrico no almoço
            
            # Calcular BID/ASK com assimetria
            half_spread = spread // 2
            asymmetry = int(random.gauss(0, asymmetry_factor * spread))
            
            bid = price - half_spread + asymmetry
            ask = price + (spread - half_spread) - asymmetry
            
            # Garantir múltiplos do tick size
            bid = round(bid / self.tick_size) * self.tick_size
            ask = round(ask / self.tick_size) * self.tick_size
            
            # LAST pode ser BID ou ASK (alternando)
            if random.random() < 0.6:  # 60% das vezes LAST = BID
                last = bid
            else:
                last = ask
            
            # Slippage ocasional
            if random.random() < self.slippage_probability:
                slippage_ticks = random.randint(-self.max_slippage_ticks, self.max_slippage_ticks)
                last = last + (slippage_ticks * self.tick_size)
            
            bids.append(bid)
            asks.append(ask)
            lasts.append(last)
        
        df['<BID>'] = bids
        df['<ASK>'] = asks
        df['<LAST>'] = lasts
        
        # Estatísticas do spread gerado
        df['calc_spread'] = df['<ASK>'] - df['<BID>']
        print(f"  Spread médio gerado: {df['calc_spread'].mean():.2f}")
        print(f"  Spread min/max: {df['calc_spread'].min():.0f}/{df['calc_spread'].max():.0f}")
        
        return df
    
    def _add_market_microstructure(self, df):
        """
        Adiciona micro-estruturas de mercado para tornar os dados mais realistas
        """
        print("Adicionando micro-estruturas de mercado...")
        
        # Padrões de negociação intraday
        df['hour'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>']).dt.hour
        
        # 1. Padrão de abertura (mais volatilidade)
        opening_mask = df['hour'] == 9
        if opening_mask.any():
            # Aumentar spread na abertura
            df.loc[opening_mask, '<ASK>'] = df.loc[opening_mask, '<ASK>'] + 2
            df.loc[opening_mask, '<BID>'] = df.loc[opening_mask, '<BID>'] - 2
        
        # 2. Padrão de fechamento (aceleração)
        closing_mask = df['hour'] >= 17
        if closing_mask.any():
            # Mais ticks de compra no fechamento
            mask = closing_mask & (np.random.random(len(df)) < 0.3)
            df.loc[mask, '<LAST>'] = df.loc[mask, '<ASK>']
        
        # 3. Pequenas reversões (micro-estrutura)
        for i in range(1, len(df)-1):
            if random.random() < 0.05:  # 5% de chance
                # Pequena reversão de tick
                df.loc[df.index[i], '<LAST>'] = df.loc[df.index[i-1], '<LAST>']
        
        return df
    
    def _fill_small_gaps(self, df, max_gap_seconds=3):
        """
        Preenche gaps pequenos com ticks inteligentes
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
        df['time_diff'] = df['datetime'].diff().dt.total_seconds()
        
        gaps = df[df['time_diff'] > 1].index
        
        if len(gaps) == 0:
            return df
        
        print(f"Gaps encontrados: {len(gaps)}")
        
        new_rows = []
        
        for idx in gaps:
            prev_idx = idx - 1
            gap_size = df.loc[idx, 'time_diff']
            
            # Só preencher gaps pequenos
            if 1 < gap_size <= max_gap_seconds:
                prev_time = df.loc[prev_idx, 'datetime']
                prev_bid = df.loc[prev_idx, '<BID>']
                prev_ask = df.loc[prev_idx, '<ASK>']
                prev_last = df.loc[prev_idx, '<LAST>']
                
                # Adicionar tick intermediário
                mid_time = prev_time + timedelta(seconds=gap_size/2)
                
                # Preço intermediário (tendência)
                next_last = df.loc[idx, '<LAST>']
                mid_last = (prev_last + next_last) / 2
                mid_last = round(mid_last / self.tick_size) * self.tick_size
                
                new_rows.append({
                    '<DATE>': mid_time.strftime('%Y.%m.%d'),
                    '<TIME>': mid_time.strftime('%H:%M:%S'),
                    '<BID>': mid_last - self.base_spread//2,
                    '<ASK>': mid_last + self.base_spread//2,
                    '<LAST>': mid_last,
                    '<VOLUME>': 1,
                    'datetime': mid_time
                })
        
        if new_rows:
            df_fill = pd.DataFrame(new_rows)
            df = pd.concat([df, df_fill], ignore_index=True)
            df = df.sort_values('datetime').reset_index(drop=True)
            print(f"  {len(new_rows)} gaps preenchidos")
        
        # Remover colunas temporárias
        df = df.drop(columns=['datetime', 'time_diff'], errors='ignore')
        
        return df
    
    def _print_detailed_statistics(self, mt5_df, original_df):
        """
        Imprime estatísticas detalhadas para validação
        """
        print("\n" + "="*70)
        print("📊 ESTATÍSTICAS DETALHADAS DO ARQUIVO GERADO")
        print("="*70)
        
        # Estatísticas básicas
        print(f"\n📁 Arquivo: {self.symbol}_MT5_TICKS.csv")
        print(f"📈 Total de ticks: {len(mt5_df):,}")
        print(f"📅 Período: {mt5_df['<DATE>'].iloc[0]} a {mt5_df['<DATE>'].iloc[-1]}")
        
        # Estatísticas de preço
        print(f"\n💰 PREÇOS:")
        print(f"   Mínimo: {mt5_df['<LAST>'].min():.2f}")
        print(f"   Máximo: {mt5_df['<LAST>'].max():.2f}")
        print(f"   Médio: {mt5_df['<LAST>'].mean():.2f}")
        
        # Estatísticas de spread
        mt5_df['spread'] = mt5_df['<ASK>'] - mt5_df['<BID>']
        print(f"\n📏 SPREAD (pontos):")
        print(f"   Mínimo: {mt5_df['spread'].min():.0f}")
        print(f"   Máximo: {mt5_df['spread'].max():.0f}")
        print(f"   Médio: {mt5_df['spread'].mean():.2f}")
        print(f"   Desvio padrão: {mt5_df['spread'].std():.2f}")
        
        # Distribuição do spread
        print(f"\n📊 DISTRIBUIÇÃO DO SPREAD:")
        spread_dist = mt5_df['spread'].value_counts().sort_index()
        for spread, count in spread_dist.head(10).items():
            pct = (count / len(mt5_df)) * 100
            print(f"   {spread:2d} pontos: {count:6,d} ticks ({pct:.1f}%)")
        
        # Estatísticas de volume
        print(f"\n📦 VOLUME:")
        print(f"   Volume total: {mt5_df['<VOLUME>'].sum():,}")
        print(f"   Volume médio por tick: {mt5_df['<VOLUME>'].mean():.2f}")
        print(f"   Volume zero: {(mt5_df['<VOLUME>'] == 0).sum()} ticks")
        
        # Volatilidade implícita
        mt5_df['returns'] = mt5_df['<LAST>'].pct_change() * 100
        print(f"\n📈 VOLATILIDADE:")
        print(f"   Retorno médio: {mt5_df['returns'].mean():.4f}%")
        print(f"   Desvio padrão: {mt5_df['returns'].std():.4f}%")
        
        # Comparação com dados originais
        reduction = (1 - len(mt5_df)/len(original_df)) * 100
        print(f"\n💾 OTIMIZAÇÃO:")
        print(f"   Redução de dados: {reduction:.1f}%")
        print(f"   Original: {len(original_df):,} → Final: {len(mt5_df):,}")
        
        # Amostra dos primeiros ticks
        print("\n🔍 PRIMEIROS 5 TICKS DO ARQUIVO:")
        sample = mt5_df[['<DATE>', '<TIME>', '<BID>', '<ASK>', '<LAST>', '<VOLUME>']].head()
        for idx, row in sample.iterrows():
            print(f"   {row['<DATE>']} {row['<TIME>']}  "
                  f"BID:{row['<BID>']:6.0f}  ASK:{row['<ASK>']:6.0f}  "
                  f"LAST:{row['<LAST>']:6.0f}  VOL:{row['<VOLUME>']:3d}")

def configure_mt5_symbol_instructions():
    """
    Retorna instruções para configurar o símbolo no MT5
    """
    instructions = """
🔧 CONFIGURAÇÕES RECOMENDADAS NO MT5:

1️⃣  CRIAR/EDITAR SÍMBOLO:
   • Nome: (use o mesmo do arquivo)
   • Tipo: Forex (ou CFD)
   • Digitos: 2 (para Mini Índice)
   • Contract size: 1
   • Tick size: 5
   • Tick value: 0.05 (R$0,05 por tick)

2️⃣  CONFIGURAÇÕES DE NEGOCIAÇÃO:
   • Spread: Current (usar do arquivo)
   • Stop level: 0
   • Freeze level: 0
   
3️⃣  HORÁRIO DE NEGOCIAÇÃO:
   • Segunda a Sexta: 09:00 - 18:25
   • Configure na aba "Time"

4️⃣  NO STRATEGY TESTER:
   • Modelo: Todos os ticks
   • Spread: Current
   • Execution: Instant
   • Slippage: 5 pontos (1 tick)
   • Deposit: conforme sua conta

5️⃣  VALIDAÇÃO:
   • Execute uma simulação curta (1 hora)
   • Compare com conta demo no mesmo período
   • Ajuste parâmetros se necessário
"""
    return instructions

def main():
    print("="*80)
    print("🚀 CONVERSOR REALISTA DE TICKS PARA MT5")
    print("   Versão com BID/ASK dinâmicos e micro-estruturas")
    print("="*80)
    
    # Solicitar caminho do banco de dados
    db_path = input("\n📁 Caminho do arquivo SQLite: ").strip('"').strip("'")
    
    if not os.path.exists(db_path):
        print("❌ Arquivo não encontrado!")
        return
    
    # Solicitar símbolo
    symbol = input("🏷️  Símbolo (Enter para WINJ26): ").strip() or "WINJ26"
    
    # Criar conversor
    converter = MT5RealisticTickConverter(db_path, symbol)
    
    print("\n⚙️  OPÇÕES DE CONVERSÃO:")
    print("   1 - Recomendado (todos os ajustes realistas)")
    print("   2 - Customizado")
    
    option = input("\nEscolha (1/2): ").strip()
    
    if option == '1':
        # Opção recomendada
        df = converter.convert_ticks_to_mt5(
            remove_duplicates=True,
            aggregate_volume=True,
            add_realistic_spread=True
        )
    else:
        # Opções customizadas
        remove_dups = input("\nRemover ticks duplicados? (S/N) [S]: ").strip().upper() != 'N'
        agg_volume = input("Agregar volume por segundo? (S/N) [S]: ").strip().upper() != 'N'
        realistic = input("Gerar BID/ASK realistas? (S/N) [S]: ").strip().upper() != 'N'
        
        df = converter.convert_ticks_to_mt5(
            remove_duplicates=remove_dups,
            aggregate_volume=agg_volume,
            add_realistic_spread=realistic
        )
    
    if df is not None:
        print("\n" + "="*80)
        print("✅ CONVERSÃO CONCLUÍDA COM SUCESSO!")
        print("="*80)
        
        print(configure_mt5_symbol_instructions())
        
        print("\n📁 Arquivo gerado:")
        print(f"   {symbol}_MT5_TICKS.csv")
        print(f"   Local: {os.path.abspath(f'{symbol}_MT5_TICKS.csv')}")
        
        print("\n📥 Para importar no MT5:")
        print("   1. Copie o arquivo para a pasta Files do MT5")
        print("   2. No MT5, clique com botão direito no Market Watch")
        print("   3. Selecione 'Símbolos' → 'Custom'")
        print("   4. Crie ou edite seu símbolo")
        print("   5. Na aba 'Ticks', clique em 'Importar Ticks'")
        print("   6. Selecione o arquivo gerado")
        
        print("\n💡 DICA: Compare o BT com a conta demo")
        print("   Ajuste os parâmetros de spread e slippage no EA")

if __name__ == "__main__":
    main()