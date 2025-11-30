import numpy as np
import itertools
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
from datetime import datetime
from scipy.integrate import quad

class PDDE:
    def __init__(self, caso=None):
        if caso is None:
            caso = self.definir_caso_com_produtibilidade_nao_linear()
        self.relatorio_completo = []
        self.executar_pdde(caso)

    def definir_caso(self):
        """Define o caso com 3 UHEs - 2 reservatórios + 1 fio d'água com baixa produtividade"""
        return {
                'DGer': {
                    'CDef': 500.0,
                    'Carga': [100.0, 80.0, 50.0],
                    'Nr_Disc': 5,
                    'Nr_Est': 3,
                    'Nr_Cen': 2
                },
                'UHE': [
                    {
                        'id': 0,
                        'Nome': 'UHE 1 - Reservatório',
                        'Vmax': 100.0,
                        'Vmin': 20.0,
                        'Prod': 0.95,
                        'Engol': 75.0,
                        'Afl': [
                            [23, 16],
                            [19, 14], 
                            [15, 11]
                        ],
                        'Reservatorio': True,
                        'Montante': []
                    },
                    {
                        'id': 1,
                        'Nome': 'UHE 2 - Reservatório', 
                        'Vmax': 100.0,
                        'Vmin': 25.0,
                        'Prod': 0.85,
                        'Engol': 95.0,
                        'Afl': [
                            [18, 12],
                            [16, 10],
                            [13, 8]
                        ],
                        'Reservatorio': True,
                        'Montante': []  
                    },
                    {
                        'id': 2,
                        'Nome': 'UHE 3 - Fio dÁgua',
                        'Vmax': 0.0,
                        'Vmin': 0.0,
                        'Prod': 0.50,
                        'Engol': 30.0,
                        'Afl': [
                            [15, 13],
                            [14, 12],
                            [13, 11]
                        ],
                        'Reservatorio': False,
                        'Montante': [1]
                    }
                ],
                'UTE': [
                    {'Nome': 'GT_1', 'Capac': 20.0, 'Custo': 5.0},
                    {'Nome': 'GT_2', 'Capac': 30.0, 'Custo': 10.0}
                ]
            }
    
    def definir_caso_com_produtibilidade_nao_linear(self):
            """Caso com curvas cota-volume para produtibilidade não linear"""
            caso = self.definir_caso()
            
            for i, uhe in enumerate(caso['UHE']):
                if uhe['Reservatorio']:
                    uhe['CotaVol'] = {
                        'a1': 100.0 + i*20,
                        'a2': 0.8,
                        'a3': -0.002,
                        'a4': 0.00002,
                        'a5': -0.0000002
                    }
                    uhe['Eficiencia'] = 0.85
                    uhe['CotaFuga'] = 50.0 + i*10
                    uhe['Perdas'] = 0.5
                    
            return caso
    
    def calcular_cota_reservatorio(self, uhe, volume):
        """Calcula cota do reservatório em função do volume"""
        if not uhe['Reservatorio']:
            return uhe.get('CotaFuga', 50.0)
        
        cv = uhe['CotaVol']
        return (cv['a1'] + cv['a2']*volume + cv['a3']*volume**2 + 
                cv['a4']*volume**3 + cv['a5']*volume**4)
    
    def calcular_altura_queda_equivalente(self, uhe, volume):
        """Calcula altura de queda equivalente"""
        if not uhe['Reservatorio']:
            return 10.0  # Altura fixa para fio d'água
        
        cota_reservatorio = self.calcular_cota_reservatorio(uhe, volume)
        cota_fuga = uhe.get('CotaFuga', 10.0)
        perdas = uhe.get('Perdas', 0.5)
        
        return max(cota_reservatorio - cota_fuga - perdas, 10.0)
    
    def calcular_produtibilidade_instantanea(self, uhe, volume):
        """Calcula produtibilidade instantânea baseada na altura de queda"""
        if not uhe['Reservatorio']:
            return uhe['Prod']
        
        altura_queda = self.calcular_altura_queda_equivalente(uhe, volume)
        eficiencia = uhe.get('Eficiencia', 0.85)
        
        # ρ = η × g × H / 1000 [MW/(m³/s)]
        prod_instantanea = (eficiencia * 9.81 * altura_queda) / 1000.0
        
        return max(prod_instantanea, 0.1)
    
    def calcular_energia_armazenada(self, uhe, volume):
        """Calcula energia armazenada integrando a produtibilidade"""
        if not uhe['Reservatorio']:
            return 0.0
        
        # Integração numérica da energia
        n_pontos = 20
        volumes = np.linspace(uhe['Vmin'], volume, n_pontos)
        energia = 0.0
        
        for i in range(1, len(volumes)):
            v1, v2 = volumes[i-1], volumes[i]
            rho1 = self.calcular_produtibilidade_instantanea(uhe, v1)
            rho2 = self.calcular_produtibilidade_instantanea(uhe, v2)
            energia += 0.5 * (rho1 + rho2) * (v2 - v1)
        
        return energia
    
    def criar_reservatorio_equivalente(self, caso, volumes):
        """Cria reservatório equivalente com produtibilidade, cota e altura equivalentes"""
        
        uhes_reservatorio = [i for i, uhe in enumerate(caso['UHE']) if uhe['Reservatorio']]
        uhes_fiodagua = [i for i, uhe in enumerate(caso['UHE']) if not uhe['Reservatorio']]
        
        # =============================================================================
        # 1. CÁLCULO DAS GRANDEZAS EQUIVALENTES
        # =============================================================================
        
        # Volumes totais
        volume_total = sum(volumes[i] for i in uhes_reservatorio)
        
        # Energia armazenada total
        energia_total = sum(self.calcular_energia_armazenada(caso['UHE'][i], volumes[i]) 
                          for i in uhes_reservatorio)
        
        # Cota equivalente (média ponderada pelo volume)
        if volume_total > 0:
            cota_equiv = sum(self.calcular_cota_reservatorio(caso['UHE'][i], volumes[i]) * volumes[i] 
                           for i in uhes_reservatorio) / volume_total
        else:
            cota_equiv = 100.0
        
        # Altura de queda equivalente (média ponderada)
        if volume_total > 0:
            altura_equiv = sum(self.calcular_altura_queda_equivalente(caso['UHE'][i], volumes[i]) * volumes[i] 
                             for i in uhes_reservatorio) / volume_total
        else:
            altura_equiv = 50.0
        
        # Produtibilidade equivalente (média ponderada pelo volume)
        if volume_total > 0:
            prod_equiv = sum(self.calcular_produtibilidade_instantanea(caso['UHE'][i], volumes[i]) * volumes[i] 
                           for i in uhes_reservatorio) / volume_total
        else:
            prod_equiv = 0.7
        
        # Engolimento equivalente
        engol_equiv = sum(caso['UHE'][i]['Engol'] for i in uhes_reservatorio)
        
        # =============================================================================
        # 2. CURVA COTA-VOLUME EQUIVALENTE (aproximação linear)
        # =============================================================================
        
        # Para o equivalente, usamos uma curva linear simplificada
        # cota = a + b * volume
        vol_min_equiv = sum(caso['UHE'][i]['Vmin'] for i in uhes_reservatorio)
        vol_max_equiv = sum(caso['UHE'][i]['Vmax'] for i in uhes_reservatorio)
        
        # Calcular cotas nos pontos mínimo e máximo
        cota_min = sum(self.calcular_cota_reservatorio(caso['UHE'][i], caso['UHE'][i]['Vmin']) 
                      for i in uhes_reservatorio) / len(uhes_reservatorio)
        
        cota_max = sum(self.calcular_cota_reservatorio(caso['UHE'][i], caso['UHE'][i]['Vmax']) 
                      for i in uhes_reservatorio) / len(uhes_reservatorio)
        
        # Coeficientes da reta
        b_equiv = (cota_max - cota_min) / (vol_max_equiv - vol_min_equiv) if (vol_max_equiv - vol_min_equiv) > 0 else 0
        a_equiv = cota_min - b_equiv * vol_min_equiv
        
        # =============================================================================
        # 3. PRODUTIBILIDADE EQUIVALENTE EM FUNÇÃO DO VOLUME
        # =============================================================================
        
        # Para o equivalente, a produtibilidade varia linearmente com o volume
        # ρ = c + d * volume
        
        # Calcular produtibilidades nos pontos mínimo e máximo
        prod_min = sum(self.calcular_produtibilidade_instantanea(caso['UHE'][i], caso['UHE'][i]['Vmin']) 
                      for i in uhes_reservatorio) / len(uhes_reservatorio)
        prod_max = sum(self.calcular_produtibilidade_instantanea(caso['UHE'][i], caso['UHE'][i]['Vmax']) 
                      for i in uhes_reservatorio) / len(uhes_reservatorio)
        
        d_equiv = (prod_max - prod_min) / (vol_max_equiv - vol_min_equiv) if (vol_max_equiv - vol_min_equiv) > 0 else 0
        c_equiv = prod_min - d_equiv * vol_min_equiv
        
        equivalente = {
            # Grandezas básicas
            'volume_total': volume_total,
            'energia_armazenada': energia_total,
            'produtibilidade_equivalente': prod_equiv,
            'cota_equivalente': cota_equiv,
            'altura_queda_equivalente': altura_equiv,
            'engolimento_equivalente': engol_equiv,
            
            # Curvas equivalentes
            'CotaVol_equiv': {'a': a_equiv, 'b': b_equiv},
            'ProdVol_equiv': {'c': c_equiv, 'd': d_equiv},
            
            # Informações de agregação
            'usinas_agregadas': uhes_reservatorio,
            'usinas_fiodagua': uhes_fiodagua,
            'volumes_individuais': [volumes[i] for i in uhes_reservatorio],
            
            # Limites
            'Vmin_equiv': vol_min_equiv,
            'Vmax_equiv': vol_max_equiv,
            'ProdMin_equiv': max(prod_min, 0.1),
            'ProdMax_equiv': min(prod_max, 1.5)
        }
        
        return equivalente
    
    def calcular_produtibilidade_equivalente(self, equivalente, volume_equiv):
        """Calcula produtibilidade do reservatório equivalente em função do volume"""
        pv = equivalente['ProdVol_equiv']
        prod = pv['c'] + pv['d'] * volume_equiv
        return max(equivalente['ProdMin_equiv'], min(prod, equivalente['ProdMax_equiv']))
    
    def calcular_energia_armazenada_equivalente(self, equivalente, volume_equiv):
        """Calcula energia armazenada do equivalente integrando a produtibilidade"""
        # Para curva linear, a energia é a integral: E = ∫ (c + d*v) dv = c*v + d*v²/2
        pv = equivalente['ProdVol_equiv']
        energia = (pv['c'] * volume_equiv + pv['d'] * volume_equiv**2 / 2)
        return max(energia, 0.0)
    
    def processar_resultados_equivalentes(self, model, caso, equivalente, volume_inicial):
        """Processa resultados do modelo equivalente"""
        
        volumes_finais = []
        turbinamentos = []
        vertimentos = []
        
        # Distribuir resultados do equivalente para as usinas individuais
        for i, uhe in enumerate(caso['UHE']):
            if uhe['Reservatorio']:
                # Distribuição proporcional baseada no volume inicial
                proporcao = volume_inicial[i] / equivalente['volume_total'] if equivalente['volume_total'] > 0 else 0
                vol_final = value(model.x_volume_final_equiv) * proporcao
                turb = value(model.x_volume_turbinado_equiv) * proporcao
                vert = value(model.x_volume_vertido_equiv) * proporcao
            else:
                # Fio d'água
                vol_final = 0.0
                turb = value(model.x_volume_turbinado_fiodagua[i])
                vert = value(model.x_volume_vertido_fiodagua[i])
            
            volumes_finais.append(vol_final)
            turbinamentos.append(turb)
            vertimentos.append(vert)
        
        # Calcular geração térmica
        geracao_termica = [value(model.x_geracao_termica[i]) for i in model.UTE]
        
        # Calcular custos
        custo_imediato = 0
        for i in model.UTE:
            custo_imediato += value(model.x_geracao_termica[i]) * caso['UTE'][value(i)]['Custo']
        custo_imediato += value(model.x_deficit) * caso['DGer']['CDef']
        custo_imediato += 0.001 * value(model.x_volume_vertido_equiv)
        
        custo_futuro = value(model.x_alpha)
        
        return {
            'status': 'optimal',
            'custo_imediato': max(0, custo_imediato),
            'custo_futuro': max(0, custo_futuro),
            'cma_duais': [-0.1] * len(caso['UHE']),  # Aproximado
            'cmo_dual': caso['DGer']['CDef'] * 0.5,  # Aproximado
            'volumes_finais': volumes_finais,
            'geracao_termica': geracao_termica,
            'turbinamento': turbinamentos,
            'vertimento': vertimentos,
            'deficit': value(model.x_deficit),
            'equivalente': equivalente
        }

    def executar_pdde(self, caso):
        print("=== PDDE com Modelagem Não Linear Completa ===")
        print("=== Usando relações cota-volume diretamente no modelo ===")
        
        # Configurações
        NCEN = caso['DGer']['Nr_Cen']
        NDISC = caso['DGer']['Nr_Disc']
        NEST = caso['DGer']['Nr_Est']
        NUHE = len(caso['UHE'])
        NUTE = len(caso['UTE'])

        print(f"Sistema: {NUHE} UHEs, {NUTE} UTE, {NEST} estágios, {NCEN} cenários")
        
        # Identificar usinas com reservatório
        uhes_reservatorio = [i for i, uhe in enumerate(caso['UHE']) if uhe['Reservatorio']]
        NUHE_RESERV = len(uhes_reservatorio)
        
        print(f"Usinas com reservatório: {NUHE_RESERV}")
        print(f"Usina fio d'água: {[i for i, uhe in enumerate(caso['UHE']) if not uhe['Reservatorio']]}")
        
        # Discretização para reservatórios
        disc_arrays = [np.linspace(30, 70, NDISC) for _ in range(NUHE_RESERV)]
        discretizacoes = list(product(*disc_arrays))
        print(f"Pontos de discretização: {len(discretizacoes)}")
        
        listaDeCortes = []
        
        # LOOP - BACKWARD
        for iest in range(NEST, 0, -1):
            print(f"\n*** ESTÁGIO {iest} ***")
            print(f"{'='*50}")
            
            cortes_futuros = [c for c in listaDeCortes if c['estagio'] > iest]
            print(f"Cortes futuros carregados: {len(cortes_futuros)}")
            
            cortes_estagio_atual = []
            
            for idx, discretizacao in enumerate(discretizacoes):
                volume_inicial = [0.0] * NUHE
                
                for j, uhe_idx in enumerate(uhes_reservatorio):
                    usi = caso['UHE'][uhe_idx]
                    vol = usi['Vmin'] + ((usi['Vmax'] - usi['Vmin']) * discretizacao[j] / 100)
                    volume_inicial[uhe_idx] = vol
                
                print(f"\n  Ponto {idx+1}/{len(discretizacoes)}")
                reserv_volumes = [f'{volume_inicial[i]:.1f}' for i in uhes_reservatorio]
                print(f"    Volume Inicial Reservatórios = {reserv_volumes}")
                
                custo_total_acum = 0.0
                cma_duais_acum = np.zeros(NUHE)
                cmo_duais_acum = 0.0
                cenarios_validos = 0
                
                for icen in range(NCEN):
                    aflu = []
                    for i, usi in enumerate(caso['UHE']):
                        aflu.append(usi['Afl'][iest-1][icen])
                    
                    print(f"    Afluencia = {aflu}")
                    
                    # USAR A NOVA FUNÇÃO COM MODELAGEM NÃO LINEAR
                    resultado = self.solve_DespachoEconomico_NaoLinear(
                        volume_inicial, aflu, NUHE, NUTE, iest, caso, cortes_futuros
                    )
                    
                    if resultado is not None and resultado['status'] == 'optimal':
                        custo_cenario = resultado['custo_imediato'] + resultado['custo_futuro']
                        custo_total_acum += custo_cenario
                        # Para não linear, usar aproximações para duais
                        cma_duais_acum += resultado['cma_duais']
                        cmo_duais_acum += resultado['cmo_dual']
                        cenarios_validos += 1
                        
                        self.adicionar_ao_relatorio(iest, idx, icen, volume_inicial, aflu, resultado, caso)
                        
                        print(f"    Custo total: {custo_cenario:.2f}")
                        print(f"    Déficit: {resultado['deficit']:.2f}")
                        print(f"    Solver: {resultado.get('solver_utilizado', 'N/A')}")
                    else:
                        print(f"    SOLUÇÃO NÃO ÓTIMA!")
                        # Fallback para modelo linear se necessário
                        # resultado_fallback = self.solve_DespachoEconomico(
                        #     volume_inicial, aflu, NUHE, NUTE, iest, caso, cortes_futuros
                        # )
                        # if resultado_fallback and resultado_fallback['status'] == 'optimal':
                        #     custo_cenario = resultado_fallback['custo_imediato'] + resultado_fallback['custo_futuro']
                        #     custo_total_acum += custo_cenario
                        #     cma_duais_acum += resultado_fallback['cma_duais']
                        #     cmo_duais_acum += resultado_fallback['cmo_dual']
                        #     cenarios_validos += 1
                        #     self.adicionar_ao_relatorio(iest, idx, icen, volume_inicial, aflu, resultado_fallback, caso)
                        #     print(f"    Usando fallback linear. Custo: {custo_cenario:.2f}")
                        # else:
                        #     custo_total_acum += 1e6
                        #     cenarios_validos += 1
                
                if cenarios_validos > 0:
                    custo_medio = custo_total_acum / cenarios_validos
                    cma_medio = cma_duais_acum / cenarios_validos
                    cmo_medio = cmo_duais_acum / cenarios_validos
                else:
                    custo_medio = 1e6
                    cma_medio = np.zeros(NUHE)
                    cmo_medio = caso['DGer']['CDef']
                
                # Calcular coeficiente para o corte
                coef_equiv = -0.1  # Coeficiente aproximado para não linear
                
                termo_independente = custo_medio - coef_equiv * sum(volume_inicial)
                termo_independente = max(0.0, termo_independente)
                
                corte = {
                    'estagio': iest,
                    'coefs': cma_medio.tolist(),
                    'coef_equiv': coef_equiv,
                    'termo_independente': termo_independente,
                    'volume_inicial': volume_inicial.copy(),
                    'custo_medio': custo_medio,
                    'cmo_medio': cmo_medio,
                }
                
                cortes_estagio_atual.append(corte)
                listaDeCortes.append(corte)
                
                print(f"    Corte gerado - Coef Equiv: {coef_equiv:.6f}, Termo: {termo_independente:.6f}")
            
            if NUHE_RESERV == 2:
                print(f"\nPlotando superfície de custo futuro para estágio {iest}...")
                self.plota_FuncaoCustoFuturo_3D(listaDeCortes, caso, iest, uhes_reservatorio)
            
            self.gerar_relatorio_estagio(iest, caso)
        
        self.gerar_relatorio_final(caso)

    def adicionar_ao_relatorio(self, estagio, ponto_discretizacao, cenario, volume_inicial, afluencia, resultado, caso):
        """Adiciona uma entrada ao relatório de operação - ATUALIZADO PARA NÃO LINEAR"""
        entrada = {
            'estagio': estagio,
            'ponto_discretizacao': ponto_discretizacao,
            'cenario': cenario,
            'volume_inicial': volume_inicial.copy(),
            'afluencia': afluencia.copy(),
            'volumes_finais': resultado['volumes_finais'].copy(),
            'geracao_uhe': [],
            'geracao_ute': resultado['geracao_termica'].copy(),
            'turbinamento': resultado['turbinamento'].copy(),
            'vertimento': resultado['vertimento'].copy(),
            'deficit': resultado['deficit'],
            'custo_imediato': resultado['custo_imediato'],
            'custo_futuro': resultado['custo_futuro'],
            'custo_total': resultado['custo_imediato'] + resultado['custo_futuro'],
            'produtibilidades': resultado.get('produtibilidades', []),
            'alturas_queda': resultado.get('alturas_queda', []),
            'cotas_montante': resultado.get('cotas_montante', []),
            'solver_utilizado': resultado.get('solver_utilizado', 'N/A')
        }
        
        # Calcular geração das UHEs usando produtibilidade variável do modelo não linear
        if 'produtibilidades' in resultado and resultado['produtibilidades']:
            # Usar as produtibilidades calculadas pelo modelo não linear
            for i, uhe in enumerate(caso['UHE']):
                geracao = resultado['turbinamento'][i] * resultado['produtibilidades'][i]
                entrada['geracao_uhe'].append(geracao)
        else:
            # Fallback: usar cálculo externo
            for i, uhe in enumerate(caso['UHE']):
                if uhe['Reservatorio']:
                    prod = self.calcular_produtibilidade_instantanea(uhe, resultado['volumes_finais'][i])
                else:
                    prod = uhe['Prod']
                geracao = resultado['turbinamento'][i] * prod
                entrada['geracao_uhe'].append(geracao)
        
        self.relatorio_completo.append(entrada)

    def plotar_curvas_equivalentes(self, caso):
        """Plota as curvas do reservatório equivalente"""
        # Testar com volumes médios
        volumes_test = [50.0, 60.0, 40.0,]  # Volumes de teste
        equivalente = self.criar_reservatorio_equivalente(caso, volumes_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Curva cota-volume equivalente
        vol_range = np.linspace(equivalente['Vmin_equiv'], equivalente['Vmax_equiv'], 50)
        cotas_equiv = [equivalente['CotaVol_equiv']['a'] + equivalente['CotaVol_equiv']['b'] * v 
                      for v in vol_range]
        
        axes[0,0].plot(vol_range, cotas_equiv, 'b-', linewidth=2, label='Equivalente')
        
        # Plotar curvas individuais para comparação
        for i in equivalente['usinas_agregadas']:
            uhe = caso['UHE'][i]
            vol_uhe = np.linspace(uhe['Vmin'], uhe['Vmax'], 50)
            cotas_uhe = [self.calcular_cota_reservatorio(uhe, v) for v in vol_uhe]
            axes[0,0].plot(vol_uhe, cotas_uhe, '--', alpha=0.7, label=f'UHE {i+1}')
        
        axes[0,0].set_xlabel('Volume (hm³)')
        axes[0,0].set_ylabel('Cota (m)')
        axes[0,0].set_title('Curva Cota-Volume Equivalente')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Curva produtibilidade-volume equivalente
        prod_equiv = [self.calcular_produtibilidade_equivalente(equivalente, v) for v in vol_range]
        
        axes[0,1].plot(vol_range, prod_equiv, 'r-', linewidth=2, label='Equivalente')
        
        # Plotar curvas individuais
        for i in equivalente['usinas_agregadas']:
            uhe = caso['UHE'][i]
            vol_uhe = np.linspace(uhe['Vmin'], uhe['Vmax'], 50)
            prod_uhe = [self.calcular_produtibilidade_instantanea(uhe, v) for v in vol_uhe]
            axes[0,1].plot(vol_uhe, prod_uhe, '--', alpha=0.7, label=f'UHE {i+1}')
        
        axes[0,1].set_xlabel('Volume (hm³)')
        axes[0,1].set_ylabel('Produtibilidade (MW/m³/s)')
        axes[0,1].set_title('Curva Produtibilidade-Volume Equivalente')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Curva energia-volume equivalente
        energia_equiv = [self.calcular_energia_armazenada_equivalente(equivalente, v) for v in vol_range]
        
        axes[1,0].plot(vol_range, energia_equiv, 'g-', linewidth=2, label='Equivalente')
        
        # Plotar curvas individuais
        for i in equivalente['usinas_agregadas']:
            uhe = caso['UHE'][i]
            vol_uhe = np.linspace(uhe['Vmin'], uhe['Vmax'], 50)
            energia_uhe = [self.calcular_energia_armazenada(uhe, v) for v in vol_uhe]
            axes[1,0].plot(vol_uhe, energia_uhe, '--', alpha=0.7, label=f'UHE {i+1}')
        
        axes[1,0].set_xlabel('Volume (hm³)')
        axes[1,0].set_ylabel('Energia Armazenada (MWmes)')
        axes[1,0].set_title('Curva Energia-Volume Equivalente')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Resumo do equivalente
        axes[1,1].axis('off')
        info_text = (
            f"Reservatório Equivalente - Resumo\n\n"
            f"Volume Total: {equivalente['volume_total']:.1f} hm³\n"
            f"Energia Armazenada: {equivalente['energia_armazenada']:.1f} MWmes\n"
            f"Produtibilidade Equivalente: {equivalente['produtibilidade_equivalente']:.3f} MW/m³/s\n"
            f"Cota Equivalente: {equivalente['cota_equivalente']:.1f} m\n"
            f"Altura Queda Equivalente: {equivalente['altura_queda_equivalente']:.1f} m\n"
            f"Engolimento Equivalente: {equivalente['engolimento_equivalente']:.1f} m³/s\n"
            f"Usinas Agregadas: {len(equivalente['usinas_agregadas'])}\n"
            f"Usinas Fio d'Água: {len(equivalente['usinas_fiodagua'])}"
        )
        axes[1,1].text(0.1, 0.9, info_text, transform=axes[1,1].transAxes, fontsize=12,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

    def gerar_relatorio_estagio(self, estagio, caso):
        """Gera relatório detalhado para um estágio específico"""
        print(f"\n{'='*80}")
        print(f"RELATÓRIO DE GERAÇÃO - ESTÁGIO {estagio}")
        print(f"{'='*80}")
        
        # Filtrar entradas do estágio
        entradas_estagio = [e for e in self.relatorio_completo if e['estagio'] == estagio]
        
        if not entradas_estagio:
            print("Nenhum dado disponível para este estágio.")
            return
        
        # Calcular médias por ponto de discretização
        pontos_unicos = set(e['ponto_discretizacao'] for e in entradas_estagio)
        
        for ponto in sorted(pontos_unicos):
            entradas_ponto = [e for e in entradas_estagio if e['ponto_discretizacao'] == ponto]
            
            print(f"\n--- Ponto de Discretização {ponto + 1} ---")
            
            # Médias dos cenários para este ponto
            n_cenarios = len(entradas_ponto)
            ger_uhe_media = np.zeros(len(caso['UHE']))
            ger_ute_media = np.zeros(len(caso['UTE']))
            turb_medio = np.zeros(len(caso['UHE']))
            vol_final_medio = np.zeros(len(caso['UHE']))
            vertimento_medio = np.zeros(len(caso['UHE']))
            deficit_medio = 0
            custo_medio = 0
            
            for entrada in entradas_ponto:
                for i in range(len(caso['UHE'])):
                    ger_uhe_media[i] += entrada['geracao_uhe'][i]
                    turb_medio[i] += entrada['turbinamento'][i]
                    vertimento_medio[i] += entrada['vertimento'][i]
                    vol_final_medio[i] += entrada['volumes_finais'][i]
                
                for i in range(len(caso['UTE'])):
                    ger_ute_media[i] += entrada['geracao_ute'][i]
                
                deficit_medio += entrada['deficit']
                custo_medio += entrada['custo_total']
            
            ger_uhe_media /= n_cenarios
            ger_ute_media /= n_cenarios
            turb_medio /= n_cenarios
            vol_final_medio /= n_cenarios
            deficit_medio /= n_cenarios
            custo_medio /= n_cenarios
            
            # Volume inicial (igual para todos os cenários do mesmo ponto)
            vol_inicial = entradas_ponto[0]['volume_inicial']
            
            print(f"Volumes Iniciais: {[f'{v:.1f}' for v in vol_inicial]}")
            print(f"Volumes Finais:   {[f'{v:.1f}' for v in vol_final_medio]}")
            afluencia_value = [f"{entradas_ponto[0]['afluencia'][i]:.1f}" for i in range(len(caso['UHE']))]
            print(f"Afluência:        {afluencia_value}")
            
            print(f"\nGERAÇÃO HIDRELÉTRICA (MW):")
            for i, uhe in enumerate(caso['UHE']):
                print(f"  {uhe['Nome']}: {ger_uhe_media[i]:.2f} MW (Turb: {turb_medio[i]:.1f} / Vert: {vertimento_medio[i]:.1f} m³/s)")
            
            print(f"\nGERAÇÃO TÉRMICA (MW):")
            for i, ute in enumerate(caso['UTE']):
                print(f"  {ute['Nome']}: {ger_ute_media[i]:.2f} MW")
            
            print(f"\nSISTEMA:")
            print(f"  Déficit: {deficit_medio:.2f} MW")
            print(f"  Custo Total: R$ {custo_medio:.2f}")
            print(f"  Demanda: {caso['DGer']['Carga'][estagio-1]:.1f} MW")
            
            # Verificar balanço energético
            total_gerado = sum(ger_uhe_media) + sum(ger_ute_media) + deficit_medio
            print(f"  Balanço Energético: {total_gerado:.1f} MW (Demanda: {caso['DGer']['Carga'][estagio-1]:.1f} MW)")

    def gerar_relatorio_final(self, caso):
        """Gera relatório final consolidado de todo o processo"""
        print(f"\n{'='*100}")
        print(f"RELATÓRIO FINAL CONSOLIDADO - PDDE")
        print(f"{'='*100}")
        print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sistema: {len(caso['UHE'])} UHEs, {len(caso['UTE'])} UTE, {caso['DGer']['Nr_Est']} estágios")
        print(f"{'='*100}")
        
        # Estatísticas por estágio
        for estagio in sorted(set(e['estagio'] for e in self.relatorio_completo)):
            entradas_estagio = [e for e in self.relatorio_completo if e['estagio'] == estagio]
            
            print(f"\nESTÁGIO {estagio}:")
            print(f"{'-'*50}")
            
            # Calcular totais médios
            total_ger_uhe = np.zeros(len(caso['UHE']))
            total_ger_ute = np.zeros(len(caso['UTE']))
            total_deficit = 0
            total_custo = 0
            
            for entrada in entradas_estagio:
                for i in range(len(caso['UHE'])):
                    total_ger_uhe[i] += entrada['geracao_uhe'][i]
                for i in range(len(caso['UTE'])):
                    total_ger_ute[i] += entrada['geracao_ute'][i]
                total_deficit += entrada['deficit']
                total_custo += entrada['custo_total']
            
            n_entradas = len(entradas_estagio)
            total_ger_uhe /= n_entradas
            total_ger_ute /= n_entradas
            total_deficit /= n_entradas
            total_custo /= n_entradas
            
            print("Geração Média por Usina:")
            for i, uhe in enumerate(caso['UHE']):
                print(f"  {uhe['Nome']:25} {total_ger_uhe[i]:8.2f} MW")
            
            for i, ute in enumerate(caso['UTE']):
                print(f"  {ute['Nome']:25} {total_ger_ute[i]:8.2f} MW")
            
            print(f"\n  {'Déficit Médio:':25} {total_deficit:8.2f} MW")
            print(f"  {'Custo Total Médio:':25} R$ {total_custo:8.2f}")
            
            # Participação percentual
            total_geracao = sum(total_ger_uhe) + sum(total_ger_ute)
            if total_geracao > 0:
                print(f"\nParticipação na Geração:")
                for i, uhe in enumerate(caso['UHE']):
                    participacao = (total_ger_uhe[i] / total_geracao) * 100
                    print(f"  {uhe['Nome']:25} {participacao:6.1f}%")
                
                for i, ute in enumerate(caso['UTE']):
                    participacao = (total_ger_ute[i] / total_geracao) * 100
                    print(f"  {ute['Nome']:25} {participacao:6.1f}%")
        
        # Estatísticas globais
        print(f"\n{'='*100}")
        print("ESTATÍSTICAS GLOBAIS:")
        print(f"{'-'*100}")
        
        total_ger_uhe_global = np.zeros(len(caso['UHE']))
        total_ger_ute_global = np.zeros(len(caso['UTE']))
        total_deficit_global = 0
        total_custo_global = 0
        
        for entrada in self.relatorio_completo:
            for i in range(len(caso['UHE'])):
                total_ger_uhe_global[i] += entrada['geracao_uhe'][i]
            for i in range(len(caso['UTE'])):
                total_ger_ute_global[i] += entrada['geracao_ute'][i]
            total_deficit_global += entrada['deficit']
            total_custo_global += entrada['custo_total']
        
        n_total = len(self.relatorio_completo)
        total_ger_uhe_global /= n_total
        total_ger_ute_global /= n_total
        total_deficit_global /= n_total
        total_custo_global /= n_total
        
        print("Médias Globais:")
        for i, uhe in enumerate(caso['UHE']):
            print(f"  {uhe['Nome']:25} {total_ger_uhe_global[i]:8.2f} MW")
        
        for i, ute in enumerate(caso['UTE']):
            print(f"  {ute['Nome']:25} {total_ger_ute_global[i]:8.2f} MW")
        
        print(f"\n  {'Déficit Médio Global:':25} {total_deficit_global:8.2f} MW")
        print(f"  {'Custo Total Médio Global:':25} R$ {total_custo_global:8.2f}")
        
        total_geracao_global = sum(total_ger_uhe_global) + sum(total_ger_ute_global)
        print(f"  {'Geração Total Média:':25} {total_geracao_global:8.2f} MW")
        
        print(f"\nTotal de Simulações: {n_total}")
        print(f"{'='*100}")

    def solve_DespachoEconomico_NaoLinear(self, volume_inicial, afluencia, nuhe, nute, imes, caso, cortes):
            """Resolve o problema de despacho econômico com modelagem não linear completa"""
            try:
                model = ConcreteModel()
                # Conjuntos
                model.UHE = RangeSet(0, nuhe-1)
                model.UTE = RangeSet(0, nute-1)
                
                # Variáveis de decisão
                model.x_volume_final = Var(model.UHE, 
                                        bounds=lambda m, i: (caso['UHE'][i]['Vmin'], caso['UHE'][i]['Vmax']))
                model.x_volume_turbinado = Var(model.UHE, 
                                            bounds=lambda m, i: (0, caso['UHE'][i]['Engol']))
                model.x_volume_vertido = Var(model.UHE, bounds=(0, None))
                model.x_geracao_termica = Var(model.UTE, 
                                            bounds=lambda m, i: (0, caso['UTE'][i]['Capac']))
                model.x_deficit = Var(bounds=(0, None))
                model.x_alpha = Var(bounds=(0, None))
                
                # NOVAS VARIÁVEIS PARA MODELAGEM NÃO LINEAR
                model.x_cota_montante = Var(model.UHE, bounds=(0, 500))  # Cota a montante
                model.x_altura_queda = Var(model.UHE, bounds=(0, 200))   # Altura de queda
                model.x_produtibilidade = Var(model.UHE, bounds=(0.1, 2.0)) # Produtibilidade dinâmica
                
                # Função objetivo
                def FOB(m):
                    custo_termica = sum(caso['UTE'][i]['Custo'] * m.x_geracao_termica[i] for i in m.UTE)
                    penal_vertim = sum(0.001 * m.x_volume_vertido[i] for i in m.UHE)
                    return custo_termica + penal_vertim + caso['DGer']['CDef'] * m.x_deficit + m.x_alpha
                
                model.FOB = Objective(rule=FOB, sense=minimize)
                
                # NOVAS RESTRIÇÕES NÃO LINEARES
                def C_CotaVolume(m, i):
                    """RESTRIÇÃO NÃO LINEAR: Relação cota-volume"""
                    if caso['UHE'][i]['Reservatorio']:
                        cv = caso['UHE'][i]['CotaVol']
                        return m.x_cota_montante[i] == (
                            cv['a1'] + 
                            cv['a2'] * m.x_volume_final[i] + 
                            cv['a3'] * m.x_volume_final[i]**2 + 
                            cv['a4'] * m.x_volume_final[i]**3 + 
                            cv['a5'] * m.x_volume_final[i]**4
                        )
                    else:
                        # Fio d'água: cota constante
                        return m.x_cota_montante[i] == caso['UHE'][i].get('CotaFuga', 50.0)
                
                def C_AlturaQueda(m, i):
                    """RESTRIÇÃO NÃO LINEAR: Altura de queda = cota_montante - cota_fuga - perdas"""
                    if caso['UHE'][i]['Reservatorio']:
                        return m.x_altura_queda[i] == (
                            m.x_cota_montante[i] - 
                            caso['UHE'][i].get('CotaFuga', 50.0) - 
                            caso['UHE'][i].get('Perdas', 0.5)
                        )
                    else:
                        # Fio d'água: altura fixa
                        return m.x_altura_queda[i] == 10.0
                
                def C_Produtibilidade(m, i):
                    """RESTRIÇÃO NÃO LINEAR: Produtibilidade = η*g*H / 1MW """
                    if caso['UHE'][i]['Reservatorio']:
                        eficiencia = caso['UHE'][i].get('Eficiencia', 0.85)
                        return m.x_produtibilidade[i] == (eficiencia * 9.81 * m.x_altura_queda[i]) / 1e3 #[MW]
                    else:
                        # Fio d'água: produtibilidade fixa
                        return m.x_produtibilidade[i] == caso['UHE'][i]['Prod']
                
                # Restrições de operação hidráulica
                def C_VolFinal_FioDagua(m, i):
                    """Restrição de volume para fio d'água"""
                    if caso['UHE'][i]['Reservatorio'] == False:
                        return m.x_volume_final[i] == 0.0
                    else:
                        return Constraint.Skip
                
                def C_BalancoHidrico(m, i):
                    """Restrição de operação"""
                    uhe = caso['UHE'][i]
                    
                    if uhe['Reservatorio'] == False:
                        # Fio d'água: calcula a afluência total 
                        afluencia_total = afluencia[i]
                        
                        # Adiciona a contribuição das usinas a montante
                        for montante_id in uhe['Montante']:
                            afluencia_total += (m.x_volume_turbinado[montante_id] + 
                                            m.x_volume_vertido[montante_id])
                        
                        return m.x_volume_turbinado[i] + m.x_volume_vertido[i] == afluencia_total
                    else:
                        # Reservatório: balanço tradicional
                        return (m.x_volume_final[i] + m.x_volume_turbinado[i] + m.x_volume_vertido[i] == 
                            volume_inicial[i] + afluencia[i])
                
                # MODIFICAÇÃO IMPORTANTE: Balanço de potência com produtibilidade variável
                
                def C_BalancoDePotencia(m):
                    """RESTRIÇÃO NÃO LINEAR: Balanço de potência com produtibilidade dinâmica"""
                    gterm = sum(m.x_geracao_termica[i] for i in m.UTE)
                    # AGORA usa produtibilidade variável em vez de fixa
                    ghidr = sum(m.x_produtibilidade[i] * m.x_volume_turbinado[i] for i in m.UHE)
                    return gterm + ghidr + m.x_deficit == caso['DGer']['Carga'][imes-1]
                
                # Adicionar restrições ao modelo
                model.C_CotaVolume = Constraint(model.UHE, rule=C_CotaVolume)
                model.C_AlturaQueda = Constraint(model.UHE, rule=C_AlturaQueda)
                model.C_Produtibilidade = Constraint(model.UHE, rule=C_Produtibilidade)
                model.C_VolFinal_FioDagua = Constraint(model.UHE, rule=C_VolFinal_FioDagua)
                model.C_BalancoHidrico = Constraint(model.UHE, rule=C_BalancoHidrico)
                model.C_BalancoDePotencia = Constraint(rule=C_BalancoDePotencia)
                
                # Cortes de Benders
                cortes_validos = [c for c in cortes if c['estagio'] == imes + 1]
                
                if cortes_validos:
                    def C_cortes(m, idx):
                        corte = cortes_validos[idx]
                        somatorio = sum(corte['coefs'][i] * m.x_volume_final[i] for i in m.UHE)
                        return m.x_alpha >= somatorio + corte['termo_independente']
                    
                    model.cortes = Constraint(RangeSet(0, len(cortes_validos)-1), rule=C_cortes)
                
                # RESOLVER COM SOLVER NÃO LINEAR
                print(f"    Resolvendo com solver não linear...")
                
                # Tentar diferentes solvers não lineares
                solvers_nao_lineares = ['ipopt', 'bonmin', 'couenne']
                solver_encontrado = None
                
                for solver_name in solvers_nao_lineares:
                    if SolverFactory(solver_name).available():
                        solver_encontrado = solver_name
                        break
                
                if solver_encontrado is None:
                    print("    Nenhum solver não linear encontrado. Usando glpk como fallback.")
                    solver_encontrado = 'glpk'
                
                solver = SolverFactory(solver_encontrado)
                
                # Configurar opções para solvers não lineares
                if solver_encontrado == 'ipopt':
                    solver.options['max_iter'] = 1000
                    solver.options['tol'] = 1e-6
                    solver.options['linear_solver'] = 'ma57'  # Solver linear mais robusto
                elif solver_encontrado == 'bonmin':
                    solver.options['max_iter'] = 1000
                    solver.options['tol'] = 1e-6
                elif solver_encontrado == 'couenne':
                    solver.options['allowable_gap'] = 1e-6
                    solver.options['max_iter'] = 1000
                
                print(f"    Usando solver: {solver_encontrado}")
                
                # Resolver o modelo
                results = solver.solve(model, tee=False)
                
                if results.solver.termination_condition == TerminationCondition.optimal:
                    # Processar resultados
                    custo_imediato = 0
                    for i in model.UTE:
                        custo_imediato += value(model.x_geracao_termica[i]) * caso['UTE'][value(i)]['Custo']
                    
                    custo_imediato += value(model.x_deficit) * caso['DGer']['CDef']
                    custo_imediato += 0.001 * sum(value(model.x_volume_vertido[i]) for i in model.UHE)
                    custo_futuro = value(model.x_alpha)
                    
                    # Extrair valores das novas variáveis não lineares
                    cotas_montante = [value(model.x_cota_montante[i]) for i in model.UHE]
                    alturas_queda = [value(model.x_altura_queda[i]) for i in model.UHE]
                    produtibilidades = [value(model.x_produtibilidade[i]) for i in model.UHE]
                    
                    print(f"    Produtibilidades calculadas: {[f'{p:.3f}' for p in produtibilidades]}")
                    print(f"    Alturas de queda: {[f'{h:.1f}' for h in alturas_queda]}")
                    
                    return {
                        'status': 'optimal',
                        'custo_imediato': max(0, custo_imediato),
                        'custo_futuro': max(0, custo_futuro),
                        'cma_duais': [-0.1] * nuhe,  # Simplificado para não linear
                        'cmo_dual': caso['DGer']['CDef'] * 0.5,
                        'volumes_finais': [value(model.x_volume_final[i]) for i in model.UHE],
                        'geracao_termica': [value(model.x_geracao_termica[i]) for i in model.UTE],
                        'turbinamento': [value(model.x_volume_turbinado[i]) for i in model.UHE],
                        'vertimento': [value(model.x_volume_vertido[i]) for i in model.UHE],
                        'deficit': value(model.x_deficit),
                        'produtibilidades': produtibilidades,
                        'alturas_queda': alturas_queda,
                        'cotas_montante': cotas_montante,
                        'solver_utilizado': solver_encontrado
                    }
                else:
                    print(f'  Status não ótimo: {results.solver.termination_condition}')
                    return None

            except Exception as e:
                print(f"  Erro na otimização não linear: {e}")
                return None   

    def plota_FuncaoCustoFuturo_3D(self, cortes, caso, imes, uhes_reservatorio):
        """Plota a função de custo futuro como superfície 3D considerando o reservatório equivalente"""
        if len(uhes_reservatorio) != 2:
            print("Plot 3D requer exatamente 2 usinas com reservatório")
            return False
        
        # Filtra cortes do estágio atual
        cortes_estagio = [c for c in cortes if c['estagio'] == imes]
        
        if not cortes_estagio:
            print(f"Nenhum corte encontrado para o estágio {imes}")
            return False
        
        # Índices das usinas com reservatório
        uhe1_idx, uhe2_idx = uhes_reservatorio
        
        # Cria meshgrid para as 2 UHEs com reservatório
        uhe1_range = np.linspace(caso['UHE'][uhe1_idx]['Vmin'], caso['UHE'][uhe1_idx]['Vmax'], 30)
        uhe2_range = np.linspace(caso['UHE'][uhe2_idx]['Vmin'], caso['UHE'][uhe2_idx]['Vmax'], 30)
        UHE1, UHE2 = np.meshgrid(uhe1_range, uhe2_range)
        
        # Calcula Z (custo futuro) para cada ponto do grid considerando o equivalente
        Z = np.zeros_like(UHE1)
        
        for i in range(UHE1.shape[0]):
            for j in range(UHE1.shape[1]):
                volume_uhe1 = UHE1[i, j]
                volume_uhe2 = UHE2[i, j]
                
                # Criar volumes iniciais para o ponto do grid
                VOL_FIXO = [0.0] * len(caso['UHE'])
                VOL_FIXO[uhe1_idx] = volume_uhe1
                VOL_FIXO[uhe2_idx] = volume_uhe2
                
                # Calcular o reservatório equivalente para este ponto
                equivalente = self.criar_reservatorio_equivalente(caso, VOL_FIXO)
                volume_equiv = equivalente['volume_total']
                
                # Calcula valor de cada corte neste ponto usando o coeficiente equivalente
                valores_cortes = []
                for corte in cortes_estagio:
                    # Usar o coeficiente equivalente se disponível, senão usar a média dos coeficientes individuais
                    if 'coef_equiv' in corte:
                        valor = corte['coef_equiv'] * volume_equiv + corte['termo_independente']
                    else:
                        # Fallback: usar média dos coeficientes das UHEs individuais
                        coef_medio = (corte['coefs'][uhe1_idx] + corte['coefs'][uhe2_idx]) / 2
                        valor = coef_medio * volume_equiv + corte['termo_independente']
                    
                    valores_cortes.append(max(0.0, valor))
                
                # Função de custo futuro é o máximo dos cortes
                Z[i, j] = max(valores_cortes) if valores_cortes else 0.0
        
        # Garantir que Z seja completamente não-negativo
        Z = np.maximum(Z, 0.0)
        
        # Plot 3D
        fig = plt.figure(figsize=(16, 12))
        
        # Subplot 1: Superfície de custo
        ax1 = fig.add_subplot(projection='3d')
        
        # Plota superfície
        surf = ax1.plot_surface(UHE1, UHE2, Z, cmap='plasma', alpha=0.8, 
                            linewidth=0, antialiased=True)
        
        # Scatter plot dos pontos calculados
        x_points = []
        y_points = []
        z_points = []
        
        for corte in cortes_estagio:
            x_points.append(corte['volume_inicial'][uhe1_idx])
            y_points.append(corte['volume_inicial'][uhe2_idx])
            z_points.append(corte['custo_medio'])
        
        ax1.scatter(x_points, y_points, z_points, color='red', s=50, alpha=0.9, 
                label='Pontos de Discretização')
        
        # Configurações do gráfico
        ax1.set_xlabel(f'Volume {caso["UHE"][uhe1_idx]["Nome"].split(" - ")[0]} [hm³]', 
                    fontsize=10, labelpad=10)
        ax1.set_ylabel(f'Volume {caso["UHE"][uhe2_idx]["Nome"].split(" - ")[0]} [hm³]', 
                    fontsize=10, labelpad=10)
        ax1.set_zlabel('Custo Futuro Esperado [R$]', fontsize=10, labelpad=10)
        ax1.set_title(f'Função de Custo Futuro - Estágio {imes}\n(Reservatório Equivalente)', 
                    fontsize=12, fontweight='bold', pad=15)
        
        ax1.set_zlim(bottom=0)
        fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=20, pad=0.1, label='Custo [R$]')
        ax1.legend()
        ax1.view_init(elev=25, azim=225)
        
        # # Subplot 2: Curvas de nível
        # ax2 = fig.add_subplot(122)
        
        # # Plotar curvas de nível
        # contour = ax2.contourf(UHE1, UHE2, Z, levels=20, cmap='plasma')
        # ax2.contour(UHE1, UHE2, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # # Pontos de discretização
        # ax2.scatter(x_points, y_points, color='red', s=30, alpha=0.8, 
        #         label='Pontos Calculados')
        
        # ax2.set_xlabel(f'Volume {caso["UHE"][uhe1_idx]["Nome"].split(" - ")[0]} [hm³]', fontsize=10)
        # ax2.set_ylabel(f'Volume {caso["UHE"][uhe2_idx]["Nome"].split(" - ")[0]} [hm³]', fontsize=10)
        # ax2.set_title(f'Curvas de Nível - Estágio {imes}\n(Reservatório Equivalente)', 
        #             fontsize=12, fontweight='bold', pad=15)
        # ax2.grid(True, alpha=0.3)
        # ax2.legend()
        
        # # Adicionar barra de cores para as curvas de nível
        # cbar = plt.colorbar(contour, ax=ax2, shrink=0.8)
        # cbar.set_label('Custo Futuro [R$]', fontsize=10)
        
        # Adicionar informações do reservatório equivalente
        if cortes_estagio and 'equivalente' in cortes_estagio[0]:
            equiv_info = cortes_estagio[0]['equivalente']
            info_text = (
                f"Reservatório Equivalente:\n"
                f"• Volume: {equiv_info['volume_total']:.1f} hm³\n"
                f"• Produtibilidade: {equiv_info['produtibilidade_equivalente']:.3f}\n"
                f"• Altura Queda: {equiv_info['altura_queda_equivalente']:.1f} m\n"
                f"• Cortes: {len(cortes_estagio)}"
            )
            
            # Adicionar caixa de texto no gráfico de superfície
            ax1.text2D(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # # Adicionar no gráfico de contorno também
            # ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=9,
            #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Plot adicional: Mostrar a relação entre volume equivalente e custo futuro
        self.plota_curva_custo_volume_equivalente(cortes_estagio, imes)
        
        return True

    def plota_curva_custo_volume_equivalente(self, cortes_estagio, imes):
        """Plota curva mostrando relação entre volume equivalente e custo futuro"""
        if not cortes_estagio or 'equivalente' not in cortes_estagio[0]:
            return
        
        # Extrair volumes equivalentes e custos
        volumes_equiv = []
        custos = []
        
        for corte in cortes_estagio:
            if 'equivalente' in corte:
                equiv = corte['equivalente']
                volumes_equiv.append(equiv['volume_total'])
                custos.append(corte['custo_medio'])
        
        if not volumes_equiv:
            return
        
        # Ordenar por volume para melhor visualização
        sorted_data = sorted(zip(volumes_equiv, custos))
        volumes_sorted, custos_sorted = zip(*sorted_data)
        
        # Criar figura
        plt.figure(figsize=(10, 6))
        
        # Plotar pontos
        plt.scatter(volumes_sorted, custos_sorted, color='blue', s=50, alpha=0.7, 
                label='Pontos de Discretização')
        
        # Tentar ajustar uma curva suave
        if len(volumes_sorted) > 3:
            # Interpolação suave
            from scipy.interpolate import interp1d
            x_smooth = np.linspace(min(volumes_sorted), max(volumes_sorted), 100)
            try:
                f = interp1d(volumes_sorted, custos_sorted, kind='cubic', fill_value='extrapolate')
                y_smooth = f(x_smooth)
                plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.8, label='Tendência')
            except:
                # Fallback para linear se cubic falhar
                plt.plot(volumes_sorted, custos_sorted, 'r--', linewidth=2, alpha=0.8, label='Tendência')
        
        plt.xlabel('Volume Equivalente [hm³]', fontsize=12)
        plt.ylabel('Custo Futuro Esperado [R$]', fontsize=12)
        plt.title(f'Relação Volume Equivalente vs Custo Futuro\nEstágio {imes}', 
                fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Adicionar informações estatísticas
        if len(volumes_sorted) > 1:
            correlacao = np.corrcoef(volumes_sorted, custos_sorted)[0,1]
            info_text = f'Correlação: {correlacao:.3f}'
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Executa o PDDE
    pdde = PDDE()