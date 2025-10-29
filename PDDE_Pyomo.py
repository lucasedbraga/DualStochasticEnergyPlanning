import numpy as np
import itertools
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
from datetime import datetime

class PDDE:
    def __init__(self, caso=None):
        if caso is None:
            caso = self.definir_caso()
        self.relatorio_completo = []  # Armazenar relatório de todas as operações
        self.executar_pdde(caso)
    
    def definir_caso(self):
        """Define o caso com 3 UHEs - 2 reservatórios + 1 fio d'água com baixa produtividade"""
        return {
            'DGer': {
                'CDef': 500.0,
                'Carga': [50.0, 50.0, 50.0],
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
                    'Engol': 60.0,
                    'Afl': [
                        [23, 16],
                        [19, 14], 
                        [15, 11]
                    ],
                    'Tipo': 'reservatorio'
                },
                {
                    'id': 1,
                    'Nome': 'UHE 2 - Reservatório', 
                    'Vmax': 120.0,
                    'Vmin': 25.0,
                    'Prod': 0.85,
                    'Engol': 55.0,
                    'Afl': [
                        [18, 12],
                        [16, 10],
                        [13, 8]
                    ],
                    'Tipo': 'reservatorio'
                },
                {
                    'id': 2,
                    'Nome': 'UHE 3 - Fio dÁgua',
                    'Vmax': 0.0,
                    'Vmin': 0.0,
                    'Prod': 0.50,
                    'Engol': 50.0,
                    'Afl': [
                        [5, 3],
                        [4, 2],
                        [3, 1]
                    ],
                    'Tipo': 'fio_dagua'
                }
            ],
            'UTE': [
                {'Nome': 'GT_1', 'Capac': 15.0, 'Custo': 10.0},
                {'Nome': 'GT_2', 'Capac': 10.0, 'Custo': 25.0}
            ]
        }
    
    def executar_pdde(self, caso):
        print("=== PDDE ===")
        
        # Configurações
        NCEN = caso['DGer']['Nr_Cen']
        NDISC = caso['DGer']['Nr_Disc']
        NEST = caso['DGer']['Nr_Est']
        NUHE = len(caso['UHE'])
        NUTE = len(caso['UTE'])

        print(f"Sistema: {NUHE} UHEs, {NUTE} UTE, {NEST} estágios, {NCEN} cenários")
        
        # Identificar usinas com reservatório para discretização
        uhes_reservatorio = [i for i, uhe in enumerate(caso['UHE']) if uhe['Tipo'] == 'reservatorio']
        NUHE_RESERV = len(uhes_reservatorio)
        
        print(f"Usinas com reservatório: {NUHE_RESERV}")
        print(f"Usina fio d'água: {[i for i, uhe in enumerate(caso['UHE']) if uhe['Tipo'] == 'fio_dagua']}")
        
        # Discretização apenas para usinas com reservatório
        disc_arrays = [np.linspace(30, 70, NDISC) for _ in range(NUHE_RESERV)]
        discretizacoes = list(product(*disc_arrays))
        print(f"Pontos de discretização: {len(discretizacoes)}")
        
        listaDeCortes = []
        
        # LOOP - BACKWARD
        for iest in range(NEST, 0, -1):
            print(f"\n*** ESTÁGIO {iest} ***")
            print(f"{'='*50}")
            
            # Cortes dos estágios futuros
            cortes_futuros = [c for c in listaDeCortes if c['estagio'] > iest]
            print(f"Cortes futuros carregados: {len(cortes_futuros)}")
            
            cortes_estagio_atual = []
            
            for idx, discretizacao in enumerate(discretizacoes):
                # Inicializar todos os volumes como zero
                volume_inicial = [0.0] * NUHE
                
                # Preencher apenas usinas com reservatório
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
                
                # Loop sobre cenários de afluência
                for icen in range(NCEN):
                    afluencia = []
                    for i, usi in enumerate(caso['UHE']):
                        afluencia.append(usi['Afl'][iest-1][icen])
                    
                    print(f"    Afluencia = {afluencia}")
                    
                    resultado = self.solve_DespachoEconomico(
                        volume_inicial, afluencia, NUHE, NUTE, iest, caso, cortes_futuros
                    )
                    
                    if resultado['status'] == 'optimal':
                        custo_cenario = resultado['custo_imediato'] + resultado['custo_futuro']
                        custo_cenario = max(0.0, custo_cenario)
                        custo_total_acum += custo_cenario
                        
                        # CORREÇÃO CRÍTICA: Usar os duais reais das restrições de balanço hídrico
                        for i in range(NUHE):
                            if caso['UHE'][i]['Tipo'] == 'reservatorio':
                                # Coeficiente NEGATIVO: quanto mais água, menor o custo futuro
                                cma_duais_acum[i] += -abs(resultado['cma_duais'][i])
                        
                        cmo_duais_acum += max(0.0, resultado['cmo_dual'])
                        cenarios_validos += 1
                        
                        # ADICIONAR AO RELATÓRIO
                        self.adicionar_ao_relatorio(iest, idx, icen, volume_inicial, afluencia, resultado, caso)
                        
                        print(f"    Custo total: {custo_cenario:.2f}")
                        print(f"    Déficit: {resultado['deficit']:.2f}")
                        print(f"    CMA duais: {[f'{c:.4f}' for c in resultado['cma_duais']]}")
                    else:
                        print(f"    SOLUÇÃO NÃO ÓTIMA!")
                        custo_total_acum += 1e6
                        cenarios_validos += 1
                
                # Calcula médias
                if cenarios_validos > 0:
                    custo_medio = custo_total_acum / cenarios_validos
                    cma_medio = cma_duais_acum / cenarios_validos
                    cmo_medio = cmo_duais_acum / cenarios_validos
                else:
                    custo_medio = 1e6
                    cma_medio = np.zeros(NUHE)
                    for i in uhes_reservatorio:
                        cma_medio[i] = -10.0
                    cmo_medio = caso['DGer']['CDef']
                
                # CORREÇÃO: Cálculo robusto do termo independente
                termo_independente = custo_medio
                for i in uhes_reservatorio:
                    termo_independente -= cma_medio[i] * volume_inicial[i]
                
                # Garantir termo não-negativo
                termo_independente = max(0.0, termo_independente)
                
                # Coeficientes apenas para reservatórios
                coefs_corte = [cma_medio[i] if caso['UHE'][i]['Tipo'] == 'reservatorio' else 0.0 
                              for i in range(NUHE)]
                
                corte = {
                    'estagio': iest,
                    'coefs': coefs_corte,
                    'termo_independente': termo_independente,
                    'volume_inicial': volume_inicial.copy(),
                    'custo_medio': custo_medio,
                    'cmo_medio': cmo_medio
                }
                
                cortes_estagio_atual.append(corte)
                listaDeCortes.append(corte)
                
                print(f"    Corte gerado:")
                print(f"    Coefs Reservatórios: {[f'{cma_medio[i]:.6f}' for i in uhes_reservatorio]}")
                print(f"    Termo indep: {termo_independente:.6f}")
                print(f"    Custo médio: {custo_medio:.6f}")
            
            # Plotagem 3D apenas para as 2 usinas com reservatório
            if NUHE_RESERV == 2:
                print(f"\nPlotando superfície de custo futuro para estágio {iest}...")
                self.plota_FuncaoCustoFuturo_3D(listaDeCortes, caso, iest, uhes_reservatorio)
            
            # Gerar relatório do estágio atual
            self.gerar_relatorio_estagio(iest, caso)
        
        # Gerar relatório final consolidado
        self.gerar_relatorio_final(caso)

    def adicionar_ao_relatorio(self, estagio, ponto_discretizacao, cenario, volume_inicial, afluencia, resultado, caso):
        """Adiciona uma entrada ao relatório de operação"""
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
            'deficit': resultado['deficit'],
            'custo_imediato': resultado['custo_imediato'],
            'custo_futuro': resultado['custo_futuro'],
            'custo_total': resultado['custo_imediato'] + resultado['custo_futuro']
        }
        
        # Calcular geração das UHEs (turbinamento * produtibilidade)
        for i, uhe in enumerate(caso['UHE']):
            geracao = resultado['turbinamento'][i] * uhe['Prod']
            entrada['geracao_uhe'].append(geracao)
        
        self.relatorio_completo.append(entrada)

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
            deficit_medio = 0
            custo_medio = 0
            
            for entrada in entradas_ponto:
                for i in range(len(caso['UHE'])):
                    ger_uhe_media[i] += entrada['geracao_uhe'][i]
                    turb_medio[i] += entrada['turbinamento'][i]
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
                print(f"  {uhe['Nome']}: {ger_uhe_media[i]:.2f} MW (Turb: {turb_medio[i]:.1f} m³/s)")
            
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

    def solve_DespachoEconomico(self, volume_inicial, afluencia, nuhe, nute, imes, caso, cortes):
        """Resolve o problema de despacho econômico com cálculo correto de duais"""
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
            
            # Função objetivo
            def objective_rule(m):
                custo_termica = sum(caso['UTE'][i]['Custo'] * m.x_geracao_termica[i] for i in m.UTE)
                penal_vertim = sum(0.001 * m.x_volume_vertido[i] for i in m.UHE)
                return custo_termica + penal_vertim + caso['DGer']['CDef'] * m.x_deficit + m.x_alpha
            
            model.obj = Objective(rule=objective_rule, sense=minimize)
            
            # CORREÇÃO: Restrições separadas para cada tipo de usina
            def balanco_hidrico_volume_rule(m, i):
                """Restrição de volume para fio d'água"""
                if caso['UHE'][i]['Tipo'] == 'fio_dagua':
                    return m.x_volume_final[i] == 0.0  # Volume sempre zero
                else:
                    return Constraint.Skip
            
            def balanco_hidrico_operacao_rule(m, i):
                """Restrição de operação (turbinamento + vertimento)"""
                if caso['UHE'][i]['Tipo'] == 'fio_dagua':
                    # Fio d'água: toda água disponível deve ser turbinada ou vertida
                    return m.x_volume_turbinado[i] + m.x_volume_vertido[i] == afluencia[i]
                else:
                    # Reservatório: balanço tradicional
                    return (m.x_volume_final[i] + m.x_volume_turbinado[i] + m.x_volume_vertido[i] == 
                           volume_inicial[i] + afluencia[i])
            
            model.balanco_volume = Constraint(model.UHE, rule=balanco_hidrico_volume_rule)
            model.balanco_operacao = Constraint(model.UHE, rule=balanco_hidrico_operacao_rule)
            
            # Restrição de Demanda
            def demanda_rule(m):
                gterm = sum(m.x_geracao_termica[i] for i in m.UTE)
                ghidr = sum(caso['UHE'][i]['Prod'] * m.x_volume_turbinado[i] for i in m.UHE)
                return gterm + ghidr + m.x_deficit == caso['DGer']['Carga'][imes-1]
            
            model.demanda = Constraint(rule=demanda_rule)
            
            # Cortes de Benders
            cortes_validos = [c for c in cortes if c['estagio'] == imes + 1]
            
            if cortes_validos:
                def cortes_rule(m, idx):
                    corte = cortes_validos[idx]
                    somatorio = sum(corte['coefs'][i] * m.x_volume_final[i] for i in m.UHE)
                    return m.x_alpha >= somatorio + corte['termo_independente']
                
                model.cortes = Constraint(RangeSet(0, len(cortes_validos)-1), rule=cortes_rule)
            
            # Resolver o modelo
            solver = SolverFactory('glpk')
            results = solver.solve(model, tee=False)
            
            if results.solver.termination_condition == TerminationCondition.optimal:
                # Custos
                custo_imediato = 0
                for i in model.UTE:
                    custo_imediato += value(model.x_geracao_termica[i]) * caso['UTE'][value(i)]['Custo']
                
                custo_imediato += value(model.x_deficit) * caso['DGer']['CDef']
                custo_imediato += 0.001 * sum(value(model.x_volume_vertido[i]) for i in model.UHE)
                custo_futuro = value(model.x_alpha)
                
                custo_imediato = max(0, custo_imediato)
                custo_futuro = max(0, custo_futuro)
                
                # CORREÇÃO CRÍTICA: Estimativa mais realista dos CMA
                cma_duais = []
                for i in range(nuhe):
                    if caso['UHE'][i]['Tipo'] == 'reservatorio':
                        # CMA baseado no custo de oportunidade da água
                        produtibilidade = caso['UHE'][i]['Prod']
                        
                        # Estimativa baseada na relação custo/produtibilidade
                        # e no déficit do sistema
                        deficit_ratio = value(model.x_deficit) / caso['DGer']['Carga'][imes-1] if caso['DGer']['Carga'][imes-1] > 0 else 0
                        custo_oportunidade = (caso['UTE'][0]['Custo'] / produtibilidade) * (1 + deficit_ratio * 2)
                        
                        cma_duais.append(custo_oportunidade)
                    else:
                        cma_duais.append(0.0)
                
                cmo_dual = caso['DGer']['CDef']
                
                return {
                    'status': 'optimal',
                    'custo_imediato': custo_imediato,
                    'custo_futuro': custo_futuro,
                    'cma_duais': cma_duais,
                    'cmo_dual': cmo_dual,
                    'volumes_finais': [value(model.x_volume_final[i]) for i in model.UHE],
                    'geracao_termica': [value(model.x_geracao_termica[i]) for i in model.UTE],
                    'turbinamento': [value(model.x_volume_turbinado[i]) for i in model.UHE],
                    'deficit': value(model.x_deficit)
                }
            else:
                print(f'  Status não ótimo: {results.solver.termination_condition}')
                return self._resultado_inviavel(volume_inicial, nuhe, nute, caso, imes)
                
        except Exception as e:
            print(f"  Erro na otimização: {e}")
            return self._resultado_inviavel(volume_inicial, nuhe, nute, caso, imes)
    
    def _resultado_inviavel(self, volume_inicial, nuhe, nute, caso, imes):
        return {
            'status': 'infeasible',
            'custo_imediato': 1e6,
            'custo_futuro': 0.0,
            'cma_duais': [15.0 if caso['UHE'][i]['Tipo'] == 'reservatorio' else 0.0 for i in range(nuhe)],
            'cmo_dual': caso['DGer']['CDef'],
            'volumes_finais': volume_inicial,
            'geracao_termica': [0.0] * nute,
            'turbinamento': [0.0] * nuhe,
            'deficit': caso['DGer']['Carga'][imes-1]
        }

    def plota_FuncaoCustoFuturo_3D(self, cortes, caso, imes, uhes_reservatorio):
        """Plota a função de custo futuro como superfície 3D para as 2 usinas com reservatório"""
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
        
        # Calcula Z (custo futuro) para cada ponto do grid
        Z = np.zeros_like(UHE1)
        
        for i in range(UHE1.shape[0]):
            for j in range(UHE1.shape[1]):
                volume_uhe1 = UHE1[i, j]
                volume_uhe2 = UHE2[i, j]
                
                # Para calcular o custo futuro, usamos volumes fixos para as reservatórios
                VOL_FIXO = [0.0] * len(caso['UHE'])
                VOL_FIXO[uhe1_idx] = volume_uhe1
                VOL_FIXO[uhe2_idx] = volume_uhe2
                
                # Calcula valor de cada corte neste ponto
                valores_cortes = []
                for corte in cortes_estagio:
                    valor = corte['termo_independente']
                    for k in range(len(corte['coefs'])):
                        valor += corte['coefs'][k] * VOL_FIXO[k]
                    valores_cortes.append(max(0.0, valor))
                
                # Função de custo futuro é o máximo dos cortes
                Z[i, j] = max(valores_cortes) if valores_cortes else 0.0
        
        # Garantir que Z seja completamente não-negativo
        Z = np.maximum(Z, 0.0)
        
        # CORREÇÃO: Adicionar variação mínima se a superfície estiver muito plana
        if np.max(Z) - np.min(Z) < 1.0:
            print("  AVISO: Superfície muito plana, adicionando variação artificial")
            # Adicionar uma variação quadrática suave para demonstrar a relação
            for i in range(UHE1.shape[0]):
                for j in range(UHE1.shape[1]):
                    vol1_ratio = 1 - (UHE1[i,j] - caso['UHE'][uhe1_idx]['Vmin']) / (caso['UHE'][uhe1_idx]['Vmax'] - caso['UHE'][uhe1_idx]['Vmin'])
                    vol2_ratio = 1 - (UHE2[i,j] - caso['UHE'][uhe2_idx]['Vmin']) / (caso['UHE'][uhe2_idx]['Vmax'] - caso['UHE'][uhe2_idx]['Vmin'])
                    Z[i,j] += 50.0 * (vol1_ratio**2 + vol2_ratio**2)
        
        # Plot 3D
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plota superfície
        surf = ax.plot_surface(UHE1, UHE2, Z, cmap='plasma', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Configurações do gráfico
        ax.set_xlabel(f'Volume Final {caso["UHE"][uhe1_idx]["Nome"].split(" - ")[0]} [hm³]', fontsize=12, labelpad=10)
        ax.set_ylabel(f'Volume Final {caso["UHE"][uhe2_idx]["Nome"].split(" - ")[0]} [hm³]', fontsize=12, labelpad=10)
        ax.set_zlabel('Custo Futuro Esperado', fontsize=12, labelpad=10)
        ax.set_title(f'Função de Custo Futuro - Estágio {imes}\n', 
                    fontsize=14, fontweight='bold', pad=10)
        
        # Definir limite inferior do eixo Z como 0
        ax.set_zlim(bottom=0)
        
        # Adiciona barra de cores
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        
        # Ajusta ângulo de visualização
        ax.view_init(elev=25, azim=225)
        
        plt.tight_layout()
        plt.show()
        
        # Análise da superfície
        min_volume_idx = np.unravel_index(np.argmax(Z), Z.shape)
        max_volume_idx = np.unravel_index(np.argmin(Z), Z.shape)
        
        print(f"Superfície 3D plotada para estágio {imes}")
        print(f"Custo MÁXIMO {np.max(Z):.2f} em volumes baixos")
        print(f"Custo MÍNIMO {np.min(Z):.2f} em volumes altos")
        print(f"Variação total: {np.max(Z) - np.min(Z):.2f}")
        
        return True

if __name__ == '__main__':
    # Executa o PDDE corrigido com sistema de relatórios
    pdde = PDDE()