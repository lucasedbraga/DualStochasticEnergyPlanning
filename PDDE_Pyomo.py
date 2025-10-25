import numpy as np
import itertools
from itertools import product
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory

class PDDE:
    def __init__(self, caso=None):
    
        if caso is None:
            caso = {
                'DGer': {
                    'CDef': 500.0,
                    'Carga': [50.0, 50.0, 50.0],
                    'Nr_Disc': 3,
                    'Nr_Est': 3,
                    'Nr_Cen': 2
                },
                'UHE': [{
                    'Nome': 'UHE DO MARCATO',
                    'Vmax': 100.0,
                    'Vmin': 20.0,
                    'Prod': 0.95,
                    'Engol': 60.0,
                    'Afl': [[23, 16], [19, 14], [15, 11]]
                }],
                'UTE': [
                    {'Nome': 'GT_1', 'Capac': 15.0, 'Custo': 10.0},
                    {'Nome': 'GT_2', 'Capac': 10.0, 'Custo': 25.0}
                ]
            }
    
        print("=== INICIANDO PDDE COM PYOMO ===")
        
        # Configurações
        NCEN = caso['DGer']['Nr_Cen']
        NDISC = caso['DGer']['Nr_Disc']
        NEST = caso['DGer']['Nr_Est']
        NUHE = len(caso['UHE'])
        NUTE = len(caso['UTE'])

        print(f"Sistema: {NUHE} UHEs, {NUTE} UTE, {NEST} estágios, {NCEN} cenários")
        
        # Discretização
        disc_arrays = [np.linspace(30, 70, NDISC) for _ in range(NUHE)]
        discretizacoes = list(product(*disc_arrays))
        print(f"Número de pontos de discretização: {len(discretizacoes)}")
        
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
                volume_inicial = []
                for i, usi in enumerate(caso['UHE']):
                    vol = usi['Vmin'] + ((usi['Vmax'] - usi['Vmin']) * discretizacao[i] / 100)
                    volume_inicial.append(vol)
                
                print(f"\n  Ponto {idx+1}/{len(discretizacoes)}")
                print(f"    Volume Inicial = {[f'{v:.1f}' for v in volume_inicial]}")
                
                custo_total_acum = 0.0
                cma_duais_acum = np.zeros(NUHE)
                cmo_duais_acum = 0.0
                
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
                        custo_total_acum += custo_cenario
                        
                        for i in range(NUHE):
                            cma_duais_acum[i] += resultado['cma_duais'][i]
                        cmo_duais_acum += resultado['cmo_dual']
                        
                        print(f"    Custo total: {custo_cenario:.2f}")
                        print(f"    Déficit: {resultado['deficit']:.2f}")
                        print(f"    CMA duais: {[f'{c:.6f}' for c in resultado['cma_duais']]}")
                        print(f"    CMO dual: {resultado['cmo_dual']:.6f}")
                    else:
                        print(f"    SOLUÇÃO NÃO ÓTIMA!")
                        custo_total_acum += 1e6
                
                # Calcula médias
                custo_medio = custo_total_acum / NCEN
                cma_medio = cma_duais_acum / NCEN
                cmo_medio = cmo_duais_acum / NCEN
                
                # Gera corte
                termo_independente = custo_medio - sum(cma_medio[i] * volume_inicial[i] for i in range(NUHE))
                
                corte = {
                    'estagio': iest,
                    'coefs': cma_medio.tolist(),
                    'termo_independente': termo_independente,
                    'volume_inicial': volume_inicial.copy(),
                    'custo_medio': custo_medio,
                    'cmo_medio': cmo_medio
                }
                
                cortes_estagio_atual.append(corte)
                listaDeCortes.append(corte)
                
                print(f"    Corte gerado:")
                print(f"    Coefs: {[f'{c:.6f}' for c in cma_medio]}")
                print(f"    Termo indep: {termo_independente:.6f}")
                print(f"    CMO médio: {cmo_medio:.6f}")
            
            # Plotagem
            print(f"\nPlotando funções de custo futuro para estágio {iest}...")
            for uhe_idx in range(NUHE):
                self.plota_FuncaoCustoFuturo(listaDeCortes, caso, iest, uhe_idx)

    def solve_DespachoEconomico(self, volume_inicial, afluencia, nuhe, nute, imes, caso, cortes):
        # Criar modelo Pyomo
        model = ConcreteModel()
        model.dual = Suffix(direction=Suffix.IMPORT)
        
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
        
        # Restrições de Balanço Hídrico
        def balanco_hidrico_rule(m, i):
            return (m.x_volume_final[i] + m.x_volume_turbinado[i] + m.x_volume_vertido[i] == 
                    volume_inicial[i] + afluencia[i])
        
        model.balanco_hidrico = Constraint(model.UHE, rule=balanco_hidrico_rule)
        
        # Restrição de Demanda
        def demanda_rule(m):
            gterm = sum(m.x_geracao_termica[i] for i in m.UTE)
            ghidr = sum(caso['UHE'][i]['Prod'] * m.x_volume_turbinado[i] for i in m.UHE)
            return gterm + ghidr + m.x_deficit == caso['DGer']['Carga'][imes-1]
        
        model.demanda = Constraint(rule=demanda_rule)
        
        # Cortes de Benders
        def cortes_rule(m, corte_idx):
            corte = cortes[corte_idx]
            if corte['estagio'] == imes + 1:
                somatorio = sum(corte['coefs'][i] * m.x_volume_final[i] for i in m.UHE)
                return m.x_alpha >= somatorio + corte['termo_independente']
            else:
                return Constraint.Skip
        
        model.cortes = Constraint(RangeSet(0, len(cortes)-1), rule=cortes_rule)
        
        # Resolver o modelo
        solver = SolverFactory('glpk')
        results = solver.solve(model, tee=False)
        
        # Processar resultados
        if results.solver.termination_condition == TerminationCondition.optimal:
            # Custos
            custo_imediato = 0
            for i in model.UTE:
                custo_imediato += value(model.x_geracao_termica[i]) * caso['UTE'][i]['Custo']
            
            custo_imediato += value(model.x_deficit) * caso['DGer']['CDef']
            custo_imediato += 0.001 * sum(value(model.x_volume_vertido[i]) for i in model.UHE)
            custo_futuro = value(model.x_alpha)
            
            cma_duais = []
            for i in model.UHE:
                dual_val = model.dual[model.balanco_hidrico[i]]
                cma_duais.append(dual_val)
            
            cmo_dual = model.dual[model.demanda]
            
            # Debug
            print(f"    Ger UHE: {[value(model.x_volume_turbinado[i]) for i in model.UHE]}")
            print(f"    Ger UTE: {[value(model.x_geracao_termica[i]) for i in model.UTE]}")
            print(f"    VolFinal UHE: {[value(model.x_volume_final[i]) for i in model.UHE]}")
            print(f"    Déficit: {value(model.x_deficit):.2f}")
            print(f"    CMA Duais: {[f'{c:.6f}' for c in cma_duais]}")
            print(f"    CMO Dual: {cmo_dual:.6f}")
            
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
            

    def solve_FuncaoCustoFuturo(self, cortes, nuhe, imes, VOL_FIXO):
        """Calcula o valor da função de custo futuro para volumes finais fixos"""
        cortes_futuros = [c for c in cortes if c['estagio'] == imes + 1]
        
        if not cortes_futuros:
            return 0.0
            
        model = ConcreteModel()
        model.alpha = Var(bounds=(0, None))
        
        # Cortes como restrições
        def cortes_rule(m, idx):
            corte = cortes_futuros[idx]
            somatorio = sum(corte['coefs'][i] * VOL_FIXO[i] for i in range(nuhe))
            return m.alpha >= somatorio + corte['termo_independente']
        
        model.cortes = Constraint(RangeSet(0, len(cortes_futuros)-1), rule=cortes_rule)
        model.obj = Objective(expr=model.alpha, sense=minimize)
        
        solver = SolverFactory('glpk')
        results = solver.solve(model, tee=False)
        
        if results.solver.termination_condition == TerminationCondition.optimal:
            return value(model.alpha)
        else:
            return 0.0

    def plota_FuncaoCustoFuturo(self, cortes, caso, imes, uhe_index=0):
        """Plota a função de custo futuro para 1 UHE específica"""
        nuhe = len(caso['UHE'])
        
        # Gera range de volumes para plotar
        x_min = caso['UHE'][uhe_index]['Vmin']
        x_max = caso['UHE'][uhe_index]['Vmax']
        x = np.linspace(x_min, x_max, 100)
        
        # Para cada ponto x, calcula o máximo entre todos os cortes
        y = []
        for volume in x:
            VOL_FIXO = []
            for i in range(nuhe):
                if i == uhe_index:
                    VOL_FIXO.append(volume)
                else:
                    vol_medio = (caso['UHE'][i]['Vmin'] + caso['UHE'][i]['Vmax']) / 2
                    VOL_FIXO.append(vol_medio)
            
            # Calcula valor de cada corte neste ponto
            valores_cortes = []
            for corte in cortes:
                if corte['estagio'] == imes:
                    valor = corte['termo_independente']
                    for i in range(nuhe):
                        valor += corte['coefs'][i] * VOL_FIXO[i]
                    valores_cortes.append(valor)
            
            # Função de custo futuro é o máximo dos cortes
            y.append(max(valores_cortes) if valores_cortes else 0)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Plota a função de custo futuro (envelope)
        plt.plot(x, y, 'b-', linewidth=3, label='Função Custo Futuro')
        
        # Plota cada corte individualmente como reta
        for idx, corte in enumerate(cortes):
            if corte['estagio'] == imes:
                y_corte = []
                for volume in x:
                    valor = corte['termo_independente']
                    for i in range(nuhe):
                        if i == uhe_index:
                            valor += corte['coefs'][i] * volume
                        else:
                            vol_medio = (caso['UHE'][i]['Vmin'] + caso['UHE'][i]['Vmax']) / 2
                            valor += corte['coefs'][i] * vol_medio
                    y_corte.append(valor)
                
                plt.plot(x, y_corte, 'r--', alpha=0.5, linewidth=1, 
                        label=f'Corte {idx+1}' if idx < 5 else "")
        
        plt.title(f'Função de Custo Futuro - Estágio {imes} - UHE{uhe_index+1}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel(f'Volume Final UHE{uhe_index+1} [hm³]', fontsize=14)
        plt.ylabel('Custo Futuro Esperado', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
        
        return True

if __name__ == '__main__':
    PDDE()