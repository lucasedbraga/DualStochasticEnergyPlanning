import numpy as np
import itertools
from itertools import product
import matplotlib.pyplot as plt
import pulp

class PDDE:
    def __init__(self):
        pass

    def solve_DespachoEconomico(self, volume_inicial, afluencia, nuhe, nute, imes, caso, cortes):
        # Criar modelo PuLP
        model = pulp.LpProblem("Despacho_Economico", pulp.LpMinimize)
        
        # Variáveis de decisão
        x_volume_final = [pulp.LpVariable(f'x_volume_final_{i}', 
                                        lowBound=caso['UHE'][i]['Vmin'],
                                        upBound=caso['UHE'][i]['Vmax']) 
                         for i in range(nuhe)]
        
        x_volume_turbinado = [pulp.LpVariable(f'x_volume_turbinado_{i}', 
                                            lowBound=0,
                                            upBound=caso['UHE'][i]['Engol'])
                             for i in range(nuhe)]
        
        x_volume_vertido = [pulp.LpVariable(f'x_volume_vertido_{i}', lowBound=0) 
                           for i in range(nuhe)]
        
        x_geracao_termica = [pulp.LpVariable(f'x_geracao_termica_{i}', 
                                           lowBound=0,
                                           upBound=caso['UTE'][i]['Capac'])
                            for i in range(nute)]
        
        x_deficit = pulp.LpVariable('x_deficit', lowBound=0)
        x_alpha = pulp.LpVariable('x_alpha', lowBound=0)
        
        # Função objetivo
        custo_termica = pulp.lpSum([caso['UTE'][i]['Custo'] * x_geracao_termica[i] for i in range(nute)])
        penal_vertim = pulp.lpSum([0.001 * x_volume_vertido[i] for i in range(nuhe)])
        
        model += (custo_termica + penal_vertim + 
                 caso['DGer']['CDef'] * x_deficit + x_alpha)
        
        # Restrições de Balanço Hídrico
        cma_constraints = []
        for i in range(nuhe):
            constraint_name = f"BH_Usina_{i}"
            constraint = (x_volume_final[i] + x_volume_turbinado[i] + x_volume_vertido[i] == 
                         volume_inicial[i] + afluencia[i])
            cma_constraints.append(model.addConstraint(constraint, constraint_name))
        
        # Restrição de Demanda
        gterm = pulp.lpSum([x_geracao_termica[i] for i in range(nute)])
        ghidr = pulp.lpSum([caso['UHE'][i]['Prod'] * x_volume_turbinado[i] for i in range(nuhe)])
        
        cmo_constraint = model.addConstraint(
            gterm + ghidr + x_deficit == caso['DGer']['Carga'][imes-1], 
            "Demanda"
        )
        
        # Cortes de Benders
        for idx, corte in enumerate(cortes):
            if corte['estagio'] == imes + 1:
                somatorio = pulp.lpSum([corte['coefs'][i] * x_volume_final[i] for i in range(nuhe)])
                model.addConstraint(x_alpha >= somatorio + corte['termo_independente'], f"Corte_{idx}")
        
        # Resolver o modelo
        model.solve(pulp.GLPK(msg=0))
        
        # Processar resultados
        if model.status == pulp.LpStatusOptimal:
            # Custos
            custo_imediato = 0
            for i in range(nute):
                custo_imediato += x_geracao_termica[i].varValue * caso['UTE'][i]['Custo']
            
            custo_imediato += x_deficit.varValue * caso['DGer']['CDef']
            custo_imediato += 0.001 * sum(x_volume_vertido[i].varValue for i in range(nuhe))
            custo_futuro = x_alpha.varValue
            
            # OBTER DUAIS COM TRATAMENTO DE ERRO
            cma_duais = []
            for i in range(nuhe):
                constraint_name = f"BH_Usina_{i}"
                dual_val = model.constraints[constraint_name].pi
                
                # Tratar casos onde dual é None
                if dual_val is None:
                    print(f"    AVISO: Dual None para restrição {constraint_name}. Usando valor padrão.")
                    cma_duais.append(0.0)  # Valor padrão
                else:
                    cma_duais.append(-dual_val)  # Inverter sinal
            
            # Obter dual da demanda
            cmo_dual = model.constraints["Demanda"].pi
            if cmo_dual is None:
                print("    AVISO: Dual None para restrição Demanda. Usando valor padrão.")
                cmo_dual = caso['DGer']['CDef']  # Valor padrão
            
            # Debug
            print(f"    Ger UHE: {[round(x_volume_turbinado[i].varValue, 2) for i in range(nuhe)]}")
            print(f"    Ger UTE: {[round(x_geracao_termica[i].varValue, 2) for i in range(nute)]}")
            print(f"    VolFinal UHE: {[round(x_volume_final[i].varValue, 2) for i in range(nuhe)]}")
            print(f"    Déficit: {round(x_deficit.varValue, 2)}")
            print(f"    CMA Duais: {[f'{c:.6f}' for c in cma_duais]}")
            print(f"    CMO Dual: {cmo_dual:.6f}")
            
            return {
                'status': 'optimal',
                'custo_imediato': custo_imediato,
                'custo_futuro': custo_futuro,
                'cma_duais': cma_duais,
                'cmo_dual': cmo_dual,
                'volumes_finais': [x_volume_final[i].varValue for i in range(nuhe)],
                'geracao_termica': [x_geracao_termica[i].varValue for i in range(nute)],
                'turbinamento': [x_volume_turbinado[i].varValue for i in range(nuhe)],
                'deficit': x_deficit.varValue
            }
        else:
            print(f'  Status não ótimo: {pulp.LpStatus[model.status]}')
            return {
                'status': 'infeasible',
                'custo_imediato': 1e6,
                'custo_futuro': 0,
                'cma_duais': [10.0] * nuhe,  # Valores padrão
                'cmo_dual': caso['DGer']['CDef'],
                'volumes_finais': volume_inicial,
                'geracao_termica': [0] * nute,
                'turbinamento': [0] * nuhe,
                'deficit': caso['DGer']['Carga'][imes-1]
            }

    def solve_FuncaoCustoFuturo(self, cortes, nuhe, imes, VOL_FIXO):
        """Calcula o valor da função de custo futuro para volumes finais fixos"""
        cortes_futuros = [c for c in cortes if c['estagio'] == imes + 1]
        
        if not cortes_futuros:
            return 0.0
            
        model = pulp.LpProblem("Funcao_Custo_Futuro", pulp.LpMinimize)
        alpha = pulp.LpVariable('alpha', lowBound=0)
        
        # Cortes como restrições
        for idx, corte in enumerate(cortes_futuros):
            somatorio = sum(corte['coefs'][i] * VOL_FIXO[i] for i in range(nuhe))
            model += alpha >= somatorio + corte['termo_independente']
        
        model += alpha
        model.solve(pulp.GLPK(msg=0))
        
        if model.status == pulp.LpStatusOptimal:
            return alpha.varValue
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

def main():
    # Configuração do caso
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
    
    print("=== INICIANDO PDDE COM PULP - DUAIS COM TRATAMENTO ===")
       
    # Configurações
    NCEN = caso['DGer']['Nr_Cen']
    NDISC = caso['DGer']['Nr_Disc']
    NEST = caso['DGer']['Nr_Est']
    NUHE = len(caso['UHE'])
    NUTE = len(caso['UTE'])

    print(f"Sistema: {NUHE} UHEs, {NUTE} UTE, {NEST} estágios, {NCEN} cenários")
    
    # Inicializa PDDE
    pdde = PDDE()

    # Discretização
    disc_arrays = [np.linspace(30, 70, NDISC) for _ in range(NUHE)]
    discretizacoes = list(product(*disc_arrays))
    print(f"Número de pontos de discretização: {len(discretizacoes)}")
    
    # Lista global para armazenar todos os cortes
    todos_cortes = []
    
    # LOOP PRINCIPAL - BACKWARD
    for iest in range(NEST, 0, -1):
        print(f"\n*** ESTÁGIO {iest} ***")
        print(f"{'='*50}")
        
        # Cortes dos estágios futuros
        cortes_futuros = [c for c in todos_cortes if c['estagio'] > iest]
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
                
                resultado = pdde.solve_DespachoEconomico(
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
            todos_cortes.append(corte)
            
            print(f"    Corte gerado:")
            print(f"    Coefs: {[f'{c:.6f}' for c in cma_medio]}")
            print(f"    Termo indep: {termo_independente:.6f}")
            print(f"    CMO médio: {cmo_medio:.6f}")
        
        # Plotagem
        print(f"\nPlotando funções de custo futuro para estágio {iest}...")
        for uhe_idx in range(NUHE):
            pdde.plota_FuncaoCustoFuturo(todos_cortes, caso, iest, uhe_idx)

if __name__ == '__main__':
    main()