import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter
from matplotlib.patches import Arc
import numpy as np
from PIL import Image
import os

# --- Configuração da Página ---
st.set_page_config(page_title="Refração e Retrorrefletividade", layout="wide")

# --- FUNÇÕES AUXILIARES (Definidas no início para evitar erros) ---
def microns_formatter(x, pos):
    """Converte unidades do gráfico para mícrons na legenda."""
    return f"{int(x * 500)}"

# Título
st.markdown("### Refração e Retrorrefletividade - Análise de Geometria (15m / 30m)")

# --- Definição do Layout ---
# Container Superior para o Gráfico Grande
container_top = st.container()

# Container Inferior dividido em duas colunas
# col_left: Controles e Outputs
# col_right: Gráfico de Zoom
col_left, col_right = st.columns([1, 1])

# --- Coluna Esquerda: Controles ---
with col_left:
    st.subheader("Parâmetros de Entrada")
    
    # Parâmetros Físicos
    val_R = st.slider("Raio (µm)", min_value=50, max_value=1000, value=500, step=10)
    val_A_percent = st.slider("Ancoragem (%)", min_value=50.0, max_value=60.0, value=50.0, step=0.1)
    val_beta = st.slider("Inclinação da Tinta β (graus)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    val_k = st.slider("Índice de Refração (k)", min_value=1.0, max_value=4.0, value=1.9, step=0.01)
    
    # Seletor de Geometria
    geo_option = st.radio("Geometria", options=[0, 1], format_func=lambda x: "15 metros" if x == 0 else "30 metros", horizontal=True)
    
    st.markdown("---")
    st.markdown("**Raio 1 (Roxo)**")
    val_alpha_1 = st.slider("Ponto de contato P (graus)", min_value=0.0, max_value=90.0, value=0.0, step=0.1)
    
    st.markdown("**Raio 2 (Vermelho)**")
    val_alpha_2 = st.slider("Ponto de contato A (graus)", min_value=0.0, max_value=90.0, value=10.0, step=0.1)

# --- Lógica de Cálculo (Backend) ---

# Conversão de unidades e preparação
R_micrometers = float(val_R)
R = R_micrometers / 500.0 # Normalização para o plot
A = val_A_percent / 100.0
beta_radians = np.deg2rad(val_beta)
k = val_k

# Centro da esfera
dist_center_to_line = R * abs(2 * A - 1)
if A >= 0.5:
    direction_vector = np.array([np.sin(beta_radians), -np.cos(beta_radians)])
else:
    direction_vector = np.array([-np.sin(beta_radians), np.cos(beta_radians)])
x_center, y_center = dist_center_to_line * direction_vector
tan_beta = np.tan(beta_radians)

# Verificação de existência da tinta (Q)
A_quad = 1 + tan_beta**2
B_quad = -2 * (x_center + y_center * tan_beta)
C_quad = x_center**2 + y_center**2 - R**2
delta = B_quad**2 - 4 * A_quad * C_quad
Q_exists = delta >= 0

# Configuração Dinâmica da Geometria (Lógica Crítica)
if geo_option == 0: # CASO 1: 15m
    # Correção trigonométrica exata para 0.65m em 15m
    angle_inc_rad_calc = np.arctan(0.65 / 15.0)
    angle_inc_deg = -np.degrees(angle_inc_rad_calc) # Aprox -2.48
    
    X_START_PLOT = -30000.0 # Representa -15m na escala (2000u = 1m)
    OBSERVATION_HEIGHTS = [1300, 2400] # 0.65m e 1.20m
    
    wide_x_lim = (-30000, 10)
    wide_x_ticks = [-30000, 0]
    wide_x_labels = ["-15m", "0m"]
    
    wide_y_lim = (-100, 3200) # Até 1.6m
    wide_y_ticks_values = [1300, 2400]
    wide_y_ticks_labels = ["0.65m", "1.20m"]
    
else: # CASO 2: 30m
    angle_inc_deg = -1.24
    X_START_PLOT = -60000.0 # Representa -30m
    OBSERVATION_HEIGHTS = [1300, 2400] # 0.65m e 1.20m
    
    wide_x_lim = (-60000, 10)
    wide_x_ticks = [-60000, 0]
    wide_x_labels = ["-30m", "0m"]
    
    wide_y_lim = (-100, 3200) # Até 1.6m
    wide_y_ticks_values = [1300, 2400]
    wide_y_ticks_labels = ["0.65m", "1.20m"]

angle_inc_rad = np.deg2rad(angle_inc_deg)
I_vec = np.array([np.cos(angle_inc_rad), np.sin(angle_inc_rad)])

# Encontrar ponto Q (interseção tinta/esfera para referência angular)
ref_angle_Q = 0.0
if Q_exists:
    x1 = (-B_quad + np.sqrt(delta)) / (2 * A_quad)
    x2 = (-B_quad - np.sqrt(delta)) / (2 * A_quad)
    x_Q = min(x1, x2)
    y_Q = tan_beta * x_Q
    ref_angle_Q = np.arctan2(y_Q - y_center, x_Q - x_center)

# Função de Traçado de Raios (Lógica Física)
def calculate_ray(alpha_deg):
    res = {'exists': False, 'logs': {}}
    if not Q_exists: return res
    
    # 1. Ponto de contato P
    angle_P = ref_angle_Q - np.deg2rad(alpha_deg)
    x_P = x_center + R * np.cos(angle_P)
    y_P = y_center + R * np.sin(angle_P)
    res['hit'] = (x_P, y_P)
    res['angle_P_rad'] = angle_P
    
    # 2. Ponto de origem (Farol)
    x_start = X_START_PLOT
    if I_vec[0] != 0:
        y_start = y_P + (I_vec[1] / I_vec[0]) * (x_start - x_P)
    else:
        y_start = y_P
    res['start'] = (x_start, y_start)
    
    # 3. Refração Entrada (Lei de Snell Vetorial)
    x_n, y_n = (x_P - x_center)/R, (y_P - y_center)/R
    N_vec = np.array([x_n, y_n])
    cos_z = -np.dot(N_vec, I_vec)
    
    if cos_z < 0: return res # Raio vindo de trás da normal
    
    res['exists'] = True
    z_rad = np.arccos(np.clip(cos_z, -1.0, 1.0))
    arg_arcsin = np.sin(z_rad) / k
    
    if abs(arg_arcsin) > 1: return res # Reflexão total externa (não entra)
    
    j_rad = np.arcsin(arg_arcsin)
    
    # Transmissão Fresnel
    n1, n2 = 1.0, k
    cos_t1, cos_t2 = np.cos(z_rad), np.cos(j_rad)
    rs = ((n1*cos_t1 - n2*cos_t2)/(n1*cos_t1 + n2*cos_t2))**2
    rp = ((n1*cos_t2 - n2*cos_t1)/(n1*cos_t2 + n2*cos_t1))**2
    T_1 = 1 - (rs + rp)/2
    res['T1'] = T_1
    
    # Vetor Refratado T
    c = cos_z
    radicand = 1.0 - (1.0/k)**2 * (1.0 - c**2)
    T_vec = (1.0/k) * I_vec + ((1.0/k) * c - np.sqrt(np.clip(radicand, 0, None))) * N_vec
    
    # Interseção Interna W (fundo da esfera)
    P_vec = np.array([x_P, y_P])
    O_vec = np.array([x_center, y_center])
    t_val = -2 * np.dot(T_vec, (P_vec - O_vec))
    W_vec = P_vec + t_val * T_vec
    res['W'] = (W_vec[0], W_vec[1])
    
    # Reflexão em W -> Ponto L (ou C)
    vec_normal_W = (W_vec - O_vec) / R
    vec_inc_W = T_vec
    vec_refletido = vec_inc_W - 2 * np.dot(vec_inc_W, vec_normal_W) * vec_normal_W
    L_vec = W_vec + t_val * vec_refletido
    res['exit_point'] = (L_vec[0], L_vec[1])
    res['angle_exit_rad'] = np.arctan2(L_vec[1] - y_center, L_vec[0] - x_center)
    
    # Refração Saída
    vec_normal_L = (L_vec - O_vec) / R
    vec_inc_L = vec_refletido
    arg_sin_out = k * np.sin(j_rad) # Simetria
    
    if abs(arg_sin_out) < 1.0:
        # Transmissão Fresnel Saída
        n1_out, n2_out = k, 1.0
        theta1_out, theta2_out = j_rad, np.arcsin(arg_sin_out)
        cos_t1_out, cos_t2_out = np.cos(theta1_out), np.cos(theta2_out)
        rs_out = ((n1_out*cos_t1_out - n2_out*cos_t2_out)/(n1_out*cos_t1_out + n2_out*cos_t2_out))**2
        rp_out = ((n1_out*cos_t2_out - n2_out*cos_t1_out)/(n1_out*cos_t2_out + n2_out*cos_t1_out))**2
        T_out = 1 - (rs_out + rp_out)/2
        res['T_total'] = T_1 * T_out
        
        # Vetor Saída Final
        c1_exit = np.dot(vec_inc_L, vec_normal_L)
        radicand_out = 1.0 - k**2 * (1.0 - c1_exit**2)
        vec_saida = k * vec_inc_L - (k * c1_exit - np.sqrt(np.clip(radicand_out, 0, None))) * vec_normal_L
        
        vx, vy = vec_saida[0], vec_saida[1]
        
        # Onde atinge o plano do observador (X_START_PLOT)
        if vx < 0:
            t_M = (X_START_PLOT - L_vec[0]) / vx
            y_M = L_vec[1] + t_M * vy
            res['end'] = (X_START_PLOT, y_M)
            res['full_path'] = True
            
            # Ângulos Finais
            vec_chao = np.array([-1.0, 0.0])
            vec_saida_unit = vec_saida / np.linalg.norm(vec_saida)
            cos_theta = np.dot(vec_saida_unit, vec_chao)
            res['ang_ground'] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            
            vec_inc_invertido = -I_vec
            cos_obs = np.dot(vec_saida_unit, vec_inc_invertido)
            res['ang_obs'] = np.degrees(np.arccos(np.clip(cos_obs, -1.0, 1.0)))
            
    return res

# --- Executar Cálculos ---
r1_data = calculate_ray(val_alpha_1)
r2_data = calculate_ray(val_alpha_2)

# --- Geração dos Outputs de Texto (Coluna Esquerda) ---
with col_left:
    st.subheader("Resultados")
    
    # Raio 1
    st.markdown("**Dados Raio 1 (Roxo):**")
    if r1_data.get('full_path'):
        st.caption(f"Luz transmitida P->W: {r1_data['T1']*100:.1f}%")
        st.caption(f"Luz transmitida L->Ext: {r1_data['T_total']*100:.1f}% (Total)")
        st.caption(f"Ângulo de saída c/ chão: {r1_data['ang_ground']:.2f}º")
        st.caption(f"Ângulo de observação: {r1_data['ang_obs']:.2f}º")
    else:
        st.caption("Raio 1 não completa o trajeto.")
        
    st.markdown("**Dados Raio 2 (Vermelho):**")
    if r2_data.get('full_path'):
        st.caption(f"Luz transmitida A->B: {r2_data['T1']*100:.1f}%")
        st.caption(f"Luz transmitida C->Ext: {r2_data['T_total']*100:.1f}% (Total)")
        st.caption(f"Ângulo de saída c/ chão: {r2_data['ang_ground']:.2f}º")
        st.caption(f"Ângulo de observação: {r2_data['ang_obs']:.2f}º")
    else:
        st.caption("Raio 2 não completa o trajeto.")
        
    st.markdown("---")
    # Cálculos Combinados (Diferenciais)
    if r1_data.get('exists') and r2_data.get('exists'):
        # Incidência
        angle_inc_diff_rad = abs(r1_data['angle_P_rad'] - r2_data['angle_P_rad'])
        if angle_inc_diff_rad > np.pi: angle_inc_diff_rad = 2*np.pi - angle_inc_diff_rad
        arc_inc = R_micrometers * angle_inc_diff_rad
        st.write(f"**Ângulo/Arco Incidência:** {np.degrees(angle_inc_diff_rad):.2f}º / {arc_inc:.1f}µm")
        
        # Saída
        if 'angle_exit_rad' in r1_data and 'angle_exit_rad' in r2_data:
            angle_out_diff_rad = abs(r1_data['angle_exit_rad'] - r2_data['angle_exit_rad'])
            if angle_out_diff_rad > np.pi: angle_out_diff_rad = 2*np.pi - angle_out_diff_rad
            arc_out = R_micrometers * angle_out_diff_rad
            st.write(f"**Ângulo/Arco Saída:** {np.degrees(angle_out_diff_rad):.2f}º / {arc_out:.1f}µm")
    
    # Alturas e Intensidade
    if r1_data.get('full_path') and r2_data.get('full_path'):
        y1 = r1_data['end'][1] / 2000.0 # Converter u para metros
        y2 = r2_data['end'][1] / 2000.0
        gamma = min(y1, y2)
        lam = max(y1, y2)
        st.write(f"**Altura min/max saída:** {gamma:.3f}m - {lam:.3f}m")
        
        # Cálculo do Alpha Dinâmico
        spread_m = abs(gamma - lam)
        dynamic_alpha = max(0.02, 0.9 * np.exp(-2.0 * spread_m))
    else:
        dynamic_alpha = 0.0 # Sem preenchimento se não houver raios

# --- Geração dos Gráficos (Plotting) ---

# Criar Figura com 2 subplots (Wide e Zoom)
# Ajuste do tamanho para caber bem na tela do Streamlit
fig, (ax_wide, ax_zoom) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1]})

# --- 1. Gráfico WIDE (Superior) ---
# Título
ax_wide.set_title("Farol: 0.65m; Visão do motorista: 1.2m", fontsize=12)
ax_wide.set_xlim(wide_x_lim)
ax_wide.set_ylim(wide_y_lim)

# Linhas de Referência
for h in OBSERVATION_HEIGHTS:
    ax_wide.axhline(y=h, color='black', linestyle=':', linewidth=1.5, alpha=0.6)

# Ticks e Labels
ax_wide.set_yticks(wide_y_ticks_values)
ax_wide.set_yticklabels(wide_y_ticks_labels)
ax_wide.tick_params(axis='y', labelsize=9)
ax_wide.set_xticks(wide_x_ticks)
ax_wide.set_xticklabels(wide_x_labels)
ax_wide.set_xlabel('Distância (m)', fontsize=10)

# Imagem do Carro
try:
    pil_image = Image.open("carro.png")
    car_width_wide = abs(X_START_PLOT) * 0.15 
    car_height_wide = car_width_wide * (pil_image.height / pil_image.width)
    farol_x_ratio = 0.75 
    farol_y_ratio = 0.5
    
    # Calcular posição do carro baseado no raio de incidência (usando R1 como ref)
    # Precisamos recalcular onde o raio do farol começa visualmente
    if r1_data.get('start'):
        y_ref_car = r1_data['start'][1]
    else:
        y_ref_car = 0 # Fallback
        
    car_x_wide = X_START_PLOT - (car_width_wide * farol_x_ratio)
    car_y_wide = y_ref_car - (car_height_wide * (1 - farol_y_ratio))
    
    ax_wide.imshow(pil_image, extent=[car_x_wide, car_x_wide + car_width_wide, 
                                      car_y_wide, car_y_wide + car_height_wide],
                   aspect='auto', zorder=-1)
except:
    ax_wide.text(X_START_PLOT, 0, "Carro.png não encontrada", fontsize=8, color='red')

# --- 2. Gráfico ZOOM (Inferior) ---
ax_zoom.set_title("Visão Zoom (Detalhe Esfera)", fontsize=12)
ax_zoom.set_xlim(-3, 3)
ax_zoom.set_ylim(-3, 3)
# O uso de FuncFormatter requer a função definida no início do script
ax_zoom.xaxis.set_major_formatter(FuncFormatter(microns_formatter))
ax_zoom.yaxis.set_major_formatter(FuncFormatter(microns_formatter))
ax_zoom.set_xlabel('mícrons', fontsize=10)
ax_zoom.set_ylabel('mícrons', fontsize=10)
ax_zoom.set_aspect('equal', adjustable='box')

# Desenhar Esfera no Zoom
theta_circle = np.linspace(0, 2 * np.pi, 400)
ax_zoom.plot(x_center + R * np.cos(theta_circle), y_center + R * np.sin(theta_circle), color='black')
ax_zoom.plot(x_center, y_center, 'ko', markersize=5)
ax_zoom.text(x_center + 0.08, y_center - 0.08, 'O', fontsize=12, color='black')

# Desenhar linha de referência inclinada (tinta)
x_line_ref = np.linspace(X_START_PLOT, 10, 2)
ax_wide.plot(x_line_ref, x_line_ref * tan_beta, color='black', linestyle='-', linewidth=1.0)
ax_zoom.plot(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10) * tan_beta, color='black', linestyle='-', linewidth=1.5)

# --- PLOTAGEM DOS RAIOS (WIDE e ZOOM) ---
def plot_ray_on_axes(ray_data, color_ray):
    if not ray_data.get('exists'): return
    
    hit = ray_data['hit']
    start = ray_data['start']
    
    # 1. Incidente (Marrom no Wide, Cor original no Zoom)
    c_inc_wide = 'brown'
    
    # Wide
    ax_wide.plot([start[0], hit[0]], [start[1], hit[1]], color=c_inc_wide, linewidth=1.5)
    # Seta Wide
    mid_x_w = start[0] + abs(start[0])*0.1
    # Equação da reta para achar y do meio
    if (hit[0]-start[0]) != 0:
        mid_y_w = start[1] + (hit[1]-start[1])/(hit[0]-start[0]) * (mid_x_w - start[0])
    else:
        mid_y_w = start[1]
        
    dx = (hit[0]-start[0]); dy = (hit[1]-start[1])
    norm = np.hypot(dx, dy)
    scale_w = abs(X_START_PLOT) * 0.05
    
    if norm != 0:
        ax_wide.arrow(mid_x_w, mid_y_w, dx/norm * scale_w, dy/norm * scale_w, 
                      head_width=abs(X_START_PLOT)*0.005, fc=c_inc_wide, ec=c_inc_wide)
    
    # Zoom
    ax_zoom.plot([-4, hit[0]], [start[1] + (hit[1]-start[1])/(hit[0]-start[0])*(-4-start[0]), hit[1]], color=color_ray, linewidth=1.5)
    ax_zoom.plot(hit[0], hit[1], 'o', color=color_ray, markersize=4)
    # Seta Zoom
    ax_zoom.arrow(-2.5, start[1] + (hit[1]-start[1])/(hit[0]-start[0])*(-2.5-start[0]), 
                  (hit[0]-start[0])/norm * 0.5, (hit[1]-start[1])/norm * 0.5, 
                  head_width=0.15, fc=color_ray, ec=color_ray)

    # 2. Interno (Só Zoom)
    if 'W' in ray_data:
        W = ray_data['W']
        exit_pt = ray_data['exit_point']
        ax_zoom.plot([hit[0], W[0]], [hit[1], W[1]], color=color_ray, linewidth=1.5)
        ax_zoom.plot([W[0], exit_pt[0]], [W[1], exit_pt[1]], color=color_ray, linewidth=1.5)
        ax_zoom.plot(exit_pt[0], exit_pt[1], 'o', color=color_ray, markersize=4)
        
        # 3. Emergente (Verde no Wide, Cor original no Zoom)
        if ray_data.get('full_path'):
            end = ray_data['end']
            c_out_wide = 'green'
            
            # Wide (Sem seta conforme solicitado)
            ax_wide.plot([exit_pt[0], end[0]], [exit_pt[1], end[1]], color=c_out_wide, linewidth=1.5)
            
            # Zoom
            slope = (end[1]-exit_pt[1])/(end[0]-exit_pt[0])
            y_at_neg3 = exit_pt[1] + slope * (-3 - exit_pt[0])
            ax_zoom.plot([exit_pt[0], -3], [exit_pt[1], y_at_neg3], color=color_ray, linewidth=1.5)
            # Seta Zoom
            ax_zoom.arrow(exit_pt[0] - 0.5, exit_pt[1] + slope * (-0.5), 
                          -0.5, slope * -0.5,
                          head_width=0.15, fc=color_ray, ec=color_ray)

plot_ray_on_axes(r1_data, '#4B0082') # Roxo
plot_ray_on_axes(r2_data, 'red')     # Vermelho

# Preenchimento (Fill)
if r1_data.get('full_path') and r2_data.get('full_path'):
    # Entrada
    x_in_wide = [r1_data['start'][0], r1_data['hit'][0], r2_data['hit'][0], r2_data['start'][0]]
    y_in_wide = [r1_data['start'][1], r1_data['hit'][1], r2_data['hit'][1], r2_data['start'][1]]
    ax_wide.fill(x_in_wide, y_in_wide, color='thistle', alpha=0.4, zorder=-2)
    
    # Saída
    x_out_wide = [r1_data['exit_point'][0], r1_data['end'][0], r2_data['end'][0], r2_data['exit_point'][0]]
    y_out_wide = [r1_data['exit_point'][1], r1_data['end'][1], r2_data['end'][1], r2_data['exit_point'][1]]
    
    # WIDE: Verde com Alpha Dinâmico
    ax_wide.fill(x_out_wide, y_out_wide, color='lime', alpha=dynamic_alpha, zorder=-2)
    
    # ZOOM: Preenchimento simplificado na área visível
    poly_zoom_x = [r1_data['exit_point'][0], -3, -3, r2_data['exit_point'][0]]
    slope1 = (r1_data['end'][1]-r1_data['exit_point'][1])/(r1_data['end'][0]-r1_data['exit_point'][0])
    y1_at_neg3 = r1_data['exit_point'][1] + slope1 * (-3 - r1_data['exit_point'][0])
    
    slope2 = (r2_data['end'][1]-r2_data['exit_point'][1])/(r2_data['end'][0]-r2_data['exit_point'][0])
    y2_at_neg3 = r2_data['exit_point'][1] + slope2 * (-3 - r2_data['exit_point'][0])
    
    poly_zoom_y = [r1_data['exit_point'][1], y1_at_neg3, y2_at_neg3, r2_data['exit_point'][1]]
    ax_zoom.fill(poly_zoom_x, poly_zoom_y, color='lightcyan', alpha=0.4, zorder=-2)


# --- Renderização Final ---

# 1. Gráfico Grande no Topo
with container_top:
    # Criar uma figura limpa apenas para o Wide se necessário, mas aqui estamos plotando a figura toda
    # Para separar como pedido (Wide em cima, Zoom em baixo direita), precisamos manipular as figuras.
    
    # No Matplotlib + Streamlit, é mais fácil criar duas figuras separadas para layout complexo
    # Vou separar a criação das figuras aqui para garantir o layout perfeito.
    
    fig_wide, ax_w = plt.subplots(figsize=(12, 3))
    # Replicar configs do Wide
    ax_w.set_title("Farol: 0.65m; Visão do motorista: 1.2m", fontsize=10)
    ax_w.set_xlim(wide_x_lim)
    ax_w.set_ylim(wide_y_lim)
    for h in OBSERVATION_HEIGHTS: ax_w.axhline(y=h, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
    ax_w.set_yticks(wide_y_ticks_values); ax_w.set_yticklabels(wide_y_ticks_labels)
    ax_w.tick_params(axis='y', labelsize=8)
    ax_w.set_xticks(wide_x_ticks); ax_w.set_xticklabels(wide_x_labels)
    ax_w.set_xlabel('Distância (m)', fontsize=9)
    try: ax_w.imshow(pil_image, extent=[car_x_wide, car_x_wide + car_width_wide, car_y_wide, car_y_wide + car_height_wide], aspect='auto', zorder=-1)
    except: pass
    
    # Redesenhar elementos no ax_w
    if r1_data.get('exists'):
        ax_w.plot([r1_data['start'][0], r1_data['hit'][0]], [r1_data['start'][1], r1_data['hit'][1]], color='brown', lw=1.5)
        # Seta
        dx = (r1_data['hit'][0]-r1_data['start'][0]); dy = (r1_data['hit'][1]-r1_data['start'][1])
        mid_x = r1_data['start'][0] + abs(X_START_PLOT)*0.1
        mid_y = r1_data['start'][1] + dy/dx * (mid_x - r1_data['start'][0])
        ax_w.arrow(mid_x, mid_y, dx*0.001, dy*0.001, head_width=abs(X_START_PLOT)*0.005, fc='brown', ec='brown')
        if r1_data.get('full_path'):
            ax_w.plot([r1_data['exit_point'][0], r1_data['end'][0]], [r1_data['exit_point'][1], r1_data['end'][1]], color='green', lw=1.5)
            
    if r2_data.get('exists'):
        ax_w.plot([r2_data['start'][0], r2_data['hit'][0]], [r2_data['start'][1], r2_data['hit'][1]], color='brown', lw=1.5)
        if r2_data.get('full_path'):
            ax_w.plot([r2_data['exit_point'][0], r2_data['end'][0]], [r2_data['exit_point'][1], r2_data['end'][1]], color='green', lw=1.5)
            
    if r1_data.get('full_path') and r2_data.get('full_path'):
        ax_w.fill(x_out_wide, y_out_wide, color='lime', alpha=dynamic_alpha, zorder=-2)
        ax_w.fill(x_in_wide, y_in_wide, color='thistle', alpha=0.4, zorder=-2)
    
    ax_w.plot(x_line_ref, x_line_ref * tan_beta, color='black', linestyle='-', linewidth=1.0)
    
    st.pyplot(fig_wide)

# 2. Gráfico Pequeno na Coluna Direita (Embaixo)
with col_right:
    fig_zoom, ax_z = plt.subplots(figsize=(5, 5))
    ax_z.set_title("Visão Zoom", fontsize=10)
    ax_z.set_xlim(-3, 3); ax_z.set_ylim(-3, 3)
    ax_z.xaxis.set_major_formatter(FuncFormatter(microns_formatter))
    ax_z.yaxis.set_major_formatter(FuncFormatter(microns_formatter))
    ax_z.set_xlabel('mícrons', fontsize=9); ax_z.set_ylabel('mícrons', fontsize=9)
    ax_z.set_aspect('equal')
    
    # Redesenhar elementos no ax_z
    ax_z.plot(x_center + R * np.cos(theta_circle), y_center + R * np.sin(theta_circle), color='black')
    ax_z.plot(x_center, y_center, 'ko', markersize=5)
    ax_z.text(x_center+0.1, y_center-0.1, 'O', fontsize=10)
    ax_z.plot(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10) * tan_beta, color='black', lw=1)
    
    if r1_data.get('exists'):
        # R1 Zoom
        ax_z.plot([-4, r1_data['hit'][0]], [r1_data['start'][1] + (r1_data['hit'][1]-r1_data['start'][1])/(r1_data['hit'][0]-r1_data['start'][0])*(-4-r1_data['start'][0]), r1_data['hit'][1]], color='#4B0082')
        ax_z.plot([r1_data['hit'][0], r1_data['W'][0]], [r1_data['hit'][1], r1_data['W'][1]], color='#4B0082')
        ax_z.plot([r1_data['W'][0], r1_data['exit_point'][0]], [r1_data['W'][1], r1_data['exit_point'][1]], color='#4B0082')
        if r1_data.get('full_path'):
             slope1 = (r1_data['end'][1]-r1_data['exit_point'][1])/(r1_data['end'][0]-r1_data['exit_point'][0])
             y_end = r1_data['exit_point'][1] + slope1 * (-3 - r1_data['exit_point'][0])
             ax_z.plot([r1_data['exit_point'][0], -3], [r1_data['exit_point'][1], y_end], color='#4B0082')
             ax_z.arrow(r1_data['exit_point'][0]-0.5, r1_data['exit_point'][1]+slope1*(-0.5), -0.5, slope1*-0.5, head_width=0.15, fc='#4B0082', ec='#4B0082')

    if r2_data.get('exists'):
        # R2 Zoom
        ax_z.plot([-4, r2_data['hit'][0]], [r2_data['start'][1] + (r2_data['hit'][1]-r2_data['start'][1])/(r2_data['hit'][0]-r2_data['start'][0])*(-4-r2_data['start'][0]), r2_data['hit'][1]], color='red')
        ax_z.plot([r2_data['hit'][0], r2_data['W'][0]], [r2_data['hit'][1], r2_data['W'][1]], color='red')
        ax_z.plot([r2_data['W'][0], r2_data['exit_point'][0]], [r2_data['W'][1], r2_data['exit_point'][1]], color='red')
        if r2_data.get('full_path'):
             slope2 = (r2_data['end'][1]-r2_data['exit_point'][1])/(r2_data['end'][0]-r2_data['exit_point'][0])
             y_end = r2_data['exit_point'][1] + slope2 * (-3 - r2_data['exit_point'][0])
             ax_z.plot([r2_data['exit_point'][0], -3], [r2_data['exit_point'][1], y_end], color='red')
             ax_z.arrow(r2_data['exit_point'][0]-0.5, r2_data['exit_point'][1]+slope2*(-0.5), -0.5, slope2*-0.5, head_width=0.15, fc='red', ec='red')

    if r1_data.get('full_path') and r2_data.get('full_path'):
        ax_z.fill(poly_zoom_x, poly_zoom_y, color='lightcyan', alpha=0.4, zorder=-2)

    st.pyplot(fig_zoom)