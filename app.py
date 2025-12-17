import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter
from matplotlib.patches import Arc
import numpy as np
from PIL import Image
import os

# --- Configuração da Página ---
st.set_page_config(page_title="Refração e Retrorrefletividade", layout="wide")

# --- CSS PERSONALIZADO (Fontes 60% menores e Branco) ---
st.markdown("""
<style>
    /* Reduzir drasticamente o tamanho da fonte (aprox 60% do anterior) */
    .stSlider label, .stRadio label, .stWidgetLabel {
        font-size: 10px !important;
    }
    .stMarkdown p {
        font-size: 10px !important;
    }
    /* Reduzir textos de ajuda/caption */
    .stCaption {
        font-size: 9px !important;
    }
    /* Títulos das colunas menores */
    h3 {
        font-size: 12px !important;
        padding-bottom: 0rem !important;
        margin-bottom: 0.2rem !important;
    }
    /* Título principal */
    h4 {
        font-size: 16px !important;
    }
    /* Reduzir espaçamento vertical entre widgets */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.3rem !important;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    /* Ajuste fino para o container principal */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES AUXILIARES ---
def microns_formatter(x, pos):
    """Converte unidades do gráfico para mícrons na legenda."""
    return f"{int(x * 500)}"

# Título (Menor para economizar espaço)
st.markdown("#### Análise de Geometria - Refração e Retrorrefletividade (15m / 30m)")

# --- Definição do Layout ---

# 1. Container Superior (Gráfico Grande)
container_top = st.container()

# 2. Container Inferior (3 Colunas Iguais)
col1, col2, col3 = st.columns([1, 1, 1])

# --- COLUNA 1: Parâmetros Globais ---
with col1:
    st.markdown("### Parâmetros Físicos")
    
    geo_option = st.radio("Geometria da Via", options=[0, 1], 
                          format_func=lambda x: "15 metros" if x == 0 else "30 metros", 
                          horizontal=True)
    
    # REMOVIDO: st.markdown("---") 
    
    val_R = st.slider("Raio da Esfera (µm)", 50, 1000, 500, 10)
    val_A_percent = st.slider("Ancoragem (%)", 50.0, 60.0, 50.0, 0.1)
    val_beta = st.slider("Inclinação Tinta β (°)", 0.0, 15.0, 5.0, 0.1)
    val_k = st.slider("Índice Refração (k)", 1.0, 4.0, 1.9, 0.01)

# --- COLUNA 2: Controles dos Raios ---
with col2:
    # REMOVIDO: st.markdown("### Raios de Luz")
    
    val_alpha_1 = st.slider("Ponto P (Roxo) - Ângulo (°)", 0.0, 90.0, 0.0, 0.1)
    val_alpha_2 = st.slider("Ponto A (Vermelho) - Ângulo (°)", 0.0, 90.0, 10.0, 0.1)

# --- LÓGICA DE CÁLCULO (BACKEND) ---

# Conversão e Preparação
R_micrometers = float(val_R)
R = R_micrometers / 500.0 
A = val_A_percent / 100.0
beta_radians = np.deg2rad(val_beta)
k = val_k

dist_center_to_line = R * abs(2 * A - 1)
if A >= 0.5:
    direction_vector = np.array([np.sin(beta_radians), -np.cos(beta_radians)])
else:
    direction_vector = np.array([-np.sin(beta_radians), np.cos(beta_radians)])
x_center, y_center = dist_center_to_line * direction_vector
tan_beta = np.tan(beta_radians)

A_quad = 1 + tan_beta**2
B_quad = -2 * (x_center + y_center * tan_beta)
C_quad = x_center**2 + y_center**2 - R**2
delta = B_quad**2 - 4 * A_quad * C_quad
Q_exists = delta >= 0

if geo_option == 0: # 15m
    angle_inc_rad_calc = np.arctan(0.65 / 15.0)
    angle_inc_deg = -np.degrees(angle_inc_rad_calc)
    X_START_PLOT = -30000.0
    OBSERVATION_HEIGHTS = [1300, 2400]
    wide_x_lim = (-30000, 10); wide_x_ticks = [-30000, 0]; wide_x_labels = ["-15m", "0m"]
    wide_y_lim = (-100, 3200); wide_y_ticks_values = [1300, 2400]; wide_y_ticks_labels = ["0.65m", "1.20m"]
else: # 30m
    angle_inc_deg = -1.24
    X_START_PLOT = -60000.0
    OBSERVATION_HEIGHTS = [1300, 2400]
    wide_x_lim = (-60000, 10); wide_x_ticks = [-60000, 0]; wide_x_labels = ["-30m", "0m"]
    wide_y_lim = (-100, 3200); wide_y_ticks_values = [1300, 2400]; wide_y_ticks_labels = ["0.65m", "1.20m"]

angle_inc_rad = np.deg2rad(angle_inc_deg)
I_vec = np.array([np.cos(angle_inc_rad), np.sin(angle_inc_rad)])

ref_angle_Q = 0.0
if Q_exists:
    x1 = (-B_quad + np.sqrt(delta)) / (2 * A_quad)
    x2 = (-B_quad - np.sqrt(delta)) / (2 * A_quad)
    x_Q = min(x1, x2)
    y_Q = tan_beta * x_Q
    ref_angle_Q = np.arctan2(y_Q - y_center, x_Q - x_center)

def calculate_ray(alpha_deg):
    res = {'exists': False}
    if not Q_exists: return res
    
    angle_P = ref_angle_Q - np.deg2rad(alpha_deg)
    x_P = x_center + R * np.cos(angle_P)
    y_P = y_center + R * np.sin(angle_P)
    res['hit'] = (x_P, y_P); res['angle_P_rad'] = angle_P
    
    x_start = X_START_PLOT
    y_start = y_P + (I_vec[1] / I_vec[0]) * (x_start - x_P) if I_vec[0] != 0 else y_P
    res['start'] = (x_start, y_start)
    
    N_vec = np.array([(x_P - x_center)/R, (y_P - y_center)/R])
    cos_z = -np.dot(N_vec, I_vec)
    if cos_z < 0: return res
    
    res['exists'] = True
    z_rad = np.arccos(np.clip(cos_z, -1.0, 1.0))
    arg_arcsin = np.sin(z_rad) / k
    if abs(arg_arcsin) > 1: return res
    j_rad = np.arcsin(arg_arcsin)
    
    n1, n2 = 1.0, k
    cos_t1, cos_t2 = np.cos(z_rad), np.cos(j_rad)
    rs = ((n1*cos_t1 - n2*cos_t2)/(n1*cos_t1 + n2*cos_t2))**2
    rp = ((n1*cos_t2 - n2*cos_t1)/(n1*cos_t2 + n2*cos_t1))**2
    T_1 = 1 - (rs + rp)/2
    res['T1'] = T_1
    
    c = cos_z; radicand = 1.0 - (1.0/k)**2 * (1.0 - c**2)
    T_vec = (1.0/k) * I_vec + ((1.0/k) * c - np.sqrt(np.clip(radicand, 0, None))) * N_vec
    P_vec = np.array([x_P, y_P]); O_vec = np.array([x_center, y_center])
    t_val = -2 * np.dot(T_vec, (P_vec - O_vec))
    W_vec = P_vec + t_val * T_vec
    res['W'] = (W_vec[0], W_vec[1])
    
    vec_normal_W = (W_vec - O_vec) / R
    vec_inc_W = T_vec
    vec_refletido = vec_inc_W - 2 * np.dot(vec_inc_W, vec_normal_W) * vec_normal_W
    L_vec = W_vec + t_val * vec_refletido
    res['exit_point'] = (L_vec[0], L_vec[1])
    res['angle_exit_rad'] = np.arctan2(L_vec[1] - y_center, L_vec[0] - x_center)
    
    vec_normal_L = (L_vec - O_vec) / R; vec_inc_L = vec_refletido
    arg_sin_out = k * np.sin(j_rad)
    if abs(arg_sin_out) < 1.0:
        n1_out, n2_out = k, 1.0
        theta1_out, theta2_out = j_rad, np.arcsin(arg_sin_out)
        cos_t1_out, cos_t2_out = np.cos(theta1_out), np.cos(theta2_out)
        rs_out = ((n1_out*cos_t1_out - n2_out*cos_t2_out)/(n1_out*cos_t1_out + n2_out*cos_t2_out))**2
        rp_out = ((n1_out*cos_t2_out - n2_out*cos_t1_out)/(n1_out*cos_t2_out + n2_out*cos_t1_out))**2
        T_out = 1 - (rs_out + rp_out)/2
        res['T_total'] = T_1 * T_out
        
        c1_exit = np.dot(vec_inc_L, vec_normal_L)
        radicand_out = 1.0 - k**2 * (1.0 - c1_exit**2)
        vec_saida = k * vec_inc_L - (k * c1_exit - np.sqrt(np.clip(radicand_out, 0, None))) * vec_normal_L
        vx, vy = vec_saida[0], vec_saida[1]
        
        if vx < 0:
            t_M = (X_START_PLOT - L_vec[0]) / vx
            y_M = L_vec[1] + t_M * vy
            res['end'] = (X_START_PLOT, y_M); res['full_path'] = True
            
            vec_chao = np.array([-1.0, 0.0])
            vec_saida_unit = vec_saida / np.linalg.norm(vec_saida)
            cos_theta = np.dot(vec_saida_unit, vec_chao)
            res['ang_ground'] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            
            vec_inc_invertido = -I_vec
            cos_obs = np.dot(vec_saida_unit, vec_inc_invertido)
            res['ang_obs'] = np.degrees(np.arccos(np.clip(cos_obs, -1.0, 1.0)))
            
    return res

r1_data = calculate_ray(val_alpha_1)
r2_data = calculate_ray(val_alpha_2)

# --- OUTPUTS NA COLUNA 2 (TEXTOS EM BRANCO) ---
with col2:
    st.markdown("### Resultados")
    
    # Texto formatado com HTML para fonte pequena E COR BRANCA (white)
    def small_text(label, value, color="white"):
        st.markdown(f"<span style='color:{color}; font-size:11px; display:block; margin-bottom:2px;'><b>{label}:</b> {value}</span>", unsafe_allow_html=True)

    if r1_data.get('full_path'):
        small_text("Transmissão (Roxo)", f"{r1_data['T_total']*100:.1f}%", "white")
        small_text("Ângulo Obs. (Roxo)", f"{r1_data['ang_obs']:.2f}º", "white")
    
    if r2_data.get('full_path'):
        small_text("Transmissão (Vermelho)", f"{r2_data['T_total']*100:.1f}%", "white")
        small_text("Ângulo Obs. (Vermelho)", f"{r2_data['ang_obs']:.2f}º", "white")
    
    # REMOVIDO: st.markdown("---")
    
    if r1_data.get('exists') and r2_data.get('exists'):
        angle_inc = np.degrees(abs(r1_data['angle_P_rad'] - r2_data['angle_P_rad']))
        arc_inc = R_micrometers * abs(r1_data['angle_P_rad'] - r2_data['angle_P_rad'])
        small_text("Arco Incidência", f"{angle_inc:.2f}º / {arc_inc:.1f}µm", "white")
        
        if 'angle_exit_rad' in r1_data and 'angle_exit_rad' in r2_data:
            angle_out = np.degrees(abs(r1_data['angle_exit_rad'] - r2_data['angle_exit_rad']))
            arc_out = R_micrometers * abs(r1_data['angle_exit_rad'] - r2_data['angle_exit_rad'])
            small_text("Arco Saída", f"{angle_out:.2f}º / {arc_out:.1f}µm", "white")

    dynamic_alpha = 0.0
    if r1_data.get('full_path') and r2_data.get('full_path'):
        y1, y2 = r1_data['end'][1]/2000.0, r2_data['end'][1]/2000.0
        gamma, lam = min(y1, y2), max(y1, y2)
        spread = abs(gamma - lam)
        small_text("Altura Saída", f"{gamma:.3f}m - {lam:.3f}m (Dif: {spread*100:.1f}cm)", "white")
        dynamic_alpha = max(0.02, 0.9 * np.exp(-2.0 * spread))

# --- PLOTAGEM ---

# 1. Gráfico WIDE (Container Topo)
with container_top:
    fig_wide, ax_w = plt.subplots(figsize=(12, 2.5))
    ax_w.set_title("Farol: 0.65m; Visão do motorista: 1.2m", fontsize=10)
    ax_w.set_xlim(wide_x_lim)
    ax_w.set_ylim(wide_y_lim)
    for h in OBSERVATION_HEIGHTS: ax_w.axhline(y=h, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
    ax_w.set_yticks(wide_y_ticks_values); ax_w.set_yticklabels(wide_y_ticks_labels)
    ax_w.tick_params(axis='y', labelsize=8)
    ax_w.set_xticks(wide_x_ticks); ax_w.set_xticklabels(wide_x_labels)
    ax_w.set_xlabel('Distância (m)', fontsize=9)
    
    try:
        pil_image = Image.open("carro.png")
        car_w = abs(X_START_PLOT) * 0.15 
        car_h = car_w * (pil_image.height / pil_image.width)
        y_ref = r1_data['start'][1] if r1_data.get('start') else 0
        ax_w.imshow(pil_image, extent=[X_START_PLOT - car_w*0.75, X_START_PLOT + car_w*0.25, 
                                       y_ref - car_h*0.5, y_ref + car_h*0.5], aspect='auto', zorder=-1)
    except: pass
    
    if r1_data.get('exists'):
        ax_w.plot([r1_data['start'][0], r1_data['hit'][0]], [r1_data['start'][1], r1_data['hit'][1]], color='brown', lw=1.5)
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
        x_out = [r1_data['exit_point'][0], r1_data['end'][0], r2_data['end'][0], r2_data['exit_point'][0]]
        y_out = [r1_data['exit_point'][1], r1_data['end'][1], r2_data['end'][1], r2_data['exit_point'][1]]
        ax_w.fill(x_out, y_out, color='lime', alpha=dynamic_alpha, zorder=-2)
        
        x_in = [r1_data['start'][0], r1_data['hit'][0], r2_data['hit'][0], r2_data['start'][0]]
        y_in = [r1_data['start'][1], r1_data['hit'][1], r2_data['hit'][1], r2_data['start'][1]]
        ax_w.fill(x_in, y_in, color='thistle', alpha=0.4, zorder=-2)
        
    x_line = np.linspace(X_START_PLOT, 10, 2)
    ax_w.plot(x_line, x_line * tan_beta, color='black', lw=1)
    
    st.pyplot(fig_wide)

# 2. Gráfico ZOOM (Container Inferior Direito - Col3)
with col3:
    fig_z, ax_z = plt.subplots(figsize=(4, 4))
    ax_z.set_title("Visão Zoom", fontsize=10)
    ax_z.set_xlim(-3, 3); ax_z.set_ylim(-3, 3)
    ax_z.xaxis.set_major_formatter(FuncFormatter(microns_formatter))
    ax_z.yaxis.set_major_formatter(FuncFormatter(microns_formatter))
    ax_z.set_xlabel('mícrons', fontsize=9); ax_z.set_ylabel('mícrons', fontsize=9)
    ax_z.set_aspect('equal')
    
    ax_z.plot(x_center + R * np.cos(np.linspace(0, 2*np.pi, 200)), y_center + R * np.sin(np.linspace(0, 2*np.pi, 200)), color='black')
    ax_z.plot(x_center, y_center, 'ko', markersize=4)
    ax_z.plot(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10) * tan_beta, color='black', lw=1)
    
    for data, clr in [(r1_data, '#4B0082'), (r2_data, 'red')]:
        if data.get('exists'):
            h, s = data['hit'], data['start']
            ax_z.plot([-4, h[0]], [s[1] + (h[1]-s[1])/(h[0]-s[0])*(-4-s[0]), h[1]], color=clr)
            if 'W' in data:
                w, ex = data['W'], data['exit_point']
                ax_z.plot([h[0], w[0]], [h[1], w[1]], color=clr)
                ax_z.plot([w[0], ex[0]], [w[1], ex[1]], color=clr)
                if data.get('full_path'):
                    sl = (data['end'][1]-ex[1])/(data['end'][0]-ex[0])
                    ax_z.plot([ex[0], -3], [ex[1], ex[1] + sl * (-3 - ex[0])], color=clr)
                    ax_z.arrow(ex[0]-0.5, ex[1]+sl*(-0.5), -0.5, sl*-0.5, head_width=0.15, fc=clr, ec=clr)

    if r1_data.get('full_path') and r2_data.get('full_path'):
        poly_x = [r1_data['exit_point'][0], -3, -3, r2_data['exit_point'][0]]
        sl1 = (r1_data['end'][1]-r1_data['exit_point'][1])/(r1_data['end'][0]-r1_data['exit_point'][0])
        sl2 = (r2_data['end'][1]-r2_data['exit_point'][1])/(r2_data['end'][0]-r2_data['exit_point'][0])
        y1 = r1_data['exit_point'][1] + sl1 * (-3 - r1_data['exit_point'][0])
        y2 = r2_data['exit_point'][1] + sl2 * (-3 - r2_data['exit_point'][0])
        poly_y = [r1_data['exit_point'][1], y1, y2, r2_data['exit_point'][1]]
        ax_z.fill(poly_x, poly_y, color='lightcyan', alpha=0.4, zorder=-2)

    st.pyplot(fig_z)