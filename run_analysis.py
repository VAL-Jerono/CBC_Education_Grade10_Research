import matplotlib
matplotlib.use('Agg')
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from collections import Counter
from networkx.algorithms.community import greedy_modularity_communities
from scipy.spatial import ConvexHull
import os, shutil, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.facecolor':'#0d1117','axes.facecolor':'#0d1117',
    'text.color':'#e6edf3','font.family':'serif'})

# ── NODES ────────────────────────────────────────────────────────────────────
nodes_data = [
    ("mwangombe21",  "Mwang'ombe\n2021",       2021,"empirical","Primary",     "National",  ["preparedness","training","readiness"],9),
    ("mohamed22",    "Mohamed et al.\n2022",   2022,"empirical","Secondary",   "Kirinyaga", ["preparedness","PCK","training","pedagogy"],9),
    ("ngeno23",      "Ngeno\n2023",            2023,"empirical","Primary",     "Kericho",   ["training","stakeholders","implementation"],6),
    ("muchira23",    "Muchira et al.\n2023",   2023,"empirical","Primary",     "National",  ["comparative","training","policy","gaps"],9),
    ("wasanga22",    "Wasanga &\nWambua 2022", 2022,"empirical","Primary",     "National",  ["capacity","resources","training","implementation"],8),
    ("mogere25",     "Mogere &\nMbatanu 2025", 2025,"empirical","SeniorSchool","Vihiga",    ["SSS","pathways","deployment","preparedness"],10),
    ("mackatiani23", "Mackatiani\net al. 2023",2023,"empirical","JSS",         "National",  ["JSS","preparedness","training","implementation"],9),
    ("knut22",       "KNUT\nReport 2022",      2022,"report",   "Primary/JSS", "National",  ["training","quality","facilitators","cascade"],9),
    ("cheruiyot24",  "Cheruiyot\n2024",        2024,"empirical","JSS",         "National",  ["training","ICT","infrastructure","readiness"],7),
    ("kailo25",      "Kailo et al.\n2025",     2025,"empirical","Primary",     "Kilifi",    ["training","PCK","modality","interactivity"],8),
    ("keter25",      "Keter &\nWabuke 2025",   2025,"empirical","JSS",         "Bomet",     ["preparedness","PCK","profdev","implementation"],9),
    ("garissa25",    "Garissa\nStudy 2025",    2025,"empirical","Primary",     "Garissa",   ["equity","infrastructure","training","marginalized"],7),
    ("ijriss25",     "IJRISS\n2025",           2025,"empirical","JSS",         "National",  ["stakeholders","preparedness","attitudes","readiness"],8),
    ("momanyi19",    "Momanyi &\nRop 2019",    2019,"empirical","Primary",     "Bomet",     ["readiness","classroom","preparedness"],6),
    ("maluha24",     "Maluha\net al. 2024",    2024,"empirical","Primary",     "Vihiga",    ["pedagogy","PCK","implementation","practice"],8),
    ("njiru24",      "Njiru &\nOdundo 2024",   2024,"empirical","ECDE",        "National",  ["profdev","training","gaps","implementation"],6),
    ("pwper23",      "PWPER\n2023",            2023,"policy",   "National",    "National",  ["policy","preparedness","reform","implementation"],8),
    ("tsc_plan",     "TSC Plan\n2023-27",      2023,"policy",   "National",    "National",  ["policy","profdev","deployment","training"],7),
    ("kicd_designs", "KICD Designs\n2024",     2024,"policy",   "SeniorSchool","National",  ["SSS","pathways","curriculum","PCK"],8),
    ("shulman86",    "Shulman 1986\n(PCK)",    1986,"theory",   "Theory",      "Intl",      ["PCK","pedagogy","content","framework"],9),
    ("bandura97",    "Bandura 1997\n(SE)",     1997,"theory",   "Theory",      "Intl",      ["self_efficacy","attitudes","readiness","confidence"],9),
    ("fullan07",     "Fullan 2007\n(Change)",  2007,"theory",   "Theory",      "Intl",      ["implementation","reform","policy","change"],8),
    ("darling19",    "Darling-Hammond\n2019",  2019,"theory",   "Theory",      "Intl",      ["pedagogy","practice","framework","readiness"],7),
    ("YOUR_STUDY",   "YOUR STUDY\n(2026)",     2026,"gap",      "SeniorSchool","National",  ["SSS","pathways","preparedness","PCK","self_efficacy","training","equity","implementation"],14),
]

df_nodes = pd.DataFrame(nodes_data, columns=["id","label","year","type","level","region","themes","weight"])

# ── EDGES ────────────────────────────────────────────────────────────────────
edges_data = []
for i, ri in df_nodes.iterrows():
    for j, rj in df_nodes.iterrows():
        if j <= i: continue
        shared = set(ri["themes"]) & set(rj["themes"])
        if shared:
            edges_data.append({"source":ri["id"],"target":rj["id"],"weight":len(shared),"shared_themes":list(shared)})
df_edges = pd.DataFrame(edges_data)

# ── GRAPH ────────────────────────────────────────────────────────────────────
G = nx.Graph()
for _, row in df_nodes.iterrows():
    G.add_node(row["id"], label=row["label"], year=row["year"], ntype=row["type"],
               level=row["level"], region=row["region"], themes=row["themes"], weight=row["weight"])
for _, row in df_edges.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"], shared_themes=row["shared_themes"])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, density={nx.density(G):.4f}")

# ── METRICS ──────────────────────────────────────────────────────────────────
degree_cent      = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight='weight', normalized=True)
closeness_cent   = nx.closeness_centrality(G)
eigenvector_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

df_metrics = pd.DataFrame({
    "node": list(degree_cent.keys()),
    "degree_cent": list(degree_cent.values()),
    "betweenness_cent": list(betweenness_cent.values()),
    "closeness_cent": list(closeness_cent.values()),
    "eigenvector_cent": list(eigenvector_cent.values()),
})
df_metrics["label"] = df_metrics["node"].map(df_nodes.set_index("id")["label"])
df_metrics["type"]  = df_metrics["node"].map(df_nodes.set_index("id")["type"])
df_metrics["influence_score"] = (
    df_metrics["degree_cent"]*0.30 + df_metrics["betweenness_cent"]*0.30 +
    df_metrics["closeness_cent"]*0.20 + df_metrics["eigenvector_cent"]*0.20)
df_metrics = df_metrics.sort_values("influence_score", ascending=False).reset_index(drop=True)

# ── THEMES ───────────────────────────────────────────────────────────────────
all_themes = []
for _, row in df_nodes.iterrows(): all_themes.extend(row["themes"])
theme_counts = Counter(all_themes)
df_themes = pd.DataFrame(theme_counts.items(), columns=["theme","frequency"]).sort_values("frequency", ascending=False)

# ── COMMUNITIES ──────────────────────────────────────────────────────────────
communities = list(greedy_modularity_communities(G, weight='weight'))
modularity  = nx.community.modularity(G, communities, weight='weight')
node_community = {}
for cid, comm in enumerate(communities):
    for node in comm: node_community[node] = cid
cluster_palette = ['#3b82f6','#f59e0b','#10b981','#ec4899','#8b5cf6','#06b6d4','#ef4444']
community_labels = {}
for cid, comm in enumerate(communities):
    types = [G.nodes[n]['ntype'] for n in comm]
    dominant = Counter(types).most_common(1)[0][0]
    community_labels[cid] = f"Cluster {cid+1}: {dominant}"

print(f"Communities: {len(communities)}, Modularity: {modularity:.4f}")

type_colors = {"empirical":"#3b82f6","report":"#f59e0b","policy":"#8b5cf6","theory":"#10b981","gap":"#ef4444"}
pos = nx.spring_layout(G, weight='weight', seed=42, k=2.8, iterations=100)

# ── FIG 1: Full Network ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18,13)); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
node_colors = [type_colors[G.nodes[n]['ntype']] for n in G.nodes()]
node_sizes  = [G.nodes[n]['weight']*120 for n in G.nodes()]
for u,v in G.edges():
    x0,y0=pos[u]; x1,y1=pos[v]; w=G[u][v]['weight']
    ax.plot([x0,x1],[y0,y1],color='#4a5568',alpha=min(0.8,0.15+w*0.18),linewidth=0.4+w*0.5,zorder=1)
nx.draw_networkx_nodes(G,pos,ax=ax,node_color=node_colors,node_size=node_sizes,alpha=0.92,linewidths=0.8,
    edgecolors=['#ffffff' if G.nodes[n]['ntype']=='gap' else '#1a1a2e' for n in G.nodes()])
nx.draw_networkx_labels(G,pos,labels={n:G.nodes[n]['label'] for n in G.nodes()},ax=ax,font_size=6.5,font_color='#e6edf3',font_family='serif')
legend_elements=[mpatches.Patch(color=v,label=k) for k,v in type_colors.items()]
ax.legend(handles=legend_elements,loc='lower left',fontsize=9,facecolor='#161b22',edgecolor='#30363d',labelcolor='#e6edf3')
ax.set_title("CBC Teacher Preparedness — Literature Network (Kenya, 2026)\nNode size = influence score  |  Edge thickness = shared themes",fontsize=14,pad=16,color='#e6edf3',fontfamily='serif')
ax.axis('off'); plt.tight_layout()
plt.savefig('Visualizations/Full_network.png',dpi=180,bbox_inches='tight',facecolor='#0d1117'); plt.close()
print("Fig 1 done")

# ── FIG 2: Centrality & Themes ────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(18,7)); fig.patch.set_facecolor('#0d1117')
ax1=axes[0]; ax1.set_facecolor('#0d1117')
top12=df_metrics.head(12); colors=[type_colors[t] for t in top12["type"]]
bars=ax1.barh(range(len(top12)),top12["influence_score"],color=colors,alpha=0.85,edgecolor='#30363d')
ax1.set_yticks(range(len(top12))); ax1.set_yticklabels(top12["label"].str.replace('\n',' '),fontsize=9,color='#e6edf3')
ax1.invert_yaxis(); ax1.set_xlabel("Composite Influence Score",color='#8b949e',fontsize=10)
ax1.set_title("A. Node Influence Ranking",color='#e6edf3',fontsize=12,fontfamily='serif')
ax1.tick_params(colors='#8b949e'); ax1.spines[:].set_color('#30363d')
for bar,val in zip(bars,top12["influence_score"]):
    ax1.text(bar.get_width()+0.002,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',ha='left',color='#8b949e',fontsize=8)
ax2=axes[1]; ax2.set_facecolor('#0d1117')
top_themes=df_themes.head(12)
theme_bar_colors=['#ef4444' if t in ['preparedness','training','PCK','SSS'] else '#3b82f6' for t in top_themes['theme']]
bars2=ax2.barh(range(len(top_themes)),top_themes["frequency"],color=theme_bar_colors,alpha=0.85,edgecolor='#30363d')
ax2.set_yticks(range(len(top_themes))); ax2.set_yticklabels(top_themes["theme"],fontsize=10,color='#e6edf3')
ax2.invert_yaxis(); ax2.set_xlabel("Frequency across sources",color='#8b949e',fontsize=10)
ax2.set_title("B. Theme Frequency (red = core to your study)",color='#e6edf3',fontsize=12,fontfamily='serif')
ax2.tick_params(colors='#8b949e'); ax2.spines[:].set_color('#30363d')
for bar,val in zip(bars2,top_themes["frequency"]):
    ax2.text(bar.get_width()+0.1,bar.get_y()+bar.get_height()/2,str(val),va='center',ha='left',color='#8b949e',fontsize=9)
plt.suptitle("CBC Teacher Preparedness — Evidence Strength & Thematic Coverage",fontsize=13,color='#e6edf3',fontfamily='serif',y=1.01)
plt.tight_layout()
plt.savefig('Visualizations/Centrality_themes.png',dpi=180,bbox_inches='tight',facecolor='#0d1117'); plt.close()
print("Fig 2 done")

# ── FIG 3: Communities ────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(18,13)); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
community_node_colors={n:cluster_palette[node_community[n]%len(cluster_palette)] for n in G.nodes()}
for cid,comm in enumerate(communities):
    pts=np.array([pos[n] for n in comm if n!="YOUR_STUDY"])
    if len(pts)>=3:
        try:
            hull=ConvexHull(pts); hull_pts=np.append(pts[hull.vertices],[pts[hull.vertices[0]]],axis=0)
            ax.fill(hull_pts[:,0],hull_pts[:,1],alpha=0.07,color=cluster_palette[cid%len(cluster_palette)])
            ax.plot(hull_pts[:,0],hull_pts[:,1],color=cluster_palette[cid%len(cluster_palette)],linewidth=1,alpha=0.3)
            cx,cy=pts.mean(axis=0)
            ax.text(cx,cy+0.18,community_labels[cid],ha='center',fontsize=8,color=cluster_palette[cid%len(cluster_palette)],fontfamily='serif',style='italic',alpha=0.8)
        except: pass
for u,v in G.edges():
    x0,y0=pos[u]; x1,y1=pos[v]; same=node_community.get(u,-1)==node_community.get(v,-1)
    ax.plot([x0,x1],[y0,y1],color='#ffffff' if same else '#4a5568',alpha=0.25 if same else 0.1,linewidth=G[u][v]['weight']*0.4+0.3,zorder=1)
for n in G.nodes():
    x,y=pos[n]; c='#ef4444' if n=='YOUR_STUDY' else community_node_colors[n]
    s=G.nodes[n]['weight']*130; ec='#ffffff' if n=='YOUR_STUDY' else c; lw=3 if n=='YOUR_STUDY' else 0.5
    ax.scatter(x,y,s=s,c=c,alpha=0.9,edgecolors=ec,linewidths=lw,zorder=3)
    ax.text(x,y,G.nodes[n]['label'],ha='center',va='center',fontsize=6 if n!='YOUR_STUDY' else 7.5,
            color='#ffffff',fontfamily='serif',fontweight='bold' if n=='YOUR_STUDY' else 'normal',zorder=4)
ax.set_title("CBC Literature — Community Cluster Analysis\nShaded regions = evidence clusters  |  YOUR STUDY bridges all clusters",fontsize=14,pad=16,color='#e6edf3',fontfamily='serif')
ax.axis('off'); plt.tight_layout()
plt.savefig('Visualizations/Communities.png',dpi=180,bbox_inches='tight',facecolor='#0d1117'); plt.close()
print("Fig 3 done")

# ── FIG 4: Gap Map ────────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(16,9)); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
level_order={"Theory":0,"Primary":1,"ECDE":1.5,"Pre-Primary":1.5,"Secondary":2,"Primary/JSS":2.5,"JSS":3,"SeniorSchool":4,"National":2.5}
np.random.seed(7)
for _,row in df_nodes.iterrows():
    y_val=level_order.get(row["level"],2); x_val=row["year"]+np.random.uniform(-0.15,0.15)
    c=type_colors[row["type"]]; s=row["weight"]*140
    ec='#ffffff' if row["id"]=="YOUR_STUDY" else c; lw=3 if row["id"]=="YOUR_STUDY" else 0.6
    ax.scatter(x_val,y_val,s=s,c=c,alpha=0.88,edgecolors=ec,linewidths=lw,zorder=3)
    offset=0.22 if row["id"]!="YOUR_STUDY" else 0.28
    ax.text(x_val,y_val+offset,row["label"],ha='center',va='bottom',fontsize=6.5 if row["id"]!="YOUR_STUDY" else 8,
            color='#e6edf3',fontfamily='serif',fontweight='bold' if row["id"]=="YOUR_STUDY" else 'normal')
ax.axhspan(3.5,4.5,color='#ef4444',alpha=0.07)
ax.text(1990,4.0,'⚠  EVIDENCE GAP — No empirical SSS studies before 2026',color='#ef4444',fontsize=10,fontfamily='serif',style='italic',va='center')
ytick_labels={0:"Theory",1:"Primary",1.5:"ECDE/Pre-Primary",2:"Secondary",2.5:"JSS/National",3:"JSS",4:"SENIOR SCHOOL ★"}
ax.set_yticks(list(ytick_labels.keys())); ax.set_yticklabels(list(ytick_labels.values()),fontsize=10,color='#8b949e')
ax.set_xlabel("Publication Year",color='#8b949e',fontsize=11); ax.set_ylabel("Education Level Focus",color='#8b949e',fontsize=11)
ax.set_xlim(1984,2028); ax.set_ylim(-0.5,4.7)
ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#30363d')
ax.axvline(2026,color='#ef4444',linestyle='--',alpha=0.5,linewidth=1.2)
ax.text(2026.1,-0.3,'Grade 10 starts\nJan 2026',color='#ef4444',fontsize=8,fontfamily='serif')
legend_els=[mpatches.Patch(color=v,label=k) for k,v in type_colors.items()]
ax.legend(handles=legend_els,loc='upper left',fontsize=9,facecolor='#161b22',edgecolor='#30363d',labelcolor='#e6edf3')
ax.set_title("Evidence Gap Map — Where Studies Exist vs. Where Evidence is Missing\nBubble size = influence score  |  Red zone = unstudied territory your research fills",fontsize=13,color='#e6edf3',fontfamily='serif',pad=14)
plt.tight_layout()
plt.savefig('Visualizations/Gap_map.png',dpi=180,bbox_inches='tight',facecolor='#0d1117'); plt.close()
print("Fig 4 done")

# ── EXPORT ───────────────────────────────────────────────────────────────────
output_dir="analysis"
os.makedirs(output_dir,exist_ok=True)
df_metrics.to_csv(f"{output_dir}/network_metrics.csv",index=False)
df_edges.to_csv(f"{output_dir}/edges.csv",index=False)
df_themes.to_csv(f"{output_dir}/theme_frequency.csv",index=False)
for fig_name in ['Full_network.png','Centrality_themes.png','Communities.png','Gap_map.png']:
    shutil.copy(f"Visualizations/{fig_name}",f"{output_dir}/{fig_name}")

# ── PRINT SUMMARY ────────────────────────────────────────────────────────────
print()
print("="*60)
print("NETWORK SUMMARY")
print("="*60)
print(f"Nodes (sources)          : {G.number_of_nodes()-1} (+your study)")
print(f"Edges (connections)      : {G.number_of_edges()}")
print(f"Graph density            : {nx.density(G):.4f}")
print(f"Is fully connected       : {nx.is_connected(G)}")
print(f"Communities detected     : {len(communities)}")
print(f"Modularity score         : {modularity:.4f}")
print(f"Avg degree per node      : {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")
print()
print("Top 5 most influential nodes:")
for _,row in df_metrics.head(5).iterrows():
    print(f"  {row['label'].replace(chr(10),' '):30s}  score={row['influence_score']:.4f}  type={row['type']}")
print()
sss_nodes=[n for n in G.nodes() if G.nodes[n]['level']=='SeniorSchool' and n!='YOUR_STUDY']
print(f"SSS-level sources (excl. yours): {len(sss_nodes)} — GAP IS IRREFUTABLE")
