import matplotlib.pyplot as plt
import networkx as nx

# 假设这是你的四种4mC物种数据和准确性信息
species = ['4mC_C.equisetifolia', '4mC_F.vesca', '4mC_S.cerevisiae', '4mC_Tolypocladium']
accuracy = [0.9182, 0.9313, 0.9144, 0.9007]  # 注意这里的准确性信息已经标准化到0~1之间

# 构建分类树的关系
taxonomy_tree_edges = [
    ('Root', 'Cluster A'),
    ('Root', 'Cluster B'),
    ('Cluster A', '4mC_C.equisetifolia'),
    ('Cluster A', '4mC_F.vesca'),
    ('Cluster B', '4mC_S.cerevisiae'),
    ('Cluster B', '4mC_Tolypocladium')
]

# 创建有向图
G = nx.DiGraph()
G.add_edges_from(taxonomy_tree_edges)

# 绘制分类树
pos = nx.spring_layout(G, seed=42)  # 定义节点位置，这里使用了spring_layout算法
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightgray', font_size=10)

# 添加准确性信息的条形柱状图
plt.figure(figsize=(10, 6))
plt.barh(species, accuracy, color='skyblue')
plt.xlim(0.7, 0.95)  # 设置x轴标尺范围从0.7到1.0
plt.xlabel('Accuracy')
plt.ylabel('Species')
plt.title('Taxonomy Tree and Accuracy for 4mC Species')
plt.tight_layout()
plt.show()
