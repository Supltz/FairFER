import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

emotion_id = 5
def extract_emotion_values(file_path, emotion_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Flags to check which section we are in
    in_p_values_section = False
    in_d_values_section = False
    current_emotion = None

    # Dictionary to store p-values and d-values for the selected emotion
    selected_emotion_values = {'p_values': [], 'd_values': []}

    for line in lines:
        # Check if we're in the P-Values section
        if 'P-Values for Differential Scores:' in line:
            in_p_values_section = True
            continue

        # Check if we've moved on to the Effect Sizes (d) section
        if 'Effect Sizes (d) for each attribute pair comparison:' in line:
            in_p_values_section = False
            in_d_values_section = True
            continue

        # Check if we're in the selected emotion section
        if f'Emotion {emotion_number}:' in line:
            current_emotion = str(emotion_number)
            continue

        # If we're no longer in the selected emotion's section, reset the current_emotion flag
        if current_emotion and 'Emotion' in line and not f'{emotion_number}:' in line:
            current_emotion = None
            continue

        # If we're in the selected emotion section of P-Values or d-Values, extract the values
        if current_emotion and (in_p_values_section or in_d_values_section):
            parts = line.split(':')
            if len(parts) == 2:
                value = round(float(parts[1].strip()), 3)
                if in_p_values_section:
                    selected_emotion_values['p_values'].append(value)
                elif in_d_values_section:
                    selected_emotion_values['d_values'].append(value)

    return selected_emotion_values['p_values'], selected_emotion_values['d_values']

p_values, d_values = extract_emotion_values('race_affectnet_utk_differential_scores_and_effects.txt', emotion_id)



adjusted_d_values = [d if p < 0.05 else 0 for p, d in zip(p_values, d_values)]



# Create a directed graph
DG = nx.DiGraph()

# Add nodes representing age ranges
#nodes = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
#nodes = ["0-3", "4-19", "20-39", "40-69", "70+"]

#nodes = ["White", "Black", "Latino_Hispanic", "East", "Southeast Asian", "Indian", "Middle Eastern"]
# nodes = ["A₁", "A₂", "A₃", "A₄", "Aₘ"]
nodes = ["White", "Black", "Asian", "Indian"]
#nodes = ["Male", "Female"]
DG.add_nodes_from(nodes)

#edges_with_weights = [('Male', 'Female', 1)]

# Define edges with weights
# edges_with_weights = [('0-2', '3-9', 1),
#  ('0-2', '10-19', 1),
#  ('0-2', '20-29', 1),
#  ('0-2', '30-39', 1),
#  ('0-2', '40-49', 1),
#  ('0-2', '50-59', 1),
#  ('0-2', '60-69', 1),
#  ('0-2', '70+', 1),
#  ('3-9', '10-19', 1),
#  ('3-9', '20-29', 1),
#  ('3-9', '30-39', 1),
#  ('3-9', '40-49', 1),
#  ('3-9', '50-59', 1),
#  ('3-9', '60-69', 1),
#  ('3-9', '70+', 1),
#  ('10-19', '20-29', 1),
#  ('10-19', '30-39', 1),
#  ('10-19', '40-49', 1),
#  ('10-19', '50-59', 1),
#  ('10-19', '60-69', 1),
#  ('10-19', '70+', 1),
#  ('20-29', '30-39', 1),
#  ('20-29', '40-49', 1),
#  ('20-29', '50-59', 1),
#  ('20-29', '60-69', 1),
#  ('20-29', '70+', 1),
#  ('30-39', '40-49', 1),
#  ('30-39', '50-59', 1),
#  ('30-39', '60-69', 1),
#  ('30-39', '70+', 1),
#  ('40-49', '50-59', 1),
#  ('40-49', '60-69', 1),
#  ('40-49', '70+', 1),
#  ('50-59', '60-69', 1),
#  ('50-59', '70+', 1),
#  ('60-69', '70+', 1)]

# edges_with_weights = [('0-3', '4-19', 1),
#  ('0-3', '20-39', 1),
#  ('0-3', '40-69', 1),
#  ('0-3', '70+', 1),
#  ('4-19', '20-39', 1),
#  ('4-19', '40-69', 1),
#  ('4-19', '70+', 1),
#  ('20-39', '40-69', 1),
#  ('20-39', '70+', 1),
#  ('40-69', '70+', 1)]

edges_with_weights = [('White', 'Black', 1),
 ('White', 'Asian', 1),
 ('Black', 'Indian', 1),
 ('Black', 'Asian', 1),
 ('Black', 'Indian', 1),
 ('Asian', 'Indian', 1)]



# edges_with_weights = [('White', 'Black', 1),
#  ('White', 'Latino_Hispanic', 1),
#  ('White', 'East', 1),
#  ('White', 'Southeast Asian', 1),
#  ('White', 'Indian', 1),
#  ('White', 'Middle Eastern', 1),
#  ('Black', 'Latino_Hispanic', 1),
#  ('Black', 'East', 1),
#  ('Black', 'Southeast Asian', 1),
#  ('Black', 'Indian', 1),
#  ('Black', 'Middle Eastern', 1),
#  ('Latino_Hispanic', 'East', 1),
#  ('Latino_Hispanic', 'Southeast Asian', 1),
#  ('Latino_Hispanic', 'Indian', 1),
#  ('Latino_Hispanic', 'Middle Eastern', 1),
#  ('East', 'Southeast Asian', 1),
#  ('East', 'Indian', 1),
#  ('East', 'Middle Eastern', 1),
#  ('Southeast Asian', 'Indian', 1),
#  ('Southeast Asian', 'Middle Eastern', 1),
#  ('Indian', 'Middle Eastern', 1)]

# adjusted_edges_with_weights = [
#     ("A₁", "A₂", "d₁₂"),
#     ("A₁", "A₃", "d₁₃"),
#     ("A₁", "A₄", "d₁₄"),
#     ("A₂", "A₃", "d₂₃"),
#     ("A₃", "A₄", "d₃₄"),
#     ("Aₘ", "A₁", "dₘ₁"),
# ]



adjusted_edges_with_weights = [(edge[0], edge[1], adjusted_d_values[i])
                                   for i, edge in enumerate(edges_with_weights)]

# for edge in adjusted_edges_with_weights:
#     DG.add_edge(edge[0], edge[1], label=edge[2])

for edge in adjusted_edges_with_weights:
    if abs(edge[2]) >= 0.001:
        if edge[2] < 0:# Only add the edge if the weight is greater than 0.01
            DG.add_edge(edge[1], edge[0], weight=abs(edge[2]))
        else:
            DG.add_edge(edge[0], edge[1], weight=edge[2])

# Calculate NAS scores
nas_scores = {}
for node in DG.nodes():
    outgoing_weight = round(sum(weight for _, _, weight in DG.out_edges(node, data='weight', default=1)),3)
    incoming_weight = round(sum(weight for _, _, weight in DG.in_edges(node, data='weight', default=1)),3)
    nas_score = round(outgoing_weight - incoming_weight, 3)
    if outgoing_weight != 0 or incoming_weight != 0:  # Exclude nodes with no edges
        nas_scores[node] = nas_score

# Rank nodes by NAS score
ranked_nodes = sorted(nas_scores.items(), key=lambda x: x[1], reverse=True)

# Print out the ranked nodes and their NAS scores
for node, score in ranked_nodes:
    print(f"Node: {node}, NAS Score: {score}")

pos = nx.circular_layout(DG)

# Increase node size for better visibility
node_size = 1100  # Adjust node size as needed

# Draw the network
nx.draw_networkx_nodes(DG, pos, node_size=node_size, node_color='none', edgecolors='black')
nx.draw_networkx_edges(DG, pos, edgelist=DG.edges(), edge_color='black', arrows=True, arrowstyle='->', node_size=node_size)
nx.draw_networkx_labels(DG, pos, font_size=12, font_family="sans-serif")

# Convert NetworkX graph to Pyvis network
nt = Network("500px", "1000px", notebook=False, directed = True)
nt.from_nx(DG)

# Clear any existing edges in 'nt' that might have been added by 'from_nx'
nt.edges = []

# Explicitly add each edge from 'DG' to 'nt', including the weight as a label
for source, target, data in DG.edges(data=True):
    weight = data.get('width', 1)  # Replace 'weight' with the correct attribute name if different
    # Explicitly add edge with weight as label for display
    nt.add_edge(source, target, label=str(weight), color='black')

for node in nt.nodes:
    node["color"] = '#97c2fc'
    node["shape"] = "circle"
    node["size"] = 80
    node["physics"] = False

# Save and display the interactive graph
nt.write_html("graph_test_{}.html".format(emotion_id), notebook=False)



