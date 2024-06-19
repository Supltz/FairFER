import argparse
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

def extract_emotion_values(file_path, emotion_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_p_values_section = False
    in_d_values_section = False
    current_emotion = None

    selected_emotion_values = {'p_values': [], 'd_values': []}

    for line in lines:
        if 'P-Values for Differential Scores:' in line:
            in_p_values_section = True
            continue

        if 'Effect Sizes (d) for each attribute pair comparison:' in line:
            in_p_values_section = False
            in_d_values_section = True
            continue

        if f'Emotion {emotion_number}:' in line:
            current_emotion = str(emotion_number)
            continue

        if current_emotion and 'Emotion' in line and not f'{emotion_number}:' in line:
            current_emotion = None
            continue

        if current_emotion and (in_p_values_section or in_d_values_section):
            parts = line.split(':')
            if len(parts) == 2:
                value = round(float(parts[1].strip()), 3)
                if in_p_values_section:
                    selected_emotion_values['p_values'].append(value)
                elif in_d_values_section:
                    selected_emotion_values['d_values'].append(value)

    return selected_emotion_values['p_values'], selected_emotion_values['d_values']

def visualize_network(emotion, emo_dataset, attribute, attr_dataset, model, nodes, edges_with_weights, adjusted_d_values, emotion_id):
    DG = nx.DiGraph()
    DG.add_nodes_from(nodes)

    adjusted_edges_with_weights = [(edge[0], edge[1], adjusted_d_values[i])
                                   for i, edge in enumerate(edges_with_weights)]

    for edge in adjusted_edges_with_weights:
        if abs(edge[2]) >= 0.001:
            if edge[2] < 0:
                DG.add_edge(edge[1], edge[0], weight=abs(edge[2]))
            else:
                DG.add_edge(edge[0], edge[1], weight=edge[2])

    nas_scores = {}
    for node in DG.nodes():
        outgoing_weight = round(sum(weight for _, _, weight in DG.out_edges(node, data='weight', default=1)), 3) # type: ignore
        incoming_weight = round(sum(weight for _, _, weight in DG.in_edges(node, data='weight', default=1)), 3) # type: ignore
        nas_score = round(outgoing_weight - incoming_weight, 3)
        if outgoing_weight != 0 or incoming_weight != 0:
            nas_scores[node] = nas_score

    ranked_nodes = sorted(nas_scores.items(), key=lambda x: x[1], reverse=True)

    for node, score in ranked_nodes:
        print(f"Node: {node}, NAS Score: {score}")

    pos = nx.circular_layout(DG)

    node_size = 1100

    nx.draw_networkx_nodes(DG, pos, node_size=node_size, node_color='none', edgecolors='black')
    nx.draw_networkx_edges(DG, pos, edgelist=DG.edges(), edge_color='black', arrows=True, arrowstyle='->', node_size=node_size)
    nx.draw_networkx_labels(DG, pos, font_size=12, font_family="sans-serif")

    nt = Network("500px", "1000px", notebook=False, directed=True)
    nt.from_nx(DG)

    nt.edges = []

    for source, target, data in DG.edges(data=True):
        weight = data.get('weight', 1)
        nt.add_edge(source, target, label=str(weight), color='black')

    for node in nt.nodes:
        node["color"] = '#97c2fc'
        node["shape"] = "circle"
        node["size"] = 80
        node["physics"] = False

    nt.write_html(f"./graphs/graph_{emotion}_{emo_dataset}_{attribute}_{attr_dataset}_{model}.html", notebook=False)

def main():
    parser = argparse.ArgumentParser(description='Visualize Graph')
    parser.add_argument('--emotion', required=True, type=str)
    parser.add_argument('--emo_dataset', required=True, type=str)
    parser.add_argument('--attr_dataset', required=True, type=str)
    parser.add_argument('--attritbue', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    
    args = parser.parse_args()
    if args.emo_dataset == 'RAF':
        expression_dict = {"Surprise": 0, "Fear": 1, "Disgust": 2, "Happiness": 3, "Sadness": 4, "Anger": 5, "Neutral": 6}
        emotion_number = expression_dict[args.emotion]
    else:
        expression_dict = {"Neutral": 0, "Happiness": 1, "Sadness": 2, "Surprise": 3, "Fear": 4, "Disgust": 5, "Anger": 6}
        emotion_number = expression_dict[args.emotion]

    p_values, d_values = extract_emotion_values(f"./effect_size/{args.attribute}_{args.emo_dataset}_{args.attr_dataset}_differential_scores_and_effects.txt", emotion_number)

    adjusted_d_values = [d if p < 0.05 else 0 for p, d in zip(p_values, d_values)]

    if args.attribute == 'gender':
        nodes = ["Male", "Female"]
        edges_with_weights = [('Male', 'Female', 1)]
    elif args.attribute == "age":
        if args.attr_dataset == "Fairface":
            nodes = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
            edges_with_weights = [('0-2', '3-9', 1),
             ('0-2', '10-19', 1),
             ('0-2', '20-29', 1),
             ('0-2', '30-39', 1),
             ('0-2', '40-49', 1),
             ('0-2', '50-59', 1),
             ('0-2', '60-69', 1),
             ('0-2', '70+', 1),
             ('3-9', '10-19', 1),
             ('3-9', '20-29', 1),
             ('3-9', '30-39', 1),
             ('3-9', '40-49', 1),
             ('3-9', '50-59', 1),
             ('3-9', '60-69', 1),
             ('3-9', '70+', 1),
             ('10-19', '20-29', 1),
             ('10-19', '30-39', 1),
             ('10-19', '40-49', 1),
             ('10-19', '50-59', 1),
             ('10-19', '60-69', 1),
             ('10-19', '70+', 1),
             ('20-29', '30-39', 1),
             ('20-29', '40-49', 1),
             ('20-29', '50-59', 1),
             ('20-29', '60-69', 1),
             ('20-29', '70+', 1),
             ('30-39', '40-49', 1),
             ('30-39', '50-59', 1),
             ('30-39', '60-69', 1),
             ('30-39', '70+', 1),
             ('40-49', '50-59', 1),
             ('40-49', '60-69', 1),
             ('40-49', '70+', 1),
             ('50-59', '60-69', 1),
             ('50-59', '70+', 1),
             ('60-69', '70+', 1)]
        elif args.attr_dataset == "UTK":
            nodes = ["0-3", "4-19", "20-39", "40-69", "70+"]
            edges_with_weights = [('0-3', '4-19', 1),
             ('0-3', '20-39', 1),
             ('0-3', '40-69', 1),
             ('0-3', '70+', 1),
             ('4-19', '20-39', 1),
             ('4-19', '40-69', 1),
             ('4-19', '70+', 1),
             ('20-39', '40-69', 1),
             ('20-39', '70+', 1),
             ('40-69', '70+', 1)]
    elif args.attribute == "race":
        if args.attr_dataset == "Fairface":
            nodes = ["White", "Black", "Latino_Hispanic", "East", "Southeast Asian", "Indian", "Middle Eastern"]
            edges_with_weights = [('White', 'Black', 1),
             ('White', 'Latino_Hispanic', 1),
             ('White', 'East', 1),
             ('White', 'Southeast Asian', 1),
             ('White', 'Indian', 1),
             ('White', 'Middle Eastern', 1),
             ('Black', 'Latino_Hispanic', 1),
             ('Black', 'East', 1),
             ('Black', 'Southeast Asian', 1),
             ('Black', 'Indian', 1),
             ('Black', 'Middle Eastern', 1),
             ('Latino_Hispanic', 'East', 1),
             ('Latino_Hispanic', 'Southeast Asian', 1),
             ('Latino_Hispanic', 'Indian', 1),
             ('Latino_Hispanic', 'Middle Eastern', 1),
             ('East', 'Southeast Asian', 1),
             ('East', 'Indian', 1),
             ('East', 'Middle Eastern', 1),
             ('Southeast Asian', 'Indian', 1),
             ('Southeast Asian', 'Middle Eastern', 1),
             ('Indian', 'Middle Eastern', 1)]
        elif args.attr_dataset == "UTK":
            nodes = ["White", "Black", "Asian", "Indian"]
            edges_with_weights = [('White', 'Black', 1),
            ('White', 'Asian', 1),
            ('Black', 'Indian', 1),
            ('Black', 'Asian', 1),
            ('Black', 'Indian', 1),
            ('Asian', 'Indian', 1)]

    visualize_network(args.emotion, args.emo_dataset, args.attribute,args.attr_dataset, args.model, nodes, edges_with_weights, adjusted_d_values, emotion_number)

if __name__ == "__main__":
    main()