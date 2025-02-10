import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA


@dataclass
class EmbeddingMetrics:
    reconstruction_error: float
    homogeneity_score: float
    separation_score: float
    clustering_quality: float


class Node2VecWalk:
    def __init__(self, graph: nx.Graph, p: float = 1.0, q: float = 1.0):
        self.graph = graph
        self.p = p
        self.q = q
        self.precompute_probabilities()

    def precompute_probabilities(self):
        self.alias_nodes = {}
        self.alias_edges = {}

        for node in self.graph.nodes():
            unnormalized_probs = [1.0 for _ in self.graph.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            self.alias_nodes[node] = self._alias_setup(normalized_probs)

        for edge in self.graph.edges():
            self.alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
            self.alias_edges[(edge[1], edge[0])] = self._get_alias_edge(
                edge[1], edge[0]
            )

    def _get_alias_edge(self, src: str, dst: str) -> Tuple[List[int], List[float]]:
        unnormalized_probs = []
        for dst_nbr in self.graph.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(1 / self.p)
            elif self.graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1 / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return self._alias_setup(normalized_probs)

    def _alias_setup(self, probs: List[float]) -> Tuple[List[int], List[float]]:
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int32)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J.tolist(), q.tolist()

    def simulate_walks(self, num_walks: int, walk_length: int) -> List[List[str]]:
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._node2vec_walk(walk_length, node))
        return walks

    def _node2vec_walk(self, walk_length: int, start_node: str) -> List[str]:
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(random.choice(cur_nbrs))
                else:
                    prev = walk[-2]
                    next_node = self._get_next_node(prev, cur)
                    walk.append(next_node)
            else:
                break
        return walk

    def _get_next_node(self, prev: str, cur: str) -> str:
        edge = (prev, cur)
        if edge in self.alias_edges:
            J, q = self.alias_edges[edge]
            next_node_idx = self._alias_draw(J, q)
            next_node = list(self.graph.neighbors(cur))[next_node_idx]
            return next_node
        return random.choice(list(self.graph.neighbors(cur)))

    def _alias_draw(self, J: List[int], q: List[float]) -> int:
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]


class TransactionGraphEmbedding(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(
        self, target_nodes: torch.Tensor, context_nodes: torch.Tensor
    ) -> torch.Tensor:
        target_embeds = self.embeddings(target_nodes)
        context_embeds = self.context_embeddings(context_nodes)
        dot_product = torch.sum(target_embeds * context_embeds, dim=1)
        return torch.sigmoid(dot_product)


class NodeEmbedder:
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_dim = config.get("embedding_dim", 128)
        self.walk_length = config.get("walk_length", 80)
        self.num_walks = config.get("num_walks", 10)
        self.p = config.get("p", 1.0)
        self.q = config.get("q", 1.0)
        self.window_size = config.get("window_size", 5)
        self.model = None
        self.node_embeddings = None
        self.node_mapping = {}

    def _create_training_data(
        self, walks: List[List[str]]
    ) -> Tuple[List[int], List[int], List[int]]:
        target_nodes = []
        context_nodes = []
        negative_nodes = []

        for walk in walks:
            for i, target in enumerate(walk):
                target_idx = self.node_mapping[target]
                window_start = max(0, i - self.window_size)
                window_end = min(len(walk), i + self.window_size + 1)

                for j in range(window_start, window_end):
                    if i != j:
                        context = walk[j]
                        context_idx = self.node_mapping[context]
                        target_nodes.append(target_idx)
                        context_nodes.append(context_idx)

                        # Generate negative samples
                        for _ in range(5):  # number of negative samples
                            neg_idx = random.randint(0, len(self.node_mapping) - 1)
                            negative_nodes.append(neg_idx)

        return target_nodes, context_nodes, negative_nodes

    def _train_model(
        self,
        target_nodes: List[int],
        context_nodes: List[int],
        negative_nodes: List[int],
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransactionGraphEmbedding(
            len(self.node_mapping), self.embedding_dim
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters())

        target_tensor = torch.LongTensor(target_nodes).to(device)
        context_tensor = torch.LongTensor(context_nodes).to(device)
        negative_tensor = torch.LongTensor(negative_nodes).to(device)

        batch_size = 512
        num_batches = len(target_nodes) // batch_size

        for epoch in range(5):
            total_loss = 0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                batch_target = target_tensor[start_idx:end_idx]
                batch_context = context_tensor[start_idx:end_idx]
                batch_negative = negative_tensor[start_idx:end_idx]

                optimizer.zero_grad()

                pos_score = self.model(batch_target, batch_context)
                neg_score = self.model(batch_target, batch_negative)

                loss = -torch.mean(
                    torch.log(pos_score + 1e-10) + torch.log(1 - neg_score + 1e-10)
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / num_batches}")

    def generate_embeddings(self, graph: nx.Graph) -> Dict[str, np.ndarray]:
        self.node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}

        walker = Node2VecWalk(graph, self.p, self.q)
        walks = walker.simulate_walks(self.num_walks, self.walk_length)

        target_nodes, context_nodes, negative_nodes = self._create_training_data(walks)
        self._train_model(target_nodes, context_nodes, negative_nodes)

        with torch.no_grad():
            embeddings = self.model.embeddings.weight.cpu().numpy()
            self.node_embeddings = {
                node: embeddings[idx] for node, idx in self.node_mapping.items()
            }

        return self.node_embeddings

    def compute_node_similarity(self, node1: str, node2: str) -> float:
        if self.node_embeddings is None:
            raise ValueError("Must generate embeddings first")

        emb1 = self.node_embeddings[node1]
        emb2 = self.node_embeddings[node2]

        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def find_similar_nodes(self, node: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.node_embeddings is None:
            raise ValueError("Must generate embeddings first")

        target_embedding = self.node_embeddings[node]
        similarities = []

        for other_node, embedding in self.node_embeddings.items():
            if other_node != node:
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((other_node, float(similarity)))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    def compute_metrics(self, graph: nx.Graph) -> EmbeddingMetrics:
        if self.node_embeddings is None:
            raise ValueError("Must generate embeddings first")

        embeddings_matrix = np.vstack(list(self.node_embeddings.values()))

        # Reconstruction error
        adj_matrix = nx.adjacency_matrix(graph)
        reconstructed = np.dot(embeddings_matrix, embeddings_matrix.T)
        reconstruction_error = np.mean((adj_matrix.toarray() - reconstructed) ** 2)

        # Homogeneity score
        distances = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_embeddings = np.vstack(
                    [self.node_embeddings[n] for n in neighbors]
                )
                centroid = np.mean(neighbor_embeddings, axis=0)
                distances.append(
                    np.mean(np.linalg.norm(neighbor_embeddings - centroid, axis=1))
                )
        homogeneity_score = 1 / (1 + np.mean(distances)) if distances else 0

        # Separation score
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        kmeans = KMeans(n_clusters=min(8, len(graph)), random_state=42)
        clusters = kmeans.fit_predict(embeddings_matrix)
        separation_score = float(silhouette_score(embeddings_matrix, clusters))

        # Clustering quality
        clustering_quality = nx.algorithms.community.modularity(
            graph, [list(c) for c in nx.community.label_propagation_communities(graph)]
        )

        return EmbeddingMetrics(
            reconstruction_error=reconstruction_error,
            homogeneity_score=homogeneity_score,
            separation_score=separation_score,
            clustering_quality=clustering_quality,
        )

    def save_embeddings(self, filepath: str):
        if self.node_embeddings is None:
            raise ValueError("Must generate embeddings first")

        np.save(
            filepath, {"embeddings": self.node_embeddings, "mapping": self.node_mapping}
        )

    def load_embeddings(self, filepath: str):
        data = np.load(filepath, allow_pickle=True).item()
        self.node_embeddings = data["embeddings"]
        self.node_mapping = data["mapping"]
