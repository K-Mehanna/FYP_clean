# Copied from https://github.com/UnheardChunk/SAA-CBR

from collections.abc import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse, lil_matrix


class AACBRParallel:
    def __init__(
        self,
        less_specific: Callable[[NDArray, NDArray], bool],
        default_case: NDArray,
        default_outcome: int,
        *,
        include_supports: bool = False,
        supported_attack_chain: bool = True,
        with_minimality: bool = True,
    ):
        self.less_specific = less_specific
        self.is_irrelevant = lambda new_case, case: np.logical_not(
            self.less_specific(case, new_case)
        )
        self.default_case = default_case
        self.default_outcome = default_outcome
        self.include_supports = include_supports
        self.supported_attack_chain = supported_attack_chain
        self.with_minimality = with_minimality

    def cartesian_product_simple_transpose(self, arrays: NDArray) -> NDArray:
        # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[i, ...] = a
        return arr.reshape(la, -1).T

    def fit(self, casebase_features: NDArray, casebase_outcomes: NDArray) -> None:
        assert len(casebase_features) == len(casebase_outcomes)

        self.casebase_features = np.append(
            casebase_features, [self.default_case], axis=0
        )
        self.casebase_outcomes = np.append(casebase_outcomes, [self.default_outcome])
        self.default_index = len(self.casebase_features) - 1
        self.casebase_indices = list(range(len(self.casebase_features)))

        # Create the abstract framework from the casebase
        if self.with_minimality:
            self._build_af()
        else:
            self._build_af_greedy()

    def _build_af(self) -> None:
        # Builds an AF with a matrix representation with a size of len(self.casebase_features)
        attacks_matrix = np.zeros(
            (len(self.casebase_features), len(self.casebase_features)), dtype=bool
        )

        if self.include_supports:
            supports_matrix = np.zeros_like(attacks_matrix, dtype=bool)

        for i in self.casebase_indices:
            source = self.casebase_features[i]
            if i == self.default_index:
                continue

            for j in self.casebase_indices:
                if i == j:
                    continue

                target = self.casebase_features[j]

                # The source can attack the target if their outcomes are different,
                # the target is less specific than the source, and the attack
                # is minimal (i.e. there is no other case that is less specific than
                # the source and more specific than the target and has the same outcome
                # as the source).
                if (
                    self.casebase_outcomes[i] != self.casebase_outcomes[j]
                    and self.less_specific(target, source)[0]
                    and self._relation_is_minimal(i, j)
                ):
                    attacks_matrix[i, j] = True

                if (
                    self.include_supports
                    and self.casebase_outcomes[i] == self.casebase_outcomes[j]
                    and self.less_specific(target, source)[0]
                    and self._relation_is_minimal(i, j, is_support=True)
                ):
                    # The source can support the target if their outcomes are the
                    # same, the target is less specific than the source and the
                    # support is minimal.
                    supports_matrix[i, j] = True

        self.attacks_matrix = attacks_matrix
        if self.include_supports:
            self.supports_matrix = supports_matrix
            # Once all attacks and supports have been fitted, find all of the
            # supported attacks in the framework
            if self.supported_attack_chain:
                self._find_supported_attacks_chained()
            else:
                self._find_supported_attacks()

    def _find_supported_attacks(self) -> None:
        # The supported attacks matrix determines the supported attack relation
        # where there is a supported attack from case i to case j if there is a
        # case k such that there i supports k and k attacks j. We can compute this
        # simply by performing a matrix multiplication between the supports
        # matrix and the attacks matrix.
        self.supported_attacks_matrix = self.supports_matrix @ self.attacks_matrix

    def _find_supported_attacks_chained(self) -> None:
        # In the chained definition, a supported attack from case i to case j
        # exists if there is a transitive chain of cases k1, k2, ..., kn such that
        # i supports k1, k1 supports k2, ..., kn-1 supports kn and kn attacks j.
        # We can compute this by finding the transitive closure of the supports
        # matrix and then performing a matrix multiplication with the attacks matrix.
        closure = np.copy(self.supports_matrix)
        for k in range(len(closure)):
            closure = np.logical_or(closure, np.outer(closure[:, k], closure[k, :]))

        self.supported_attacks_matrix = closure @ self.attacks_matrix

    def _relation_is_minimal(
        self,
        source_index: int,
        target_index: int,
        *,
        is_support: bool = False,
    ) -> bool:
        features = self.casebase_features
        outcomes = self.casebase_outcomes

        # The relation is minimal if there is no other case that is less specific
        # than the source and more specific than the target. If the relation is an
        # attack, the outcome of the other case must be the same as the source.
        # But if the relation is a support, then this doesn't matter.
        return not any(
            k != source_index
            and k != target_index
            and (is_support or outcomes[k] == outcomes[source_index])
            and self.less_specific(features[target_index], features[k])[0]
            and self.less_specific(features[k], features[source_index])[0]
            for k in self.casebase_indices
        )

    def _build_af_greedy(self) -> None:
        casebase_size = len(self.casebase_features)

        # Create a sparse matrix to store the attacks in order to save memory
        self.attacks_matrix = lil_matrix((casebase_size, casebase_size), dtype=bool)

        batch_size = 1000
        batches = np.arange(np.ceil(casebase_size / batch_size), dtype=int)
        for b in batches:
            start = b * batch_size
            end = min((b + 1) * batch_size, casebase_size)

            # Create all attacker-target index pairs
            indexes = self.cartesian_product_simple_transpose(
                [
                    np.arange(start, end),
                    np.arange(casebase_size),
                ]
            )

            attackers, attackers_labels = (
                self.casebase_features[indexes[:, 0]],
                self.casebase_outcomes[indexes[:, 0]],
            )
            targets, targets_labels = (
                self.casebase_features[indexes[:, 1]],
                self.casebase_outcomes[indexes[:, 1]],
            )

            # Delete the indexes so that we can free up some memory
            del indexes

            # We use a less strict definition of an attack here, where we don't consider
            # minimality. This is because we are using a greedy approach to build the
            # attacks matrix, and we want to avoid checking all other cases to reduce
            # the time complexity to O(n^2) instead of O(n^3).
            self.attacks_matrix[start:end, :] = np.logical_and(
                attackers_labels != targets_labels,
                self.less_specific(targets, attackers),
                dtype=bool,
            ).reshape((end - start, casebase_size))

    def show_graph_with_labels(
        self,
        new_case: NDArray = None,
        label_func: Callable = None,
        feature_names: list[str] = None,
        title: str = None,
    ) -> None:
        if new_case is not None:
            new_case_attacks = np.where(self.get_new_case_attacks_mask(new_case), -1, 0)
            attacks_matrix = np.zeros(
                (self.attacks_matrix.shape[0] + 1, self.attacks_matrix.shape[1])
            )
            attacks_matrix[:-1] = self.attacks_matrix
            attacks_matrix[-1] = new_case_attacks
            labels = np.concat((self.casebase_outcomes, [-1]))
        else:
            attacks_matrix = self.attacks_matrix
            labels = self.casebase_outcomes
        attack_rows, attack_cols = np.where(attacks_matrix != 0)
        attack_edges = list(zip(attack_rows.tolist(), attack_cols.tolist()))

        if self.include_supports:
            supports_matrix = self.supports_matrix
            support_rows, support_cols = np.where(supports_matrix)
            support_edges = list(zip(support_rows.tolist(), support_cols.tolist()))
            sa_matrix = self.supported_attacks_matrix
            supp_attack_rows, supp_attack_cols = np.where(sa_matrix)
            supp_attack_edges = list(
                zip(supp_attack_rows.tolist(), supp_attack_cols.tolist())
            )
        else:
            support_edges = []
            supp_attack_edges = []

        gr = nx.DiGraph()
        gr.add_edges_from(attack_edges)
        gr.add_node(self.default_index)  # always present even if it has no edges
        if self.include_supports:
            gr.add_edges_from(support_edges)
            gr.add_edges_from(supp_attack_edges)

        try:
            pos = nx.nx_agraph.graphviz_layout(
                gr, prog="dot", args="-Gsplines=true -Gnodesep=2"
            )
        except (ValueError, ImportError, OSError):
            # Fall back when Graphviz is unavailable in the environment.
            pos = nx.spring_layout(gr, seed=0)

        unique_labels = np.unique(labels)
        colormap = plt.get_cmap("gist_rainbow", len(unique_labels))
        label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
        node_colors = [label_to_color[labels[node]] for node in list(gr.nodes)]

        if label_func is None:
            label_func = lambda x, value: f"{x}"

        nodes_list = list(gr.nodes)
        if new_case is not None:
            nodes_list.remove(len(self.casebase_features))

        labels = {x: label_func(x, self.casebase_features[x]) for x in nodes_list}
        labels.update({self.default_index: "Default"})
        if new_case is not None:
            labels.update(
                {len(attacks_matrix) - 1: "New Case: " + label_func("", new_case)}
            )

        # for k, v in pos.items():
        #     pos.update({k: (v[0] * random.randint(-100, 100), v[1])})

        f = plt.figure(figsize=(20, 20), dpi=100)
        ax = f.add_subplot(1, 1, 1)
        if title:
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        for i, label in enumerate(unique_labels):
            ax.plot([0], [0], color=colormap(i), label=f"Outcome: {label}")

        nx.draw_networkx_nodes(gr, pos, node_color=node_colors, node_size=1000)
        nx.draw_networkx_labels(gr, pos, labels=labels, font_size=12)
        attack_edge_colors = [
            label_to_color[self.casebase_outcomes[src] if src < len(self.casebase_outcomes) else -1]
            for src, dst in attack_edges
        ]
        nx.draw_networkx_edges(
            gr,
            pos,
            edgelist=attack_edges,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=25,
            width=2.0,
            edge_color=attack_edge_colors,
            connectionstyle="arc3,rad=0.15",
        )
        if self.include_supports:
            nx.draw_networkx_edges(
                gr,
                pos,
                edgelist=support_edges,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=20,
                width=2.0,
                edge_color="green",
                connectionstyle="arc3,rad=0.15",
            )
            nx.draw_networkx_edges(
                gr,
                pos,
                edgelist=supp_attack_edges,
                arrows=True,
                arrowstyle="->",
                arrowsize=20,
                width=2.0,
                edge_color="black",
                style="dashed",
                connectionstyle="arc3,rad=0.15",
            )

        plt.legend()

        if feature_names is not None:
            MAX_DISPLAY = 30
            displayed = feature_names[:MAX_DISPLAY]
            lines = [f"{i}: {name}" for i, name in enumerate(displayed)]
            if len(feature_names) > MAX_DISPLAY:
                lines.append(f"... ({len(feature_names) - MAX_DISPLAY} more)")
            ax.text(
                1.01, 1.0,
                "Feature key\n" + "\n".join(lines),
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="lightyellow",
                    edgecolor="grey",
                    alpha=0.85,
                ),
            )
            fig = plt.gcf()
            w, h = fig.get_size_inches()
            fig.set_size_inches(w + 3.5, h)

        plt.show()

    def get_new_case_attacks_mask(self, new_cases: NDArray) -> NDArray:
        # Check if new_cases is a 1D array and expand it to 2D
        if new_cases.ndim == 1:
            new_cases = new_cases[np.newaxis, :]

        # Find which cases in the casebase are attacked by the new cases
        result = self.is_irrelevant(new_cases[:, np.newaxis, :], self.casebase_features)

        # Default should not be attacked by new case
        result[:, self.default_index] = False

        return result

    def _sparse_compute_grounded(self, new_cases_attacks: NDArray) -> NDArray:
        # This is a sparse version of the compute_grounded function
        # It uses the sparse matrix representation to compute the grounded extension
        # This is useful for large casebases where the attacks matrix is sparse

        batch_size, _ = new_cases_attacks.shape
        final_unattacked = np.zeros(
            (batch_size, self.attacks_matrix.shape[0]), dtype=bool
        )

        # Need to process each new case separately as sparse matrices can only be in 2D
        for i in range(batch_size):
            attacks_matrix = self.attacks_matrix.copy()

            # The new cases are unattacked, therefore we make the cases they attack
            # have no outgoing attacks
            attacks_matrix[new_cases_attacks[i], :] = 0

            # Find unattacked nodes (i.e columns with all 0s)
            # For each node, x, that they attack, set all attacks originating from x to 0
            # Repeat until no more changes
            while True:
                unattacked = np.logical_and(
                    attacks_matrix.getnnz(axis=0) == 0,
                    np.logical_not(new_cases_attacks[i]),
                )

                mask = unattacked[:, np.newaxis]

                # Get the rows of the attack matrix that are unattacked and zero the attacked rows
                only_unattacked = attacks_matrix.multiply(mask)

                # Need to eliminate explicit zeros so they aren't counted from getnnz()
                only_unattacked.eliminate_zeros()

                # Get the mask of the rows that are attacked by unattacked rows
                attacked = only_unattacked.getnnz(axis=0) > 0

                # If all the attacks are zero, we can stop
                if attacks_matrix[attacked].nnz == 0:
                    break

                # Zero out rows for attacked nodes
                attacks_matrix[attacked, :] = 0

            final_unattacked[i] = np.logical_and(
                attacks_matrix.getnnz(axis=0) == 0, np.logical_not(new_cases_attacks[i])
            )

        return final_unattacked

    def compute_grounded(self, new_cases_attacks: NDArray) -> NDArray:
        if new_cases_attacks.ndim == 1:
            new_cases_attacks = new_cases_attacks[np.newaxis, :]

        # If the attacks matrix is sparse, use _sparse_compute_grounded
        if issparse(self.attacks_matrix):
            return self._sparse_compute_grounded(new_cases_attacks)

        attacks_matrix = self.attacks_matrix
        if self.include_supports:
            attacks_matrix = np.logical_or(
                attacks_matrix, self.supported_attacks_matrix
            )

        batch_size, _ = new_cases_attacks.shape

        # Batch the attacks matrix - to support multiple new_cases at once
        attacks_matrix = np.tile(attacks_matrix[np.newaxis, :, :], (batch_size, 1, 1))

        # The new cases are unattacked, therefore we make the cases they attack
        # have no outgoing attacks
        attacks_matrix[new_cases_attacks, :] = 0

        # Find unattacked nodes (i.e columns with all 0s)
        # For each node, x, that they attack, set all attacks originating from x to 0
        # Repeat until no more changes
        while True:
            unattacked = np.logical_and(
                np.all(attacks_matrix == 0, axis=1),
                np.logical_not(new_cases_attacks),  # Shape: B x n
            )

            mask = unattacked[:, :, np.newaxis]

            # Get the rows of the attack matrix that are unattacked and zero the attacked rows
            only_unattacked = np.where(mask, attacks_matrix, 0)

            # Get the mask of the rows that are attacked by unattacked rows
            attacked = np.any(only_unattacked != 0, axis=1)  # Shape: B x n

            # If all the attacks are zero, we can stop
            if np.all(attacks_matrix[attacked] == 0):
                break

            # Zero out rows for attacked nodes
            attacks_matrix[attacked, :] = 0

        final_unattacked = np.logical_and(
            np.all(attacks_matrix[:,] == 0, axis=1), np.logical_not(new_cases_attacks)
        )
        return final_unattacked

    def predict(self, new_cases: NDArray) -> NDArray:
        new_cases_attacks = self.get_new_case_attacks_mask(new_cases)
        grounded = self.compute_grounded(new_cases_attacks)

        # Assuming the default case is either 1 or 0
        predicted = np.where(
            grounded[:, self.default_index],
            self.default_outcome,
            1 - self.default_outcome,
        )
        return predicted


if __name__ == "__main__":

    def strict_superset(case_a: NDArray, case_b: NDArray) -> NDArray:
        if case_a.ndim == 1:
            case_a = case_a[np.newaxis, :]
        if case_b.ndim == 1:
            case_b = case_b[np.newaxis, :]

        result = np.logical_and(
            np.all(case_a <= case_b, axis=-1), np.any(case_a < case_b, axis=-1)
        )
        return result

    label_map = {
        (0, 0, 0): "Default",
        (1, 0, 0): "Ayaan",  # 0, 0
        (0, 1, 0): "Olivia",  # 1, 0
        (0, 0, 1): "Daniel",  # 2, 0
        (3, 0, 0): "Harry",  # 3, 1
        (2, 0, 0): "Emily",  # 4, 1
        (0, 1, 1): "Grace",  # 5, 1
        (0, 0, 3): "Oliver",  # 6, 1
        (3, 1, 0): "Amelia",  # 7, 0
        (2, 2, 1): "Thomas",  # 8, 0
        (2, 1, 2): "Priya",  # 9, 0
        (1, 0, 3): "Jack",  # 10, 0
        (2, 0, 3): "Jack",  # 10, 0
        (0, 1, 3): "Aisha",  # 11, 0
        (2, 1, 3): "Jessica",  # 12, 1
        (2, 2, 2): "Hannah",  # 13, 1
        (3, 2, 1): "Charlie",  # 14, 1
        (2, 3, 1): "George",  # 15, 1
        (3, 1, 3): "Chloe",  # 16, 0
        (3, 2, 3): "Sophie",  # 17, 1
        (3, 3, 2): "Arun",  # 18, 1
    }

    X_training = np.array(
        [
            # e, d, m
            [1, 0, 0],  # 0, 0
            [0, 1, 0],  # 1, 0
            [0, 0, 1],  # 2, 0
            [3, 0, 0],  # 3, 1
            [2, 0, 0],  # 4, 1
            [0, 1, 1],  # 5, 1
            [0, 0, 3],  # 6, 1
            [3, 1, 0],  # 7, 0
            [2, 2, 1],  # 8, 0
            [2, 1, 2],  # 9, 0
            [1, 0, 3],  # 10, 0
            [0, 1, 3],  # 11, 0
            [2, 1, 3],  # 12, 1
            [2, 2, 2],  # 13, 1
            [3, 2, 1],  # 14, 1
            [2, 3, 1],  # 15, 1
            [3, 1, 3],  # 16, 0
            [3, 2, 3],  # 17, 1
            [3, 3, 2],  # 18, 1
        ]
    )
    y_training = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

    DEFAULT_OUTCOME = 1
    DEFAULT_CHAR = np.array([0, 0, 0])

    clf = AACBRParallel(
        strict_superset,
        DEFAULT_CHAR,
        DEFAULT_OUTCOME,
    )
    clf.fit(X_training, y_training)

    label_func = lambda _, value: label_map[tuple(value.tolist())]

    # clf.show_graph_with_labels(label_func=label_func)
    new_cases = np.array([[3, 1, 3]])  # 16
    print("PREDICTED", clf.predict(new_cases))
    # clf.show_graph_with_labels(new_case=new_cases[0], label_func=label_func)

    pclf = AACBRParallel(
        strict_superset,
        DEFAULT_CHAR,
        DEFAULT_OUTCOME,
    )
    pclf.fit(X_training, y_training, parallel=True)

    # pclf.show_graph_with_labels(label_func=label_func)
    new_cases = np.array([[3, 1, 3]])  # 16
    print("PREDICTED", pclf.predict(new_cases))
    # pclf.show_graph_with_labels(new_case=new_cases[0], label_func=label_func)

    assert np.all(pclf.attacks_matrix == clf.attacks_matrix)
