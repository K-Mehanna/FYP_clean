from collections import defaultdict
from collections.abc import Callable
from copy import copy

from graphviz import Digraph

from .case import Case


class AACBR:
    def __init__(
        self,
        less_specific: Callable[[Case, Case], bool],
        default_case: Case,
        *,
        include_supports: bool = False,
        supported_attack_chain: bool = True,
    ):
        # A function that defines a partial order on two cases to determine if
        # one is less specific than the other.
        self.less_specific = less_specific

        # Check that the default case is actually the default case
        assert default_case.is_default_case
        self.default_case = default_case

        # Mapping of case to the cases this is attacked by
        self.case_attacked_by = defaultdict(list)

        # Mapping of case to the cases this is supported by
        self.case_supported_by = defaultdict(list)

        # Mapping of case to the cases where supported attacks to this come from
        self.case_supported_attackers = defaultdict(list)

        # Indicates if the framework should include supports when fitting the casebase
        self.include_supports = include_supports

        # If supports are included, indicates if the supported attacks should be chained or not
        self.supported_attack_chain = supported_attack_chain

    def fit(self, casebase: list[Case]) -> None:
        # Check that another default case is not included in the provided casebase
        assert sum(1 for case in casebase if case.is_default_case) == 0

        # Check that there aren't any new cases in the casebase
        assert all(not case.is_new_case for case in casebase)

        self.casebase = [self.default_case] + casebase
        for x in self.casebase:
            for y in self.casebase:
                # Don't consider self attacks
                if x == y:
                    continue

                # Find if x attacks y
                self._attacks_case(x, y)

                if self.include_supports:
                    # Find if x supports y
                    self._supports_case(x, y)

        if self.include_supports:
            # Once all attacks and supports have been fitted, find all of the
            # supported attacks in the framework
            if self.supported_attack_chain:
                self._find_supported_attacks_chained()
            else:
                self._find_supported_attacks()

    def _attacks_case(self, x: Case, y: Case) -> None:
        # x attacks y if their outcomes are different, y is less specific
        # than x and this attack is minimal (i.e. there isn't some z such
        # that y is strictly less specific than z and z is strictly less
        # specific than x).
        if (
            x.different_outcome_to(y)
            and self.less_specific(y, x)
            and self._relation_is_minimal(x, y)
        ):
            self.case_attacked_by[y].append(x)

    def _supports_case(self, x: Case, y: Case) -> None:
        # x supports y if their outcomes are the same, y is less specific than x
        # and this support is minimal (i.e. there isn't some z such that y is
        # strictly less specific than z and z is strictly less specific than x).
        if (
            x.same_outcome_to(y)
            and self.less_specific(y, x)
            and self._relation_is_minimal(x, y, is_support=True)
        ):
            self.case_supported_by[y].append(x)

    def _find_supported_attacks_chained(self) -> None:
        # Loop over all the cases that are attacked by one or more other cases
        for case in self.case_attacked_by:
            # This will contain all the arguments that attack case via a
            # supported attack
            supported_attacks = set()

            # We want to look at all of the attackers of this case and see if
            # any other cases support the attacker, and thus there exists a
            # supported attack
            for attacker in self.case_attacked_by[case]:
                # For each attacker, keep track of the chain of supports
                # leading to it and maintain a stack to traverse to all the
                # supporting arguments to it.
                supporters = set()
                stack = list(self.case_supported_by[attacker])
                while stack:
                    supporter = stack.pop()
                    if supporter not in supporters:
                        supporters.add(supporter)
                        # Add the supporter's supporters to the stack if they
                        # haven't been added to the supporters set yet
                        stack.extend(
                            [
                                s
                                for s in self.case_supported_by[supporter]
                                if s not in supporters
                            ]
                        )

                # Here we have found all of the supported attacks stemming from
                # the current attacker's supporter chain
                supported_attacks.update(supporters)

            # Here we have found all of the supported attacks for this case
            self.case_supported_attackers[case] = list(supported_attacks)

    def _find_supported_attacks(self) -> None:
        # Loop over all the cases that are attacked by one or more other cases
        for case in self.case_attacked_by:
            # This will contain all the arguments that attack case via a
            # supported attack
            supported_attacks = set()

            # We want to look at all of the attackers of this case and see if
            # any other cases support the attacker, and thus there exists a
            # supported attack
            for attacker in self.case_attacked_by[case]:
                supported_attacks.update(self.case_supported_by[attacker])

            # Here we have found all of the supported attacks for this case
            self.case_supported_attackers[case] = list(supported_attacks)

    def _strictly_less_specific(self, x: Case, y: Case) -> bool:
        # Defines the < relation from the partial order relation
        return self.less_specific(x, y) and x.characterisation != y.characterisation

    def _relation_is_minimal(
        self, x: Case, y: Case, *, is_support: bool = False
    ) -> bool:
        for z in self.casebase:
            if x == z or y == z:
                continue

            # If we are not considering a support relation,
            # only consider cases when z has the same outcome as x
            if (not is_support) and z.different_outcome_to(x):
                continue

            # If there is another case that is closer to y than
            # x then this relation isn't minimal and return false.
            if self._strictly_less_specific(y, z) and self._strictly_less_specific(
                z, x
            ):
                return False

        return True

    def predict(
        self, new_cases: list[Case], aa_framework_filename: str | None = None
    ) -> list[int]:
        assert self.casebase is not None

        predictions = []
        for i, new_case in enumerate(new_cases):
            case_aa_framework_filename = None
            if aa_framework_filename:
                case_aa_framework_filename = f"{aa_framework_filename}_{i}"

            predictions.append(
                self._predict_single(new_case, case_aa_framework_filename)
            )

        return predictions

    def _predict_single(self, new_case: Case, aa_framework_filename: str | None) -> int:
        self._fit_new_case(new_case)

        if aa_framework_filename:
            self.draw_aa_framework(aa_framework_filename, "graphs")

        grounded_extension = self._compute_grounded_extension()

        # Ensure that the new case is removed from casebase and any attacks
        # that it has made are removed
        self._remove_new_case(new_case)

        if self.default_case in grounded_extension:
            return self.default_case.outcome
        else:
            # Assuming that the default outcome is 1 or 0
            return 1 - self.default_case.outcome

    def _remove_new_case(self, new_case: Case) -> None:
        self.casebase.remove(new_case)
        for case in self.casebase:
            if new_case in self.case_attacked_by[case]:
                self.case_attacked_by[case].remove(new_case)

    def _fit_new_case(self, new_case: Case) -> None:
        for case in self.casebase:
            # The new case attacks an already existing case if the new case is
            # irrelevant to it
            if not self.less_specific(case, new_case):
                self.case_attacked_by[case].append(new_case)

        self.casebase.append(new_case)

    def _compute_grounded_extension(self) -> set[Case]:
        attacks, _, supported_attacks = self._get_framework_relations()

        # Combine both the attacks and supported attacks in one list
        attacks += supported_attacks

        # Remaining cases to consider to be in grounded extension
        remaining = set(copy(self.casebase))
        assert len(remaining) == len(self.casebase)

        # Cases that are in the grounded extension
        in_ge = set()
        # Cases that are not in the grounded extension
        out_ge = set()
        while remaining:
            # Find all of the unattacked cases in for this iteration
            currently_attacked = set()
            for attacker, attackee in attacks:
                if (attacker in remaining) and (attackee in remaining):
                    currently_attacked.add(attackee)
            in_ge.update(remaining - currently_attacked)

            # Cases that are being attacked by a case in the grounded extension
            # is not part of the grounded extension
            for attacker, attackee in attacks:
                if (attacker in in_ge) and (attackee in currently_attacked):
                    out_ge.add(attackee)
            remaining.difference_update(in_ge | out_ge)

            if currently_attacked == remaining:
                break

        return in_ge

    def draw_aa_framework(self, file_name: str, output_dir: str) -> None:
        assert self.casebase is not None

        attacks, supports, supported_attacks = self._get_framework_relations()

        graph = Digraph("AA-Framework", filename=file_name, format="png")
        graph.attr(rankdir="TB")

        for case in self.casebase:
            graph.node(str(hash(case)), str(case))
        for attack in attacks:
            graph.edge(str(hash(attack[0])), str(hash(attack[1])))
        for support in supports:
            graph.edge(
                str(hash(support[0])),
                str(hash(support[1])),
                color="green:invis:green",
                arrowhead="empty",
                arrowsize="1.5",
            )
        for supported_attack in supported_attacks:
            graph.edge(
                str(hash(supported_attack[0])),
                str(hash(supported_attack[1])),
                style="dashed",
            )

        graph.render(directory=output_dir, cleanup=True)

    def _get_framework_relations(
        self,
    ) -> tuple[
        list[tuple[Case, Case]], list[tuple[Case, Case]], list[tuple[Case, Case]]
    ]:
        attacks, supports, supported_attacks = [], [], []
        for case in self.casebase:
            list.extend(
                attacks, [(case, attacker) for attacker in self.case_attacked_by[case]]
            )

            list.extend(
                supports,
                [(case, supporter) for supporter in self.case_supported_by[case]],
            )

            list.extend(
                supported_attacks,
                [
                    (case, supported_attacker)
                    for supported_attacker in self.case_supported_attackers[case]
                ],
            )

        return attacks, supports, supported_attacks


if __name__ == "__main__":
    subset_relation = lambda x, y: x.characterisation <= y.characterisation
    default_case = Case("default", set(), 0, is_default=True)
    casebase = [
        Case(0, {"a"}, 1),
        Case(1, {"a", "c"}, 2),
        Case(2, {"a", "b"}, 1),
        Case(3, {"a", "b", "c"}, 0),
    ]

    aacbr = AACBR(
        subset_relation,
        default_case,
        include_supports=True,
        supported_attack_chain=False,
    )
    aacbr.fit(casebase)
    aacbr.draw_aa_framework("non_boolean_outcomes_supports", "graphs")

    # new_cases = [
    #     Case(4, {"a", "d", "b"}),
    #     Case(5, {"a", "c", "d"})
    # ]
    # assert all([new_case.is_new_case for new_case in new_cases])

    # print(f"New case prediction: {aacbr.predict(new_cases, "predicted_case")}")
